# This file was mechanically split from malca.review.app; preserve behavior when editing.
@app.callback(
    Output('sidebar-status', 'children', allow_duplicate=True),
    Input('export-btn', 'n_clicks'),
    [State('export-path', 'value'),
     State('export-only-reviewed', 'value')],
    prevent_initial_call=True
)
def export_reviews_callback(n_clicks, export_path, only_reviewed):
    """Export reviews to file."""
    if not n_clicks or not export_path:
        return no_update

    try:
        with closing(db_connect(Path(DB_PATH))) as conn:
            out_path = Path(export_path).expanduser()
            only_reviewed_flag = 'yes' in (only_reviewed or [])
            export_reviews(conn, out_path, only_reviewed=only_reviewed_flag)
            reviewed_text = " (reviewed only)" if only_reviewed_flag else ""
            return f"✓ Exported to {out_path.name}{reviewed_text}"
    except Exception as e:
        return f"✗ Export failed: {str(e)}"


@app.callback(
    Output('sidebar-status', 'children', allow_duplicate=True),
    Input('export-review-sync-btn', 'n_clicks'),
    [State('review-sync-dir', 'value'),
     State('review-sync-hash-assets', 'value')],
    prevent_initial_call=True
)
def export_review_sync_callback(n_clicks, out_dir, hash_assets):
    """Export the Git-trackable review bundle."""
    if not n_clicks:
        return no_update

    target_dir = Path(str(out_dir or "reviews")).expanduser()
    result = auto_export_review_bundle(
        Path(DB_PATH),
        target_dir,
        hash_assets='yes' in (hash_assets or []),
        logger=lambda _message: None,
    )
    if not result.get("ok"):
        return f"✗ Git bundle export failed: {result.get('error', 'unknown error')}"
    return (
        f"✓ Exported Git bundle to {result['out_dir']} "
        f"({result['candidates_exported']} candidates, {result['reviews_exported']} reviews)"
    )


@app.callback(
    Output('sidebar-status', 'children', allow_duplicate=True),
    Input('merge-review-db-btn', 'n_clicks'),
    State('merge-target-db-path', 'value'),
    prevent_initial_call=True,
)
def merge_review_db_callback(n_clicks, target_db_path):
    """Merge the current review DB into another review DB."""
    if not n_clicks or not target_db_path:
        return no_update

    try:
        source_db = Path(DB_PATH).expanduser().resolve()
        target_db = Path(str(target_db_path)).expanduser().resolve()
        result = merge_review_databases(source_db, target_db, only_reviewed=True)
        with closing(db_connect(Path(DB_PATH))) as conn:
            save_app_state(conn, 'last_merge_target_db', str(target_db))
        return (
            f"✓ Merged into {target_db.name} | "
            f"reviews inserted={result['reviews_inserted']}, updated={result['reviews_updated']}, "
            f"skipped={result['reviews_skipped']}"
        )
    except Exception as e:
        return f"✗ Merge failed: {str(e)}"


# Help modal
@app.callback(
    Output("help-modal", "is_open"),
    [Input("help-link", "n_clicks"),
     Input("close-help", "n_clicks"),
     Input("keyboard-input", "value")],
    State("help-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_help_modal(n1, n2, key_value, is_open):
    """Toggle help modal."""
    ctx = callback_context
    if not ctx.triggered:
        return is_open

    trigger = ctx.triggered[0]['prop_id']

    if 'keyboard-input' in trigger:
        if _keyboard_key(key_value) == '?':
            return not is_open
        return is_open

    if n1 or n2:
        return not is_open
    return is_open


# Static file server
@app.server.route('/plots/<path:filename>')
def serve_plot(filename):
    """Serve plot images."""
    suffix = Path(str(filename)).suffix.lower()
    if suffix not in _PLOT_STATIC_EXTENSIONS:
        abort(404)
    return send_from_directory(str(_plot_asset_root()), filename)


# Reset queue to beginning
@app.callback(
    [Output('current-index', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True)],
    Input('reset-queue-btn', 'n_clicks'),
    prevent_initial_call=True
)
def reset_queue_position(n_clicks):
    """Reset queue position to the beginning (index 0)."""
    if not n_clicks:
        return no_update, no_update
    return 0, "Queue position reset to beginning."


