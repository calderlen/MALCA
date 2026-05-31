# This file was mechanically split from malca.review.app; preserve behavior when editing.
@app.callback(
    [Output({'type': 'meta-details', 'group': ALL}, 'open'),
     Output('toggle-meta-all', 'children')],
    Input('toggle-meta-all', 'n_clicks'),
    State({'type': 'meta-details', 'group': ALL}, 'open'),
    prevent_initial_call=True,
)
def toggle_all_metadata_panels(n_clicks, open_states):
    _ = n_clicks
    if not open_states:
        return [], no_update

    any_closed = any(not bool(v) for v in open_states)
    new_open = True if any_closed else False
    label = 'Collapse all' if new_open else 'Expand all'
    return [new_open for _ in open_states], label


app.clientside_callback(
    """
    function(needsFollowup) {
        return needsFollowup ? '[,] Followup: ON' : '[,] Followup: off';
    }
    """,
    Output('followup-indicator', 'children'),
    Input('needs-followup-store', 'data'),
    prevent_initial_call=False,
)


def _format_hms(total_seconds: float) -> str:
    seconds = max(0, int(total_seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


@app.callback(
    Output('review-session-start', 'data'),
    Input('queue-filter-hash-store', 'data'),
    State('review-session-start', 'data'),
    prevent_initial_call=False,
)
def sync_review_session_start(queue_hash, session_start):
    """Reset session-timer origin when the active queue changes."""
    now = time.time()
    if not isinstance(session_start, dict):
        return {'ts': now, 'filter_hash': queue_hash}

    if session_start.get('filter_hash') != queue_hash:
        return {'ts': now, 'filter_hash': queue_hash}

    ts = session_start.get('ts')
    if ts is None:
        return {'ts': now, 'filter_hash': queue_hash}

    return session_start


@app.callback(
    Output('review-progress-state', 'data'),
    Input('review-db-scope', 'data'),
    Input('queue-data', 'modified_timestamp'),
    Input('review-pass-store', 'data'),
    Input('import-trigger', 'data'),
    prevent_initial_call=False,
)
def load_review_progress_state(_db_scope, _queue_data_ts, _review_pass, _import_trigger):
    """Load reviewed/total counts for the progress indicator without interval polling."""
    reviewed, total = _progress_counts()
    return {'reviewed': int(reviewed), 'total': int(total)}


app.clientside_callback(
    """
    function(reviewPass) {
        return 'Pass: ' + String(reviewPass || 1);
    }
    """,
    Output('pass-indicator', 'children'),
    Input('review-pass-store', 'data'),
    prevent_initial_call=False,
)


app.clientside_callback(
    """
    function(needsFollowup, score, candidateId, queueSize) {
        if (!candidateId || parseInt(queueSize == null ? 0 : queueSize, 10) <= 0) {
            return 'Status: —';
        }
        if (needsFollowup) {
            return 'Status: needs_followup';
        }
        if (score !== null && score !== undefined && score !== '') {
            return 'Status: reviewed';
        }
        return 'Status: unreviewed';
    }
    """,
    Output('status-indicator', 'children'),
    Input('needs-followup-store', 'data'),
    Input('current-score', 'data'),
    [State('current-candidate-id', 'data'),
     State('queue-size-store', 'data')],
    prevent_initial_call=False,
)

# Auto-populate import candidates from run directory inferred via plot directory
