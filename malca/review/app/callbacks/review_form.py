# This file was mechanically split from malca.review.app; preserve behavior when editing.
@app.callback(
    [Output('event-class-store', 'data'),
     Output('needs-followup-store', 'data'),
     Output('review-pass-store', 'data'),
     Output('notes', 'value'),
     Output('current-score', 'data'),
     Output('taxonomy-selection-store', 'data'),
     Output('active-taxonomy-menu', 'data'),
     Output('taxonomy-submenu-store', 'data')],
    Input('current-candidate-id', 'data'),
    State('queue-size-store', 'data'),
    prevent_initial_call=False
)
def load_review_form(candidate_id, queue_size):
    """Load existing review for current candidate into stores."""
    if not candidate_id or int(queue_size or 0) == 0:
        return 'unclassified', False, 1, '', None, {}, '', ''

    with closing(db_connect(Path(DB_PATH))) as conn:
        review = get_review(conn, str(candidate_id))

    # Coerce legacy/unknown classes into the current tag set.
    allowed_classes = {'unclassified'} | set(CLASS_KEY_MAP.values())
    event_class = (review.get('event_class') or 'unclassified')
    if event_class not in allowed_classes:
        event_class = 'other'

    selection = selection_from_review(review)

    return (
        event_class,
        review.get('workflow_status', 'unreviewed') == 'needs_followup',
        review.get('review_pass', 1),
        review.get('notes', ''),
        review.get('interest_score'),
        selection,
        '',
        '',
    )


app.clientside_callback(
    """
    function(n1, n2, n3, n4, queueSize, candidateId, taxonomySelection, needsFollowup, notes, saveRequest) {
        var no = window.dash_clientside.no_update;
        if (!candidateId || parseInt(queueSize == null ? 0 : queueSize, 10) <= 0) {
            var triggered = window.dash_clientside.callback_context.triggered || [];
            if (!triggered.length) {
                return [no, no, no];
            }
            return [no, 'Queue is empty', no];
        }

        var triggered = window.dash_clientside.callback_context.triggered || [];
        if (!triggered.length) {
            return [no, no, no];
        }
        var triggerId = String(triggered[0].prop_id || '').split('.')[0];
        if (!triggerId.startsWith('score-')) {
            return [no, no, no];
        }

        var score = parseInt(triggerId.split('-')[1], 10);
        if (!Number.isFinite(score)) {
            return [no, no, no];
        }

        var nextNonce = 1;
        if (saveRequest && typeof saveRequest === 'object' && typeof saveRequest.nonce === 'number') {
            nextNonce = saveRequest.nonce + 1;
        }

        return [
            score,
            '✓ Confidence: ' + String(score),
            {
                nonce: nextNonce,
                candidate_id: String(candidateId),
                score: score,
                taxonomy: taxonomySelection || {},
                needs_followup: !!needsFollowup,
                notes: notes || '',
                increment_pass: false,
                event_type: 'button',
            }
        ];
    }
    """,
    [Output('current-score', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True),
     Output('review-save-request', 'data', allow_duplicate=True)],
    [Input(f'score-{i}', 'n_clicks') for i in range(1, 5)],
    [State('queue-size-store', 'data'),
     State('current-candidate-id', 'data'),
     State('taxonomy-selection-store', 'data'),
     State('needs-followup-store', 'data'),
     State('notes', 'value'),
     State('review-save-request', 'data')],
    prevent_initial_call=True,
)


# Save button
@app.callback(
    [Output('notification', 'children', allow_duplicate=True),
     Output('review-pass-store', 'data', allow_duplicate=True)],
    Input('save-btn', 'n_clicks'),
    [State('queue-size-store', 'data'),
     State('current-candidate-id', 'data'),
     State('current-score', 'data'),
     State('taxonomy-selection-store', 'data'),
     State('needs-followup-store', 'data'),
     State('notes', 'value')],
    prevent_initial_call=True
)
def save_review_callback(n_clicks, queue_size, candidate_id, score,
                         taxonomy_selection, needs_followup, notes):
    """Save review."""
    if not n_clicks or int(queue_size or 0) <= 0 or not candidate_id:
        return no_update, no_update

    new_pass, _ = _do_save(
        str(candidate_id), score, taxonomy_selection, needs_followup, notes, 'save_button',
    )

    return "✓ Saved", new_pass


# Back button (previous candidate)
@app.callback(
    [Output('current-index', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True)],
    Input('back-btn', 'n_clicks'),
    State('current-index', 'data'),
    prevent_initial_call=True
)
def back_callback(n_clicks, idx):
    """Go to previous candidate."""
    if not n_clicks:
        return no_update, no_update
    new_idx = max(0, (idx or 0) - 1)
    if new_idx == idx:
        return no_update, "Already at first candidate"
    return new_idx, "← Previous"


# Done button (save + next)
@app.callback(
    [Output('current-index', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True),
     Output('review-pass-store', 'data', allow_duplicate=True)],
    Input('done-btn', 'n_clicks'),
    [State('current-index', 'data'),
     State('queue-size-store', 'data'),
     State('current-candidate-id', 'data'),
     State('current-score', 'data'),
     State('taxonomy-selection-store', 'data'),
     State('needs-followup-store', 'data'),
     State('notes', 'value')],
    prevent_initial_call=True
)
def done_callback(n_clicks, idx, queue_size, candidate_id, score,
                  taxonomy_selection, needs_followup, notes):
    """Save and go to next."""
    if not n_clicks or int(queue_size or 0) <= 0 or not candidate_id:
        return no_update, no_update, no_update

    if score is None:
        return no_update, "⚠ Confidence required", no_update

    taxonomy_selection = taxonomy_selection if isinstance(taxonomy_selection, dict) else {}
    if not taxonomy_selection.get('morphology_primary'):
        return no_update, "⚠ Morphology required", no_update

    new_pass, _ = _do_save(
        str(candidate_id), score, taxonomy_selection, needs_followup, notes, 'done_button',
        increment_pass=True,
    )

    queue_size = int(queue_size or 0)
    new_idx = min(idx + 1, queue_size - 1)

    return new_idx, "✓ Saved + Next →", new_pass


# --- Display callbacks for stores → visible indicators ---

app.clientside_callback(
    """
    function(currentScore) {
        var score = parseInt(currentScore, 10);
        if (!Number.isFinite(score) || [1, 2, 3, 4].indexOf(score) === -1) {
            score = null;
        }
        var out = [];
        for (var i = 1; i <= 4; i += 1) {
            out.push(i === score ? 'score-btn active' : 'score-btn');
        }
        return out;
    }
    """,
    [Output(f'score-{i}', 'className') for i in range(1, 5)],
    Input('current-score', 'data'),
    prevent_initial_call=False,
)


