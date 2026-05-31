# This file was mechanically split from malca.review.app; preserve behavior when editing.
@app.callback(
    [Output('sidebar', 'className'),
     Output('sidebar-toggle', 'className'),
     Output('sidebar-state', 'data')],
    [Input('sidebar-toggle', 'n_clicks'),
     Input('keyboard-input', 'value')],
    [State('sidebar-state', 'data'),
     State('active-taxonomy-menu', 'data')],
    prevent_initial_call=True
)
def toggle_sidebar(n_clicks, key_value, is_expanded, active_taxonomy_menu):
    """Toggle sidebar visibility."""
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update

    trigger = ctx.triggered[0]['prop_id']

    # Check if Escape was pressed
    key = _keyboard_key(key_value)
    if 'keyboard-input' in trigger and key == 'Escape':
        if active_taxonomy_menu:
            return no_update, no_update, no_update
        is_expanded = not is_expanded

    # Check if toggle button was clicked
    elif 'sidebar-toggle' in trigger and n_clicks:
        is_expanded = not is_expanded
    else:
        return no_update, no_update, no_update

    sidebar_class = 'sidebar expanded' if is_expanded else 'sidebar'
    toggle_class = 'sidebar-toggle sidebar-expanded' if is_expanded else 'sidebar-toggle'
    return sidebar_class, toggle_class, is_expanded


# --- All filter State components used by load_queue -------------------------
# Auto-generated from _SIDEBAR_GROUPS so the UI and callback always stay in sync.
_BOOL_MODE_STATES: list[tuple[str, str]] = []
_NUM_COLUMNS: list[str] = []
_NUM_INPUT_STATES: list[tuple[dict[str, str], str]] = []
_TEXT_STATES: list[tuple[str, str]] = []
_SELECT_STATES: list[tuple[str, str]] = []
_QUEUE_FILTER_APP_STATE_KEY = "dash_queue_filter_ui_state_v1"
_REVIEW_GUI_STATE_APP_STATE_KEY = "dash_review_gui_state_v1"

for _grp_name, _grp_items in _SIDEBAR_GROUPS:
    for _ftype, _col in _grp_items:
        _cid = _col_id(_col)
        if _ftype == 'bool':
            _BOOL_MODE_STATES.append((f'{_cid}-mode', f'{_col}_mode'))
        elif _ftype == 'num':
            _NUM_COLUMNS.append(_col)
            _NUM_INPUT_STATES.append(({'type': 'num-filter-min-input', 'col': _col}, f'min_{_col}'))
            _NUM_INPUT_STATES.append(({'type': 'num-filter-max-input', 'col': _col}, f'max_{_col}'))
        elif _ftype == 'text':
            _TEXT_STATES.append((f'filter-{_cid}', _col))
        elif _ftype == 'select':
            _SELECT_STATES.append((f'exclude-{_cid}', f'exclude_{_col}'))

# Build the callback I/O lists dynamically.
_FILTER_VALUE_STATES = (
    [State('filter-unreviewed', 'value'),
     State('filter-failed', 'value'),
     State('select-filter-mode', 'value')]
    + [State(cid, 'value') for cid, _ in _BOOL_MODE_STATES]
    + [State(cid, 'value') for cid, _ in _NUM_INPUT_STATES]
    + [State(cid, 'value') for cid, _ in _TEXT_STATES]
    + [State(cid, 'value') for cid, _ in _SELECT_STATES]
    + [State('sort-col', 'value'),
       State('sort-desc', 'value')]
)

_FILTER_VALUE_INPUTS = (
    [Input('filter-unreviewed', 'value'),
     Input('filter-failed', 'value'),
     Input('select-filter-mode', 'value')]
    + [Input(cid, 'value') for cid, _ in _BOOL_MODE_STATES]
    + [Input(cid, 'value') for cid, _ in _NUM_INPUT_STATES]
    + [Input(cid, 'value') for cid, _ in _TEXT_STATES]
    + [Input(cid, 'value') for cid, _ in _SELECT_STATES]
    + [Input('sort-col', 'value'),
       Input('sort-desc', 'value')]
)

_FILTER_VALUE_OUTPUTS = (
    [Output('filter-unreviewed', 'value', allow_duplicate=True),
     Output('filter-failed', 'value', allow_duplicate=True),
     Output('select-filter-mode', 'value', allow_duplicate=True)]
    + [Output(cid, 'value', allow_duplicate=True) for cid, _ in _BOOL_MODE_STATES]
    + [Output(cid, 'value', allow_duplicate=True) for cid, _ in _NUM_INPUT_STATES]
    + [Output(cid, 'value', allow_duplicate=True) for cid, _ in _TEXT_STATES]
    + [Output(cid, 'value', allow_duplicate=True) for cid, _ in _SELECT_STATES]
    + [Output('sort-col', 'value', allow_duplicate=True),
       Output('sort-desc', 'value', allow_duplicate=True)]
)

_queue_states = _FILTER_VALUE_STATES
_TEXT_OPTION_STATES = [State(cid, 'options') for cid, _ in _TEXT_STATES]
_SELECT_OPTION_STATES = [State(cid, 'options') for cid, _ in _SELECT_STATES]
_TEXT_OPTION_INPUTS = [Input(cid, 'options') for cid, _ in _TEXT_STATES]
_SELECT_OPTION_INPUTS = [Input(cid, 'options') for cid, _ in _SELECT_STATES]
_TEXT_VALUE_STATES = [State(cid, 'value') for cid, _ in _TEXT_STATES]
_SELECT_VALUE_STATES = [State(cid, 'value') for cid, _ in _SELECT_STATES]


