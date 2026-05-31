# This file was mechanically split from malca.review.app; preserve behavior when editing.
@app.callback(
    [Output('taxonomy-selection-store', 'data', allow_duplicate=True),
     Output('active-taxonomy-menu', 'data', allow_duplicate=True),
     Output('taxonomy-submenu-store', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True)],
    [Input({'type': 'taxonomy-primary-btn', 'value': ALL}, 'n_clicks'),
     Input('taxonomy-hypothesis-btn', 'n_clicks')],
    [State('taxonomy-selection-store', 'data'),
     State('active-taxonomy-menu', 'data')],
    prevent_initial_call=True,
)
def click_taxonomy_primary(_primary_clicks, _hypothesis_clicks, selection, active_menu):
    triggered = callback_context.triggered_id
    if not triggered:
        return no_update, no_update, no_update, no_update
    selection = selection_from_review(selection if isinstance(selection, dict) else {})
    if triggered == 'taxonomy-hypothesis-btn':
        if active_menu == 'physical_primary':
            return selection, '', '', 'Hypothesis menu closed'
        return selection, 'physical_primary', '', 'Hypothesis menu'
    if not isinstance(triggered, dict) or triggered.get('type') != 'taxonomy-primary-btn':
        return no_update, no_update, no_update, no_update
    value = str(triggered.get('value') or '')
    if selection.get('morphology_primary') == value:
        selection['morphology_primary'] = None
        selection['morphology_secondary'] = None
        return selection, '', '', 'Morphology cleared'
    selection['morphology_primary'] = value
    selection['morphology_secondary'] = None
    return selection, 'morphology_secondary', value, f"Morphology: {label_for(value)}"


@app.callback(
    [Output('taxonomy-selection-store', 'data', allow_duplicate=True),
     Output('active-taxonomy-menu', 'data', allow_duplicate=True),
     Output('taxonomy-submenu-store', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True)],
    Input({'type': 'taxonomy-option-btn', 'menu': ALL, 'value': ALL}, 'n_clicks'),
    [State('taxonomy-selection-store', 'data'),
     State('active-taxonomy-menu', 'data')],
    prevent_initial_call=True,
)
def click_taxonomy_option(_option_clicks, selection, active_menu):
    triggered = callback_context.triggered_id
    if not isinstance(triggered, dict) or triggered.get('type') != 'taxonomy-option-btn':
        return no_update, no_update, no_update, no_update
    menu = str(triggered.get('menu') or active_menu or '')
    value = str(triggered.get('value') or '')
    selection = selection_from_review(selection if isinstance(selection, dict) else {})
    if menu == 'morphology_secondary':
        selection['morphology_secondary'] = None if selection.get('morphology_secondary') == value else value
        submenu = selection.get('morphology_primary') or ''
        return selection, 'morphology_secondary', submenu, f"Detail: {label_for(selection.get('morphology_secondary') or 'cleared')}"
    if menu == 'physical_primary':
        if selection.get('physical_primary') == value:
            selection['physical_primary'] = None
            selection['physical_secondary'] = None
            return selection, '', '', 'Hypothesis cleared'
        selection['physical_primary'] = value
        selection['physical_secondary'] = None
        subclasses = TAXONOMY_KEYBOARD_PAYLOAD['physical_secondary'].get(value, [])
        if subclasses:
            return selection, 'physical_secondary', value, f"Hypothesis: {label_for(value)}"
        return selection, '', '', f"Hypothesis: {label_for(value)}"
    if menu == 'physical_secondary':
        selection['physical_secondary'] = None if selection.get('physical_secondary') == value else value
        submenu = selection.get('physical_primary') or ''
        return selection, 'physical_secondary', submenu, f"Subclass: {label_for(selection.get('physical_secondary') or 'cleared')}"
    return no_update, no_update, no_update, no_update


@app.callback(
    [Output({'type': 'taxonomy-primary-btn', 'value': ALL}, 'className'),
     Output('taxonomy-hypothesis-btn', 'className'),
     Output('taxonomy-summary', 'children')],
    [Input('taxonomy-selection-store', 'data'),
     Input('active-taxonomy-menu', 'data')],
    prevent_initial_call=False,
)
def render_taxonomy_state(selection, active_menu):
    selection = selection_from_review(selection if isinstance(selection, dict) else {})
    primary = selection.get('morphology_primary')
    primary_classes = [
        'badge-btn active' if item['value'] == primary else 'badge-btn'
        for item in MORPHOLOGY_PRIMARY
    ]
    hypothesis_class = 'badge-btn active' if active_menu in {'physical_primary', 'physical_secondary'} or selection.get('physical_primary') else 'badge-btn'
    parts = []
    if primary:
        detail = selection.get('morphology_secondary')
        parts.append(f"Morphology: {label_for(primary)}" + (f" / {label_for(detail)}" if detail else ""))
    if selection.get('physical_primary'):
        family = selection.get('physical_primary')
        subclass = selection.get('physical_secondary')
        parts.append(f"Hypothesis: {label_for(family)}" + (f" / {label_for(subclass)}" if subclass else ""))
    return primary_classes, hypothesis_class, ' | '.join(parts) if parts else 'No taxonomy selection'


@app.callback(
    Output('taxonomy-submenu-panel', 'children'),
    [Input('active-taxonomy-menu', 'data'),
     Input('taxonomy-submenu-store', 'data'),
     Input('taxonomy-selection-store', 'data')],
    prevent_initial_call=False,
)
def render_taxonomy_submenu(active_menu, submenu, selection):
    selection = selection_from_review(selection if isinstance(selection, dict) else {})
    active_menu = str(active_menu or '')
    if active_menu == 'morphology_secondary':
        options = TAXONOMY_KEYBOARD_PAYLOAD['morphology_secondary'].get(str(submenu or selection.get('morphology_primary') or ''), [])
        active_value = selection.get('morphology_secondary')
        title = 'Detail'
    elif active_menu == 'physical_primary':
        options = TAXONOMY_KEYBOARD_PAYLOAD['physical_primary']
        active_value = selection.get('physical_primary')
        title = 'Hypothesis'
    elif active_menu == 'physical_secondary':
        options = TAXONOMY_KEYBOARD_PAYLOAD['physical_secondary'].get(str(submenu or selection.get('physical_primary') or ''), [])
        active_value = selection.get('physical_secondary')
        title = 'Subclass'
    else:
        return []
    if not options:
        return []
    return [
        html.Span(f"{title}: ", style={'color': '#aaa', 'margin-right': '4px', 'font-size': '11px'}),
        *[
            html.Button(
                f'[{item["key"].upper()}] {item["label"]}',
                id={'type': 'taxonomy-option-btn', 'menu': active_menu, 'value': item['value']},
                n_clicks=0,
                className='badge-btn active' if item['value'] == active_value else 'badge-btn',
            )
            for item in options
        ],
    ]


app.clientside_callback(
    """
    function(activeClass) {
        var active = activeClass || 'unclassified';
        return ['dipper', 'microlensing', 'flare', 'ltv', 'unknown_interesting', 'instrumental', 'other']
            .map(function(tag) { return tag === active ? 'badge-btn active' : 'badge-btn'; });
    }
    """,
    [Output(f'class-badge-{tag}', 'className') for tag in CLASS_BADGE_TAGS],
    Input('event-class-store', 'data'),
    prevent_initial_call=False,
)


app.clientside_callback(
    """
    function() {
        var no = window.dash_clientside.no_update;
        var triggered = window.dash_clientside.callback_context.triggered || [];
        if (!triggered.length) {
            return [no, no];
        }
        var triggerId = String(triggered[0].prop_id || '').split('.')[0];
        if (!triggerId.startsWith('class-badge-')) {
            return [no, no];
        }
        var tag = triggerId.replace('class-badge-', '');
        var active = arguments[arguments.length - 1] || 'unclassified';
        if (active === tag) {
            return ['unclassified', 'Class: unclassified'];
        }
        return [tag, 'Class: ' + tag];
    }
    """,
    [Output('event-class-store', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True)],
    [Input(f'class-badge-{tag}', 'n_clicks') for tag in CLASS_BADGE_TAGS],
    State('event-class-store', 'data'),
    prevent_initial_call=True,
)


app.clientside_callback(
    """
    function(prefix) {
        if (!prefix) {
            return '';
        }
        return '[' + String(prefix).toUpperCase() + '] ...';
    }
    """,
    Output('prefix-indicator', 'children'),
    Input('pending-prefix', 'data'),
    prevent_initial_call=False,
)


# Expand/collapse all candidate metadata panels
