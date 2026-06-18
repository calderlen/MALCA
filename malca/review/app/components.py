# This file was mechanically split from malca.review.app; preserve behavior when editing.
def _candidate_lookup_keys(candidate_id: str, payload: dict) -> list[str]:
    keys = [str(candidate_id)]
    for key in ("candidate_id", "asas_sn_id"):
        v = payload.get(key)
        if v is not None:
            keys.append(str(v))
    path_v = payload.get("path")
    if path_v:
        keys.append(Path(str(path_v)).stem)
    seen = set()
    return [k for k in keys if k and not (k in seen or seen.add(k))]


def _render_external_followup(
    payload: dict,
    candidate_id: str,
    theme: str | None = None,
    plot_dir: str | Path | None = None,
    selected_cutout_survey: str | None = None,
) -> list:
    theme_spec = _external_followup_theme(theme)
    card_style = {
        **theme_spec["card_style"],
        'borderRadius': '4px',
        'padding': '5px 7px',
        'minWidth': 0,
        'lineHeight': 1.2,
    }
    section_title_style = {
        'fontSize': '11px',
        'fontWeight': '700',
        'lineHeight': 1.1,
        'marginBottom': '3px',
    }
    metrics_style = {
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fit, minmax(118px, 1fr))',
        'gap': '2px 8px',
    }
    metric_style = {
        'display': 'flex',
        'gap': '4px',
        'minWidth': 0,
        'fontSize': '10px',
        'lineHeight': 1.25,
        'alignItems': 'baseline',
    }
    metric_label_style = {'color': theme_spec["muted"], 'flex': '0 0 auto'}
    metric_value_style = {
        'minWidth': 0,
        'overflowWrap': 'anywhere',
    }
    muted_text_style = {'fontSize': '9px', 'lineHeight': 1.2, 'color': theme_spec["muted"]}
    error_text_style = {'fontSize': '10px', 'color': theme_spec["error"]}
    run_dir = _resolve_run_dir_from_plot_dir(plot_dir if plot_dir is not None else PLOT_DIR) or _run_dir_from_source_path()
    lookup_keys = _candidate_lookup_keys(candidate_id, payload)

    def _fmt_ms(value, digits: int = 3) -> str:
        try:
            if value is None or pd.isna(value):
                return "n/a"
            return f"{float(value):.{digits}g}"
        except Exception:
            text = str(value or "").strip()
            return text if text else "n/a"

    def _metric(label: str, value) -> html.Div:
        return html.Div([
            html.Span(f"{label}:", style=metric_label_style),
            html.Span(str(value), style=metric_value_style, title=str(value)),
        ], style=metric_style)

    def _section(title: str, metrics: list, extras: list | None = None) -> html.Div:
        children = [
            html.Div(title, style=section_title_style),
            html.Div(metrics, style=metrics_style),
        ]
        if extras:
            children.extend(extras)
        style = dict(card_style)
        if extras:
            style['gridColumn'] = '1 / -1'
        return html.Div(children, style=style)

    cutout_data = cutout_payload_for_candidate(payload, selected_key=selected_cutout_survey)
    cutout_has_coords = bool(cutout_data.get("has_coordinates"))
    cutout_card_style = dict(card_style)
    cutout_card_style.update({'gridColumn': '1 / -1', 'padding': '6px 7px 7px 7px'})
    cutout_viewer_class = "cutout-viewer" if cutout_has_coords else "cutout-viewer cutout-viewer-empty"
    cutout_status_style = muted_text_style if cutout_has_coords else error_text_style
    try:
        fwhm_overlay_fraction = float(cutout_data.get("asassn_fwhm_overlay_fraction") or 0.0)
    except (TypeError, ValueError):
        fwhm_overlay_fraction = 0.0
    fwhm_overlay_percent = max(0.0, fwhm_overlay_fraction * 100.0)
    fwhm_overlay_style = {
        'width': f"{fwhm_overlay_percent:.4g}%",
        'height': f"{fwhm_overlay_percent:.4g}%",
    }
    if not cutout_has_coords:
        fwhm_overlay_style['display'] = 'none'
    cutout_card = html.Div([
        html.Div([
            html.Div('Survey Cutout', style={**section_title_style, 'marginBottom': '0'}),
            html.A(
                'Open source image',
                id='cutout-source-link',
                href=str(cutout_data.get("source_url") or "#"),
                target='_blank',
                rel='noopener noreferrer',
                title=str(cutout_data.get("source_url") or ""),
                className='cutout-source-link',
            ),
        ], className='cutout-card-header'),
        html.Div([
            dcc.Dropdown(
                id='cutout-survey-select',
                options=available_cutout_options(),
                value=str(cutout_data.get("selected_key") or DEFAULT_CUTOUT_SURVEY_KEY),
                clearable=False,
                searchable=True,
                disabled=not cutout_has_coords,
                className='cutout-survey-select dash-dropdown',
            ),
            html.Div(
                str(cutout_data.get("message") or ""),
                id='cutout-status',
                className='cutout-status',
                style=cutout_status_style,
            ),
        ], className='cutout-controls-row'),
        html.Div([
            html.Img(
                id='cutout-image',
                src=str(cutout_data.get("image_url") or ""),
                alt='Candidate survey cutout',
                className='cutout-image',
            ),
            html.Div(
                id='cutout-asassn-fwhm-overlay',
                className='cutout-asassn-fwhm-overlay',
                style=fwhm_overlay_style,
            ),
            html.Div(
                'No coordinates',
                className='cutout-empty-label',
            ),
        ], className=cutout_viewer_class),
    ], style=cutout_card_style, className='survey-cutout-card')

    multi_survey_card = _section(
        'Multi-survey Features',
        [
            _metric("Status", payload.get('ms_feature_status') or 'missing'),
            _metric("Event", f"{payload.get('ms_event_type') or 'n/a'} @ {_fmt_ms(payload.get('ms_event_t0_jd'), 7)}"),
            _metric(
                "ZTF g-r delta",
                f"{_fmt_ms(payload.get('ms_ztf_gr_delta'))} ({payload.get('ms_ztf_gr_event_pairs', 0)} event pairs)",
            ),
            _metric("NEOWISE W1 delta", _fmt_ms(payload.get('ms_neowise_w1_delta'))),
            _metric("TESS delta F/F", _fmt_ms(payload.get('ms_tess_flux_frac_delta'))),
            _metric("Gaia G delta", _fmt_ms(payload.get('ms_gaia_epoch_g_delta'))),
        ],
    )

    # Spectra
    has_spectrum = _coerce_bool(payload.get('has_spectrum'))
    spectrum_sources = str(payload.get('spectrum_sources') or '').strip()
    spectrum_links_raw = str(payload.get('spectrum_links') or '').strip()
    spectrum_links = [x.strip() for x in spectrum_links_raw.replace(';', ',').split(',') if x.strip()]
    spectra_rows = pd.DataFrame()
    for key in lookup_keys:
        spectra_rows = _load_spectra_rows(key, run_dir)
        if not spectra_rows.empty:
            break

    spectra_metrics = [
        _metric("Has spectra", 'yes' if has_spectrum else 'no'),
        _metric("Sources", spectrum_sources or 'none'),
    ]
    spectra_extras = []
    if not spectra_rows.empty:
        spectra_rows = spectra_rows.head(12)
        hdr = html.Tr([html.Th('survey'), html.Th('sep"'), html.Th('z'), html.Th('type'), html.Th('link')])
        body = []
        for _, r in spectra_rows.iterrows():
            link_val = str(r.get('link', '') or '').strip()
            link_cell = html.Td(
                html.A('view', href=link_val, target='_blank', rel='noopener noreferrer',
                       style={'color': theme_spec["muted"], 'fontSize': '9px'})
            ) if link_val else html.Td('')
            z_val = r.get('spectrum_redshift')
            z_text = f"{float(z_val):.4f}" if pd.notna(z_val) else ''
            body.append(html.Tr([
                html.Td(str(r.get('survey', ''))),
                html.Td(f"{float(r.get('sep_arcsec')):.2f}" if pd.notna(r.get('sep_arcsec')) else ''),
                html.Td(z_text),
                html.Td(str(r.get('spectrum_spectral_type', '') or '')[:20]),
                link_cell,
            ]))
        spectra_extras.append(html.Table([html.Thead(hdr), html.Tbody(body)], style={'width': '100%', 'fontSize': '10px', 'marginTop': '4px'}))
    if spectrum_links:
        spectra_extras.append(
            html.Div([
                html.Div('Links:', style={'fontSize': '10px', 'marginTop': '4px'}),
                html.Div([
                    html.A(
                        link,
                        href=link,
                        target='_blank',
                        rel='noopener noreferrer',
                        style={'display': 'block', 'fontSize': '10px', 'color': theme_spec["muted"]},
                    )
                    for link in spectrum_links
                ]),
            ])
        )

    spectra_card = _section('Spectra', spectra_metrics, spectra_extras)

    # ATLAS summary + optional light curve panel
    atlas_metrics = [
        _metric("Photometry", 'yes' if _coerce_bool(payload.get('atlas_has_phot')) else 'no'),
        _metric("cyan n/range", f"{payload.get('atlas_n_det_cyan', 'n/a')} / {payload.get('atlas_cyan_range', 'n/a')}"),
        _metric("orange n/range", f"{payload.get('atlas_n_det_orange', 'n/a')} / {payload.get('atlas_orange_range', 'n/a')}"),
    ]
    atlas_extras = []
    if run_dir is not None:
        atlas_idx = _index_external_lc_paths(str(run_dir.resolve()), "atlas")
        for key in lookup_keys:
            path_str = atlas_idx.get(str(key))
            if path_str:
                atlas_path = Path(path_str)
                if atlas_path.exists():
                    try:
                        atlas_lc = pd.read_parquet(atlas_path)
                        atlas_fig = _build_external_lc_figure(
                            atlas_lc, "ATLAS",
                            [("c", "mag", "mag_err", "#00ccff"),
                             ("o", "mag", "mag_err", "#ff8c42")],
                            time_col="mjd",
                            filter_col="filter",
                            source_name="atlas",
                            theme=theme,
                            jd_system="mjd",
                        )
                        atlas_extras.append(_exportable_graph(atlas_fig, panel="external", name="atlas", height="250px"))
                    except Exception:
                        pass
                break

    atlas_card = _section('ATLAS', atlas_metrics, atlas_extras)

    # NEOWISE summary + optional light curve panel
    neowise_epochs = payload.get('neowise_n_epochs', 0)
    neowise_rows = pd.DataFrame()
    neowise_path = None
    if run_dir is not None:
        idx_map = _index_neowise_paths(str(run_dir.resolve()))
        for key in lookup_keys:
            path_str = idx_map.get(str(key))
            if path_str:
                neowise_path = Path(path_str)
                break
    neowise_plot = None
    if neowise_path and neowise_path.exists():
        try:
            neowise_rows = pd.read_parquet(neowise_path)
            neowise_plot = _exportable_graph(
                _build_neowise_figure_with_theme(neowise_rows, theme),
                panel="external",
                name="neowise",
                height="250px",
            )
        except Exception:
            neowise_plot = html.Div(f"Could not load NEOWISE parquet: {neowise_path}", style=error_text_style)

    neowise_metrics = [
        _metric("Epochs", neowise_epochs),
        _metric("W1 range", payload.get('neowise_w1_range', 'n/a')),
        _metric("W2 range", payload.get('neowise_w2_range', 'n/a')),
    ]
    neowise_extras = []
    if neowise_path:
        neowise_extras.append(html.Div(f"File: {neowise_path.name}", style=muted_text_style))
    if neowise_plot is not None:
        neowise_extras.append(neowise_plot)

    neowise_card = _section('NEOWISE', neowise_metrics, neowise_extras)

    # ZTF LC card
    ztf_metrics = [
        _metric("Detections", payload.get('ztf_lc_n_det', 'n/a')),
        _metric("g range", payload.get('ztf_lc_g_range', 'n/a')),
        _metric("r range", payload.get('ztf_lc_r_range', 'n/a')),
    ]
    ztf_extras = []
    if run_dir is not None:
        ztf_idx = _index_external_lc_paths(str(run_dir.resolve()), "ztf")
        for key in lookup_keys:
            path_str = ztf_idx.get(str(key))
            if path_str:
                ztf_path = Path(path_str)
                if ztf_path.exists():
                    try:
                        ztf_lc = pd.read_parquet(ztf_path)
                        ztf_fig = _build_external_lc_figure(
                            ztf_lc, "ZTF",
                            [("zg", "mag", "mag_err", "#44aa44"),
                             ("zr", "mag", "mag_err", "#dd4444"),
                             ("zi", "mag", "mag_err", "#8844cc")],
                            time_col="mjd",
                            filter_col="band",
                            source_name="ztf",
                            theme=theme,
                            jd_system="mjd",
                        )
                        ztf_extras.append(_exportable_graph(ztf_fig, panel="external", name="ztf", height="250px"))
                    except Exception:
                        pass
                break

    ztf_card = _section('ZTF', ztf_metrics, ztf_extras)

    # Gaia epoch LC card
    gaia_epoch_metrics = [
        _metric("G points", payload.get('gaia_epoch_lc_n_g', 'n/a')),
        _metric("G range", payload.get('gaia_epoch_lc_g_range', 'n/a')),
    ]
    gaia_epoch_extras = []
    if run_dir is not None:
        gaia_idx = _index_external_lc_paths(str(run_dir.resolve()), "gaia_epoch")
        for key in lookup_keys:
            path_str = gaia_idx.get(str(key))
            if path_str:
                gaia_path = Path(path_str)
                if gaia_path.exists():
                    try:
                        gaia_lc = pd.read_parquet(gaia_path)
                        gaia_fig = _build_external_lc_figure(
                            gaia_lc, "Gaia Epoch",
                            [("G", "mag", "mag_err", "#e8c547")],
                            time_col="time",
                            yaxis_label="G [mag]",
                            source_name="gaia_epoch",
                            theme=theme,
                            jd_system="bjd_gaia",
                        )
                        gaia_epoch_extras.append(_exportable_graph(gaia_fig, panel="external", name="gaia-epoch", height="250px"))
                    except Exception:
                        pass
                break

    gaia_epoch_card = _section('Gaia Epoch', gaia_epoch_metrics, gaia_epoch_extras)

    # TESS LC card
    tess_metrics = [
        _metric("Sectors", payload.get('tess_n_sectors', 'n/a')),
        _metric("Points", payload.get('tess_total_points', 'n/a')),
        _metric("Flux range", payload.get('tess_flux_range', 'n/a')),
    ]
    tess_extras = []
    if run_dir is not None:
        tess_idx = _index_external_lc_paths(str(run_dir.resolve()), "tess")
        for key in lookup_keys:
            path_str = tess_idx.get(str(key))
            if path_str:
                tess_path = Path(path_str)
                if tess_path.exists():
                    try:
                        tess_lc = pd.read_parquet(tess_path)
                        tess_fig = _build_external_lc_figure(
                            tess_lc, "TESS",
                            [("TESS", "flux", "flux_err", "#cc66ff")],
                            time_col="time",
                            yaxis_label="flux",
                            reverse_y=False,
                            source_name="tess",
                            theme=theme,
                            jd_system="btjd",
                        )
                        tess_extras.append(
                            _exportable_graph(tess_fig, panel="external", name="tess", height="250px")
                        )
                    except Exception:
                        pass
                break

    tess_card = _section('TESS', tess_metrics, tess_extras)

    # Pan-STARRS LC card
    ps1_metrics = [
        _metric("Points", payload.get('ps1_lc_n_points', 'n/a')),
    ]
    ps1_extras = []
    if run_dir is not None:
        ps1_idx = _index_external_lc_paths(str(run_dir.resolve()), "ps1")
        for key in lookup_keys:
            path_str = ps1_idx.get(str(key))
            if path_str:
                ps1_path = Path(path_str)
                if ps1_path.exists():
                    try:
                        ps1_lc = pd.read_parquet(ps1_path)
                        ps1_fig = _build_external_lc_figure(
                            ps1_lc, "Pan-STARRS",
                            [("g_ps", "mag", "mag_err", "#44aa44"),
                             ("r_ps", "mag", "mag_err", "#dd4444"),
                             ("i_ps", "mag", "mag_err", "#8844cc"),
                             ("z_ps", "mag", "mag_err", "#ccaa44"),
                             ("y_ps", "mag", "mag_err", "#aaaa33")],
                            time_col="mjd",
                            filter_col="filter",
                            source_name="ps1",
                            theme=theme,
                            jd_system="mjd",
                        )
                        ps1_extras.append(_exportable_graph(ps1_fig, panel="external", name="pan-starrs", height="250px"))
                    except Exception:
                        pass
                break

    ps1_card = _section('Pan-STARRS', ps1_metrics, ps1_extras)

    # CRTS LC card
    crts_metrics = [
        _metric("Points", payload.get('crts_lc_n_points', 'n/a')),
    ]
    crts_extras = []
    if run_dir is not None:
        crts_idx = _index_external_lc_paths(str(run_dir.resolve()), "crts")
        for key in lookup_keys:
            path_str = crts_idx.get(str(key))
            if path_str:
                crts_path = Path(path_str)
                if crts_path.exists():
                    try:
                        crts_lc = pd.read_parquet(crts_path)
                        crts_fig = _build_external_lc_figure(
                            crts_lc, "CRTS",
                            [("CV", "mag", "mag_err", "#bbbbbb")],
                            time_col="mjd",
                            source_name="crts",
                            theme=theme,
                            jd_system="mjd",
                        )
                        crts_extras.append(_exportable_graph(crts_fig, panel="external", name="crts", height="250px"))
                    except Exception:
                        pass
                break

    crts_card = _section('CRTS', crts_metrics, crts_extras)

    return [cutout_card, multi_survey_card, spectra_card, atlas_card, neowise_card, ztf_card, gaia_epoch_card, tess_card, ps1_card, crts_card]


# ---- sidebar filter helpers ------------------------------------------------
_ATF_OPTS = [
    {'label': 'Any', 'value': 'Any'},
    {'label': 'True', 'value': 'True'},
    {'label': 'False', 'value': 'False'},
    {'label': 'Unset', 'value': 'Unset'},
]
_inp_style = {'width': '100%', 'margin-bottom': '4px', 'font-size': '11px'}

_REVIEW_FILTER_SEARCH_JS = """
function(query, nextClicks, prevClicks, submitCount) {
    var ctx = (window.dash_clientside && window.dash_clientside.callback_context) || null;
    var triggered = ctx && ctx.triggered && ctx.triggered.length ? ctx.triggered[0].prop_id : '';
    var stateKey = '__malcaReviewFilterSearchState';
    var flashKey = '__malcaReviewFilterSearchFlashState';

    function normalize(text) {
        return String(text || '')
            .toLowerCase()
            .replace(/[_/\\-]+/g, ' ')
            .replace(/\\s+/g, ' ')
            .trim();
    }

    function resetHighlight(entry) {
        if (!entry || !entry.el) {
            return;
        }
        entry.el.style.outline = '';
        entry.el.style.outlineOffset = '';
        entry.el.style.borderRadius = '';
        entry.el.style.background = '';
    }

    function highlight(el) {
        var existing = window[flashKey];
        if (existing && existing.timerId) {
            window.clearTimeout(existing.timerId);
        }
        resetHighlight(existing);
        if (!el) {
            window[flashKey] = null;
            return;
        }
        el.style.outline = '2px solid rgba(114, 196, 255, 0.95)';
        el.style.outlineOffset = '3px';
        el.style.borderRadius = '8px';
        el.style.background = 'rgba(50, 89, 123, 0.16)';
        var timerId = window.setTimeout(function() {
            resetHighlight(window[flashKey]);
            window[flashKey] = null;
        }, 1400);
        window[flashKey] = {el: el, timerId: timerId};
    }

    var rawQuery = String(query || '');
    var normalizedQuery = normalize(rawQuery);
    if (!normalizedQuery) {
        window[stateKey] = {query: '', matches: [], index: -1};
        highlight(null);
        return 'Type to find a filter';
    }

    var anchors = Array.from(document.querySelectorAll('.review-filter-anchor'));
    var matches = anchors.filter(function(el) {
        return normalize(el.textContent || '').indexOf(normalizedQuery) !== -1;
    }).map(function(el) {
        return el.id;
    }).filter(Boolean);

    if (!matches.length) {
        window[stateKey] = {query: normalizedQuery, matches: [], index: -1};
        highlight(null);
        return 'No matching filters';
    }

    var state = window[stateKey] || {};
    var sameQuery = state.query === normalizedQuery
        && Array.isArray(state.matches)
        && state.matches.join('|') === matches.join('|');
    var index = sameQuery && Number.isFinite(state.index) ? state.index : 0;

    if (sameQuery) {
        if (triggered === 'review-filter-search-next-btn.n_clicks' || triggered === 'review-filter-search-query.n_submit') {
            index = (index + 1) % matches.length;
        } else if (triggered === 'review-filter-search-prev-btn.n_clicks') {
            index = (index - 1 + matches.length) % matches.length;
        } else if (index >= matches.length) {
            index = matches.length - 1;
        }
    }

    var target = document.getElementById(matches[index]);
    if (target) {
        var group = target.closest('details');
        if (group) {
            group.open = true;
        }
        target.scrollIntoView({behavior: 'smooth', block: 'center', inline: 'nearest'});
        highlight(target);
    } else {
        highlight(null);
    }

    window[stateKey] = {query: normalizedQuery, matches: matches, index: index};
    var label = target && target.getAttribute('title') ? target.getAttribute('title') : 'match';
    return String(index + 1) + ' / ' + String(matches.length) + '  ' + label;
}
"""


def _bool_mode_filter(label: str, component_id: str):
    """Return a (Label, Dropdown) pair for Any/True/False/Unset bool filter."""
    return [
        html.Label(f'{label}:'),
        dcc.Dropdown(
            id=component_id,
            options=_ATF_OPTS,
            value='Any',
            clearable=False,
            style={'margin-bottom': '4px', 'font-size': '11px'},
        ),
    ]


def _col_id(col: str) -> str:
    """snake_case → dash-case for Dash component IDs."""
    return col.replace('_', '-')


def _filter_group_slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', str(name).strip().lower()).strip('-') or 'group'


def _review_filter_group_id(name: str) -> str:
    return f'review-filter-group-{_filter_group_slug(name)}'


def _review_filter_anchor_id(group_name: str, col: str) -> str:
    return f'review-filter-anchor-{_filter_group_slug(group_name)}-{_col_id(col)}'


def _review_filter_search_text(group_name: str, col: str) -> str:
    humanized = col.replace('_', ' ')
    return f'{group_name} {col} {humanized}'


def _review_filter_anchor(group_name: str, col: str, child):
    child_items = list(child) if isinstance(child, (list, tuple)) else [child]
    return html.Div(
        [
            html.Span(_review_filter_search_text(group_name, col), style={'display': 'none'}),
            *child_items,
        ],
        id=_review_filter_anchor_id(group_name, col),
        className='review-filter-anchor',
        title=f'{group_name} / {col}',
        style={'scrollMarginTop': '72px'},
    )


def _select_all_dropdown_values(options: list[dict[str, object]] | None) -> list[str]:
    """Return all distinct option values for a multi-select dropdown."""
    values: list[str] = []
    seen: set[str] = set()
    for option in options or []:
        if not isinstance(option, dict):
            continue
        raw_value = option.get('value')
        if raw_value is None:
            continue
        value = str(raw_value)
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _select_definite_dropdown_values(
    col: str,
    options: list[dict[str, object]] | None,
) -> list[str]:
    """Return dropdown values that represent definite known-type catalog labels."""
    return [
        value
        for value in _select_all_dropdown_values(options)
        if is_definite_known_type_value(col, value)
    ]


def _vetting_known_filter_preset(
    select_options: dict[str, list[dict[str, object]] | None],
    *,
    include_uncertain: bool = True,
) -> tuple[list[str], list[list[str]]]:
    """Build the known-type exclusion preset for broad or definite-only filtering."""
    bool_values = [
        'False' if include_uncertain or col == 'microlens_match' else 'Any'
        for col in VETTING_KNOWN_BOOL_FILTERS
    ]
    select_values = [
        (
            _select_all_dropdown_values(select_options.get(col))
            if include_uncertain
            else _select_definite_dropdown_values(col, select_options.get(col))
        )
        for col in VETTING_KNOWN_SELECT_FILTERS
    ]
    return bool_values, select_values


def _num_range_filter(col: str):
    """Numeric filter with min/max inputs and a slider."""
    return html.Div([
        html.Label(f'{col}:'),
        html.Div([
            dcc.Input(
                id={'type': 'num-filter-min-input', 'col': col},
                type='number',
                placeholder='min',
                debounce=True,
                style={'width': '72px', 'font-size': '11px', 'flex': '0 0 72px'},
            ),
            dcc.RangeSlider(
                id={'type': 'num-filter-range', 'col': col},
                min=0,
                max=1,
                value=[0, 1],
                step=0.01,
                allowCross=False,
                marks=None,
                tooltip={'placement': 'bottom', 'always_visible': False},
                updatemode='mouseup',
                disabled=True,
            ),
            dcc.Input(
                id={'type': 'num-filter-max-input', 'col': col},
                type='number',
                placeholder='max',
                debounce=True,
                style={'width': '72px', 'font-size': '11px', 'flex': '0 0 72px'},
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'margin-bottom': '4px'}),
    ])


def _text_filter(col: str):
    """Dropdown for exact-match string filter (options hydrated lazily)."""
    cid = _col_id(col)
    return html.Div([
        html.Label(f'{col}:'),
        dcc.Dropdown(
            id=f'filter-{cid}',
            options=[{'label': 'Any', 'value': 'Any'}],
            value='Any',
            clearable=False,
            placeholder='Open sidebar to load options',
            style={'margin-bottom': '4px', 'font-size': '11px'},
        ),
    ])


def _select_filter(col: str):
    """Multi-select dropdown for categorical filtering (options hydrated lazily)."""
    cid = _col_id(col)
    return html.Div([
        html.Label(f'{col}:'),
        dcc.Dropdown(
            id=f'exclude-{cid}', options=[], multi=True,
            placeholder='Select values',
            style={'margin-bottom': '4px', 'font-size': '11px'},
            maxHeight=400,
            optionHeight=28,
        ),
    ])


def _make_filter_group(name: str, items: list, *, default_open: bool = False):
    """Build a collapsible html.Details for a group of filters."""
    children = []
    if name == 'Vetting':
        children.append(
            html.Div([
                html.Button(
                    'Exclude Known Types',
                    id='vetting-known-types-btn',
                    n_clicks=0,
                    className='compact-btn',
                    title='Turn on broad known-type vetting filters, including uncertainty and candidate-style labels.',
                ),
                html.Button(
                    'Exclude Certain Known Types',
                    id='vetting-definite-known-types-btn',
                    n_clicks=0,
                    className='compact-btn',
                    title='Exclude only definite catalog known-type labels; keep possible, candidate, suspected, and question-marked labels visible.',
                ),
            ], style={'display': 'flex', 'gap': '6px', 'flexWrap': 'wrap', 'margin-bottom': '6px'})
        )
    for ftype, col in items:
        if ftype == 'bool':
            children.append(_review_filter_anchor(name, col, _bool_mode_filter(col, f'{_col_id(col)}-mode')))
        elif ftype == 'num':
            children.append(_review_filter_anchor(name, col, _num_range_filter(col)))
        elif ftype == 'text':
            children.append(_review_filter_anchor(name, col, _text_filter(col)))
        elif ftype == 'select':
            children.append(_review_filter_anchor(name, col, _select_filter(col)))
    block = [
        html.Summary(name),
        html.Div(children, style={'padding-left': '6px'}),
    ]
    details_kwargs = {
        'id': _review_filter_group_id(name),
        'className': 'review-filter-group',
    }
    if default_open:
        return html.Details(block, open='open', **details_kwargs)
    return html.Details(block, **details_kwargs)


# ---------------------------------------------------------------------------
# Sidebar filter groups — single source of truth for filter UI + state lists
# Each item: ('bool', col_name) | ('num', col_name) | ('text', col_name)
# ---------------------------------------------------------------------------
_SIDEBAR_GROUPS = list(REVIEW_FILTER_SIDEBAR_GROUPS)


_DUSTYCULT_CONTROL_FIELDS = [
    ("start_jd", "dustycult-start-jd", "Start JD", 0.01),
    ("end_jd", "dustycult-end-jd", "End JD", 0.01),
    ("t0_jd", "dustycult-t0-jd", "t0 JD", 0.01),
    ("t0_width_days", "dustycult-t0-width-days", "t0 prior d", 0.1),
    ("log_v_width", "dustycult-log-v-width", "log v width", 0.05),
    ("b_center", "dustycult-b-center", "b center", 0.05),
    ("b_width", "dustycult-b-width", "b width", 0.05),
    ("log_tau0_width", "dustycult-log-tau0-width", "log tau width", 0.05),
    ("alpha_center", "dustycult-alpha-center", "alpha center", 0.05),
    ("alpha_width", "dustycult-alpha-width", "alpha width", 0.05),
    ("log_sigma_width", "dustycult-log-sigma-width", "dust shape width", 0.05),
    ("star_R", "dustycult-star-r", "R star", 0.01),
    ("star_u1", "dustycult-star-u1", "u1", 0.01),
    ("star_u2", "dustycult-star-u2", "u2", 0.01),
]


def _dustycult_number_input(field_id: str, label: str, step: float) -> html.Div:
    return html.Div([
        html.Label(
            label,
            htmlFor=field_id,
            style={'fontSize': '10px', 'color': '#7d91a6', 'marginBottom': '2px'},
        ),
        dcc.Input(
            id=field_id,
            type='number',
            step=step,
            debounce=True,
            style={
                'width': '100%',
                'fontSize': '11px',
                'height': '28px',
                'padding': '2px 6px',
            },
        ),
    ], style={'minWidth': '94px'})


def _dustycult_control_values_from_states(values: tuple[object, ...] | list[object]) -> dict[str, object]:
    return normalize_controls({
        key: value
        for (key, _field_id, _label, _step), value in zip(_DUSTYCULT_CONTROL_FIELDS, values)
        if value is not None
    })


def _dustycult_controls_layout() -> html.Div:
    return html.Div([
        html.Div(
            [
                _dustycult_number_input(field_id, label, step)
                for _key, field_id, label, step in _DUSTYCULT_CONTROL_FIELDS
            ],
            style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(96px, 1fr))',
                'gap': '8px',
                'padding': '8px 10px 0 10px',
            },
        ),
        html.Div([
            html.Button('Recompute Dip Defaults', id='dustycult-recompute-dip-btn', n_clicks=0, className='compact-btn'),
            html.Button('Quick Fit', id='dustycult-quick-fit-btn', n_clicks=0, className='compact-btn'),
            html.Button('Full Fit', id='dustycult-full-fit-btn', n_clicks=0, className='compact-btn'),
            html.Button('Export Fit PDF', id='dustycult-export-fit-btn', n_clicks=0, className='compact-btn'),
            html.Button('Export Occulter PDF', id='dustycult-export-occulter-btn', n_clicks=0, className='compact-btn'),
            html.Span(id='dustycult-run-status', style={'fontSize': '10px', 'color': '#7da8c4'}),
        ], style={'display': 'flex', 'gap': '6px', 'alignItems': 'center', 'flexWrap': 'wrap', 'padding': '8px 10px 0 10px'}),
        html.Div(id='dustycult-defaults-status', style={'fontSize': '10px', 'color': '#7d91a6', 'padding': '4px 10px 8px 10px'}),
    ])


def _phoebe_controls_layout() -> html.Div:
    return html.Div([
        html.Div([
            html.Div([
                html.Label(
                    'Period [d]',
                    htmlFor='phoebe-period-days',
                    style={'fontSize': '10px', 'color': '#7d91a6', 'marginBottom': '2px'},
                ),
                dcc.Input(
                    id='phoebe-period-days',
                    type='number',
                    min=0,
                    step=0.0001,
                    debounce=True,
                    style={'width': '100%', 'fontSize': '11px', 'height': '28px', 'padding': '2px 6px'},
                ),
            ], style={'minWidth': '120px'}),
            html.Div([
                html.Label(
                    'Model',
                    htmlFor='phoebe-model-kind',
                    style={'fontSize': '10px', 'color': '#7d91a6', 'marginBottom': '2px'},
                ),
                dcc.Dropdown(
                    id='phoebe-model-kind',
                    options=[{'label': value.capitalize(), 'value': value} for value in PHOEBE_MODEL_KINDS],
                    value='detached',
                    clearable=False,
                    style={'fontSize': '11px', 'minWidth': '145px'},
                ),
            ], style={'minWidth': '145px'}),
            html.Button('PHOEBE Fit', id='phoebe-fit-btn', n_clicks=0, className='compact-btn'),
            html.Span(id='phoebe-run-status', style={'fontSize': '10px', 'color': '#7da8c4'}),
        ], style={'display': 'flex', 'gap': '8px', 'alignItems': 'end', 'flexWrap': 'wrap', 'padding': '8px 10px 0 10px'}),
        html.Div(id='phoebe-period-status', style={'fontSize': '10px', 'color': '#7d91a6', 'padding': '4px 10px 8px 10px'}),
    ])
