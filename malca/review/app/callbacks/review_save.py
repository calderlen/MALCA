# This file was mechanically split from malca.review.app; preserve behavior when editing.
def _do_save(candidate_id, score, taxonomy_selection, needs_followup, notes, event_type, *, increment_pass=False):
    """Shared save helper.  Auto-sets status; only increments review_pass on Done."""
    with closing(db_connect(Path(DB_PATH))) as conn:
        review = get_review(conn, candidate_id)
        current_pass = max(1, review.get('review_pass', 0))
        new_pass = current_pass + 1 if increment_pass else current_pass
        workflow_status = 'needs_followup' if needs_followup else 'reviewed'
        selection = selection_from_review(taxonomy_selection if isinstance(taxonomy_selection, dict) else {})
        save_review(
            conn,
            candidate_id=candidate_id,
            interest_score=score,
            review_pass=new_pass,
            notes=notes or '',
            workflow_status=workflow_status,
            disposition=selection.get('disposition') or 'keep',
            morphology_primary=selection.get('morphology_primary'),
            morphology_secondary=selection.get('morphology_secondary'),
            morphology_secondary_json=selection.get('morphology_secondary_json'),
            morphology_polarity=selection.get('morphology_polarity'),
            morphology_recurrence=selection.get('morphology_recurrence'),
            baseline_behavior=selection.get('baseline_behavior'),
            physical_primary=selection.get('physical_primary'),
            physical_secondary=selection.get('physical_secondary'),
            classification_confidence=selection.get('classification_confidence'),
            priority_tags=selection.get('priority_tags'),
            evidence_flags=selection.get('evidence_flags'),
            model_tags=selection.get('model_tags'),
            duplicate_of=selection.get('duplicate_of'),
            known_object_id=selection.get('known_object_id'),
            known_object_source=selection.get('known_object_source'),
            legacy_review_json=review.get('legacy_review_json') or '{}',
            reviewer='calder',
            event_type=event_type,
        )
        _clear_review_state_caches()
        return new_pass, workflow_status


app.clientside_callback(
    """
    function(keyValue, currentIdx, queueSize, currentCandidateId, currentScore,
             taxonomySelection, activeMenu, taxonomySubmenu, needsFollowup, notes, saveRequest) {
        var no = window.dash_clientside.no_update;
        var taxonomy = TAXONOMY_PAYLOAD_PLACEHOLDER;
        var primaryByKey = {};
        var primaryByValue = {};
        taxonomy.morphology_primary.forEach(function(item) {
            primaryByKey[String(item.key).toLowerCase()] = item;
            primaryByValue[String(item.value)] = item;
        });
        var familyByKey = {};
        taxonomy.physical_primary.forEach(function(item) {
            familyByKey[String(item.key).toLowerCase()] = item;
        });
        var secondaryByKey = function(primary) {
            var out = {};
            (taxonomy.morphology_secondary[primary] || []).forEach(function(item) {
                out[String(item.key).toLowerCase()] = item;
            });
            return out;
        };
        var subclassByKey = function(family) {
            var out = {};
            (taxonomy.physical_secondary[family] || []).forEach(function(item) {
                out[String(item.key).toLowerCase()] = item;
            });
            return out;
        };
        var key = '';
        if (keyValue) {
            key = String(keyValue).split('\\t', 1)[0].trim();
        }
        if (!key || key === '?') {
            return [no, no, no, no, no, no, no, no, no, no];
        }

        var size = parseInt(queueSize == null ? 0 : queueSize, 10);
        if (!Number.isFinite(size) || size <= 0) {
            return [no, 'Queue is empty', no, no, no, no, no, no, no, no];
        }

        var idx = parseInt(currentIdx == null ? 0 : currentIdx, 10);
        if (!Number.isFinite(idx)) {
            idx = 0;
        }
        var candidateId = currentCandidateId == null ? '' : String(currentCandidateId);
        var selection = taxonomySelection && typeof taxonomySelection === 'object'
            ? Object.assign({}, taxonomySelection)
            : {};
        var nextActiveMenu = activeMenu || '';
        var nextSubmenu = taxonomySubmenu || '';
        var cache = window.__malcaTaxonomyKeyState;
        var nowMs = Date.now();
        if (cache && cache.candidateId === candidateId && (nowMs - cache.updatedAt) < 5000) {
            selection = Object.assign({}, selection, cache.selection || {});
            nextActiveMenu = cache.activeMenu || nextActiveMenu;
            nextSubmenu = cache.submenu || nextSubmenu;
        }
        if (!selection.priority_tags) { selection.priority_tags = []; }
        if (!selection.evidence_flags) { selection.evidence_flags = []; }
        if (!selection.model_tags) { selection.model_tags = []; }
        var detailLabelByValue = {};
        Object.keys(taxonomy.morphology_secondary || {}).forEach(function(primary) {
            (taxonomy.morphology_secondary[primary] || []).forEach(function(item) {
                detailLabelByValue[String(item.value)] = item.label || item.value;
            });
        });
        var normalizeDetailList = function(raw, scalar) {
            var source = [];
            if (Array.isArray(raw)) {
                source = raw.slice();
            } else if (typeof raw === 'string' && raw.trim()) {
                try {
                    var parsed = JSON.parse(raw);
                    source = Array.isArray(parsed) ? parsed : [raw];
                } catch (_err) {
                    source = [raw];
                }
            } else if (raw !== null && raw !== undefined && raw !== '') {
                source = [raw];
            }
            if (scalar !== null && scalar !== undefined && String(scalar).trim()) {
                source.unshift(scalar);
            }
            var seen = {};
            var out = [];
            source.forEach(function(item) {
                var text = String(item == null ? '' : item).trim();
                if (text && !seen[text]) {
                    seen[text] = true;
                    out.push(text);
                }
            });
            return out;
        };
        var setDetailList = function(details) {
            var normalized = normalizeDetailList(details, null);
            selection.morphology_secondary_list = normalized;
            selection.morphology_secondary = normalized.length ? normalized[0] : null;
            selection.morphology_secondary_json = JSON.stringify(normalized);
            return normalized;
        };
        var describeDetailList = function(details) {
            var normalized = normalizeDetailList(details, null);
            if (!normalized.length) {
                return 'cleared';
            }
            return normalized.map(function(value) {
                return detailLabelByValue[value] || value;
            }).join(', ');
        };
        var toggleDetail = function(value) {
            var details = normalizeDetailList(
                selection.morphology_secondary_list || selection.morphology_secondary_json,
                selection.morphology_secondary
            );
            var next = [];
            var removed = false;
            details.forEach(function(item) {
                if (item === value) {
                    removed = true;
                } else {
                    next.push(item);
                }
            });
            if (!removed) {
                next.push(value);
            }
            return setDetailList(next);
        };
        setDetailList(normalizeDetailList(
            selection.morphology_secondary_list || selection.morphology_secondary_json,
            selection.morphology_secondary
        ));
        var nextScore = currentScore;
        var nextFollowup = !!needsFollowup;
        var nextIdx = idx;
        var notice = no;
        var saveReq = no;
        var prefixOut = no;
        var selectionOut = no;
        var activeOut = no;
        var submenuOut = no;

        var nextNonce = 1;
        if (saveRequest && typeof saveRequest === 'object' && typeof saveRequest.nonce === 'number') {
            nextNonce = saveRequest.nonce + 1;
        }

        var buildSaveRequest = function(scoreValue, incrementPass) {
            return {
                nonce: nextNonce,
                candidate_id: candidateId,
                score: scoreValue,
                taxonomy: selection,
                needs_followup: nextFollowup,
                notes: notes || '',
                increment_pass: !!incrementPass,
                event_type: 'keyboard',
            };
        };

        var lower = key.toLowerCase();
        var emitTaxonomy = function(message) {
            selectionOut = selection;
            activeOut = nextActiveMenu;
            submenuOut = nextSubmenu;
            window.__malcaTaxonomyKeyState = {
                candidateId: candidateId,
                selection: Object.assign({}, selection),
                activeMenu: nextActiveMenu,
                submenu: nextSubmenu,
                updatedAt: Date.now()
            };
            return [no, message, no, no, no, prefixOut, no, selectionOut, activeOut, submenuOut];
        };

        if (key === 'Escape') {
            if (nextActiveMenu) {
                nextActiveMenu = '';
                nextSubmenu = '';
                return emitTaxonomy('Taxonomy menu closed');
            }
            return [no, no, no, no, no, no, no, no, no, no];
        }

        if (key === '1' || key === '2' || key === '3' || key === '4') {
            nextScore = parseInt(key, 10);
            notice = '✓ Confidence: ' + String(nextScore);
            saveReq = buildSaveRequest(nextScore, false);
            return [no, notice, nextScore, no, no, prefixOut, saveReq, no, no, no];
        }

        if (nextActiveMenu === 'morphology_secondary') {
            if (key === 'Backspace') {
                setDetailList([]);
                nextSubmenu = selection.morphology_primary || nextSubmenu;
                return emitTaxonomy('Detail cleared');
            }
            var secondary = secondaryByKey(selection.morphology_primary || nextSubmenu)[lower];
            if (secondary) {
                var details = toggleDetail(secondary.value);
                nextSubmenu = selection.morphology_primary || nextSubmenu;
                return emitTaxonomy('Detail: ' + describeDetailList(details));
            }
        }

        if (nextActiveMenu === 'physical_primary') {
            if (key === 'Backspace') {
                selection.physical_primary = null;
                selection.physical_secondary = null;
                nextActiveMenu = '';
                nextSubmenu = '';
                return emitTaxonomy('Hypothesis cleared');
            }
            var family = familyByKey[lower];
            if (family) {
                if (selection.physical_primary === family.value) {
                    selection.physical_primary = null;
                    selection.physical_secondary = null;
                    nextActiveMenu = '';
                    nextSubmenu = '';
                    return emitTaxonomy('Hypothesis cleared');
                }
                selection.physical_primary = family.value;
                selection.physical_secondary = null;
                nextActiveMenu = (taxonomy.physical_secondary[family.value] || []).length ? 'physical_secondary' : '';
                nextSubmenu = nextActiveMenu ? family.value : '';
                return emitTaxonomy('Hypothesis: ' + family.label);
            }
        }

        if (nextActiveMenu === 'physical_secondary') {
            if (key === 'Backspace') {
                selection.physical_secondary = null;
                nextActiveMenu = '';
                nextSubmenu = '';
                return emitTaxonomy('Physical subclass cleared');
            }
            var subclass = subclassByKey(selection.physical_primary || nextSubmenu)[lower];
            if (subclass) {
                selection.physical_secondary = (selection.physical_secondary === subclass.value) ? null : subclass.value;
                nextSubmenu = selection.physical_primary || nextSubmenu;
                return emitTaxonomy('Subclass: ' + (selection.physical_secondary || 'cleared'));
            }
        }

        // Let detail keys work against the selected morphology even if Dash
        // has not yet rendered the submenu after the primary keypress.
        // H is reserved for opening the hypothesis menu unless the detail
        // submenu is already active, where it selects the H-labeled detail.
        if (selection.morphology_primary && lower !== 'h') {
            var activeSecondary = secondaryByKey(selection.morphology_primary)[lower];
            if (activeSecondary) {
                var activeDetails = toggleDetail(activeSecondary.value);
                nextActiveMenu = 'morphology_secondary';
                nextSubmenu = selection.morphology_primary;
                return emitTaxonomy('Detail: ' + describeDetailList(activeDetails));
            }
        }

        if (primaryByKey[lower]) {
            var primary = primaryByKey[lower];
            if (selection.morphology_primary === primary.value) {
                selection.morphology_primary = null;
                setDetailList([]);
                nextActiveMenu = '';
                nextSubmenu = '';
                return emitTaxonomy('Morphology cleared');
            }
            selection.morphology_primary = primary.value;
            setDetailList([]);
            nextActiveMenu = 'morphology_secondary';
            nextSubmenu = primary.value;
            return emitTaxonomy('Morphology: ' + primary.label);
        }

        if (lower === 'h') {
            if (nextActiveMenu === 'physical_primary') {
                nextActiveMenu = '';
                nextSubmenu = '';
                return emitTaxonomy('Hypothesis menu closed');
            }
            nextActiveMenu = 'physical_primary';
            nextSubmenu = '';
            return emitTaxonomy('Hypothesis menu');
        }

        if (key === ',') {
            nextFollowup = !nextFollowup;
            notice = 'Followup: ' + (nextFollowup ? 'ON' : 'OFF');
            return [no, notice, no, nextFollowup, no, prefixOut, no, no, no, no];
        }

        if (key === 'Backspace') {
            nextIdx = Math.max(0, idx - 1);
            notice = '← Previous';
            return [nextIdx !== idx ? nextIdx : no, notice, no, no, no, prefixOut, no, no, no, no];
        }

        if (key === 'Tab') {
            nextIdx = Math.min(idx + 1, size - 1);
            notice = '→ Next';
            return [nextIdx !== idx ? nextIdx : no, notice, no, no, no, prefixOut, no, no, no, no];
        }

        if (key === '.') {
            if (!candidateId) {
                return [no, 'Queue is empty', no, no, no, prefixOut, no, no, no, no];
            }
            notice = '✓ Saved';
            saveReq = buildSaveRequest(currentScore, false);
            return [no, notice, no, no, no, prefixOut, saveReq, no, no, no];
        }

        if (key === 'Enter') {
            if (currentScore == null || currentScore === '') {
                return [no, '⚠ Confidence required', no, no, no, prefixOut, no, no, no, no];
            }
            if (!selection.morphology_primary) {
                return [no, '⚠ Morphology required', no, no, no, prefixOut, no, no, no, no];
            }
            nextIdx = Math.min(idx + 1, size - 1);
            notice = '✓ Saved + Next →';
            saveReq = buildSaveRequest(currentScore, true);
            return [nextIdx !== idx ? nextIdx : no, notice, no, no, no, prefixOut, saveReq, no, no, no];
        }

        return [no, no, no, no, no, prefixOut, no, no, no, no];
    }
    """.replace("TAXONOMY_PAYLOAD_PLACEHOLDER", json.dumps(TAXONOMY_KEYBOARD_PAYLOAD, sort_keys=True)),
    [Output('current-index', 'data', allow_duplicate=True),
     Output('notification', 'children', allow_duplicate=True),
     Output('current-score', 'data', allow_duplicate=True),
     Output('needs-followup-store', 'data', allow_duplicate=True),
     Output('event-class-store', 'data', allow_duplicate=True),
     Output('pending-prefix', 'data', allow_duplicate=True),
     Output('review-save-request', 'data', allow_duplicate=True),
     Output('taxonomy-selection-store', 'data', allow_duplicate=True),
     Output('active-taxonomy-menu', 'data', allow_duplicate=True),
     Output('taxonomy-submenu-store', 'data', allow_duplicate=True)],
    Input('keyboard-input', 'value'),
    [State('current-index', 'data'),
     State('queue-size-store', 'data'),
     State('current-candidate-id', 'data'),
     State('current-score', 'data'),
     State('taxonomy-selection-store', 'data'),
     State('active-taxonomy-menu', 'data'),
     State('taxonomy-submenu-store', 'data'),
     State('needs-followup-store', 'data'),
     State('notes', 'value'),
     State('review-save-request', 'data')],
    prevent_initial_call=True,
)


@app.callback(
    [Output('notification', 'children', allow_duplicate=True),
     Output('review-pass-store', 'data', allow_duplicate=True)],
    Input('review-save-request', 'data'),
    State('current-candidate-id', 'data'),
    prevent_initial_call=True,
)
def persist_review_save_request(save_request, current_candidate_id):
    """Persist clientside-queued review saves without blocking UI feedback."""
    if not isinstance(save_request, dict):
        raise dash.exceptions.PreventUpdate

    try:
        nonce = int(save_request.get('nonce', 0) or 0)
    except Exception:
        nonce = 0
    if nonce <= 0:
        raise dash.exceptions.PreventUpdate

    candidate_id = str(save_request.get('candidate_id') or '').strip()
    if not candidate_id:
        raise dash.exceptions.PreventUpdate

    raw_score = save_request.get('score')
    try:
        score = int(raw_score) if raw_score not in (None, '') else None
    except Exception:
        score = None

    try:
        new_pass, _status = _do_save(
            candidate_id,
            score,
            save_request.get('taxonomy'),
            bool(save_request.get('needs_followup')),
            save_request.get('notes'),
            str(save_request.get('event_type') or 'keyboard'),
            increment_pass=bool(save_request.get('increment_pass')),
        )
    except Exception as exc:
        traceback.print_exc()
        return f"✗ Save failed: {exc}", no_update

    pass_out = no_update
    if str(current_candidate_id or '') == candidate_id:
        pass_out = new_pass
    return no_update, pass_out


@app.callback(
    Output('plot-render-request', 'data'),
    [Input('current-index', 'data'),
     Input('current-candidate-id', 'data'),
     Input('plot-mode', 'value'),
     Input('plot-overlays', 'value'),
     Input('camera-checklist', 'value'),
     Input('plot-preset', 'value'),
     Input('residual-height-slider', 'value'),
     Input('theme-mode-store', 'data'),
     Input('queue-size-store', 'data'),
     Input('pipeline-progress-trigger', 'data'),
     Input('baseline-opacity-slider', 'value'),
     Input('band-checklist', 'value'),
     Input('round-sigfigs', 'value'),
     Input('link-radius-arcsec', 'value'),
     Input('pdm-result-store', 'data'),
     Input('pdm-min-period', 'value'),
     Input('pdm-max-period', 'value'),
     Input('pdm-manual-period', 'value'),
     Input('yaxis-mode', 'value'),
     Input('native-color-mode', 'value'),
     Input('phase-panel-mode', 'value'),
     Input('external-source-values', 'value'),
     Input('external-source-layout', 'value')],
     State('plot-render-request', 'data'),
    prevent_initial_call=True,
)
def queue_plot_render_request(idx, current_candidate_id, plot_mode, overlay_values, selected_cameras, preset, residual_height, theme_mode, _queue_size, _pipeline_progress, baseline_opacity, selected_bands, round_sigfigs, link_radius, pdm_result, pdm_min_period, pdm_max_period, pdm_manual_period, yaxis_mode, native_color_mode, phase_panel_mode, external_source_values, external_source_layout=None, existing_request=None):
    """Debounced render request queue for native plot UX."""
    if existing_request is None and isinstance(external_source_layout, dict):
        existing_request = external_source_layout
        external_source_layout = DEFAULT_EXTERNAL_SOURCE_LAYOUT
    req = existing_request or {'nonce': 0, 'ts': 0.0}
    # Determine effective phase period: manual override > search/harmonic result.
    override_period = None
    override_period_source = ''
    phase_period_pending = False
    phase_period_pending_source = ''
    suppress_catalog_phase_period = False
    candidate_id = str(current_candidate_id) if current_candidate_id is not None else None
    if pdm_manual_period is not None:
        try:
            p = float(pdm_manual_period)
            if p > 0:
                override_period = p
                override_period_source = 'manual/search'
        except (TypeError, ValueError):
            pass
    if override_period is None:
        min_p, max_p = _normalize_period_search_bounds(pdm_min_period, pdm_max_period)
        result_matches_context = False
        if isinstance(pdm_result, dict):
            result_candidate = str(pdm_result.get('candidate_id') or '')
            result_matches_context = bool(candidate_id and result_candidate == candidate_id)
            try:
                result_min = float(pdm_result.get('min_period'))
                result_max = float(pdm_result.get('max_period'))
            except (TypeError, ValueError):
                result_matches_context = False
            else:
                result_matches_context = bool(
                    result_matches_context
                    and np.isclose(result_min, min_p)
                    and np.isclose(result_max, max_p)
                )

        if isinstance(pdm_result, dict) and result_matches_context:
            if bool(pdm_result.get('pending')):
                phase_period_pending = True
                phase_period_pending_source = str(pdm_result.get('source') or pdm_result.get('method') or 'Auto period search')
            else:
                try:
                    period = float(pdm_result.get('best_period'))
                except (TypeError, ValueError):
                    period = np.nan
                if np.isfinite(period) and period > 0:
                    override_period = period
                    override_period_source = str(pdm_result.get('source') or pdm_result.get('method') or 'period search')
    normalized_external_sources = normalize_external_source_values(
        external_source_values,
        default=list(DEFAULT_EXTERNAL_SOURCE_VALUES),
    )
    return {
        'nonce': int(req.get('nonce', 0)) + 1,
        'ts': float(time.time()),
        'state': {
            'idx': idx,
            'candidate_id': candidate_id,
            'plot_mode': plot_mode,
            'overlay_values': list(overlay_values or []),
            'selected_cameras': list(selected_cameras or []),
            'selected_bands': list(selected_bands or ['g', 'V']),
            'preset': preset,
            'residual_height': float(residual_height if residual_height is not None else DEFAULT_RESIDUAL_FRACTION),
            'theme': theme_mode or DEFAULT_THEME,
            'baseline_opacity': float(baseline_opacity if baseline_opacity is not None else 0.5),
            'round_sigfigs': bool(True if round_sigfigs is None else ('yes' in round_sigfigs)),
            'link_radius': float(link_radius) if link_radius is not None else 30.0,
            'override_period': override_period,
            'override_period_source': override_period_source,
            'phase_period_pending': bool(phase_period_pending),
            'phase_period_pending_source': phase_period_pending_source,
            'suppress_catalog_phase_period': bool(suppress_catalog_phase_period),
            'yaxis_mode': str(yaxis_mode or 'mag'),
            'native_color_mode': 'band' if str(native_color_mode or 'camera') == 'band' else 'camera',
            'phase_panel_mode': str(phase_panel_mode or 'fold'),
            'external_source_values': normalized_external_sources,
            'external_source_layout': normalize_external_source_layout(external_source_layout),
            'external_source_view': legacy_external_source_view(normalized_external_sources),
        },
    }
