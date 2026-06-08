# This file was mechanically split from malca.review.app; preserve behavior when editing.
app.clientside_callback(
    """
    function(text) {
        return text;
    }
    """,
    Output('bottom-pipeline-status', 'children'),
    Input('pipeline-run-status', 'children'),
    prevent_initial_call=False
)

app.clientside_callback(
    """
    function(idx) {
        return Date.now();
    }
    """,
    Output('candidate-start-time', 'data'),
    Input('current-index', 'data')
)

app.clientside_callback(
    """
    function(idx, queueData) {
        if (!queueData || !Array.isArray(queueData.candidate_ids)) {
            return [null, 0, ''];
        }
        var ids = queueData.candidate_ids || [];
        var size = (typeof queueData.queue_size === 'number') ? queueData.queue_size : ids.length;
        var filterHash = (typeof queueData.filter_hash === 'string') ? queueData.filter_hash : '';
        var i = parseInt(idx == null ? 0 : idx, 10);
        if (!Number.isFinite(i) || i < 0 || i >= ids.length) {
            return [null, size, filterHash];
        }
        return [String(ids[i]), size, filterHash];
    }
    """,
    [Output('current-candidate-id', 'data'),
     Output('queue-size-store', 'data'),
     Output('queue-filter-hash-store', 'data')],
    [Input('current-index', 'data'),
     Input('queue-data', 'data')],
    prevent_initial_call=False
)

app.clientside_callback(
    _REVIEW_FILTER_SEARCH_JS,
    Output('review-filter-search-status', 'children'),
    Input('review-filter-search-query', 'value'),
    Input('review-filter-search-next-btn', 'n_clicks'),
    Input('review-filter-search-prev-btn', 'n_clicks'),
    Input('review-filter-search-query', 'n_submit'),
    prevent_initial_call=False,
)

app.clientside_callback(
    """
    function(n_intervals, progressState, queueSize, sessionStart, toggle) {
        function formatHms(totalSeconds) {
            var seconds = Math.max(0, Math.floor(Number(totalSeconds) || 0));
            var hours = Math.floor(seconds / 3600);
            var minutes = Math.floor((seconds % 3600) / 60);
            var secs = seconds % 60;
            var pad = function(value) {
                return String(value).padStart(2, '0');
            };
            return pad(hours) + ':' + pad(minutes) + ':' + pad(secs);
        }

        // Toggle the review-progress-indicator visibility
        var el = document.getElementById('review-progress-indicator');
        if (el) {
            if (toggle && toggle.indexOf('yes') !== -1) {
                el.style.display = '';
            } else {
                el.style.display = 'none';
            }
        }

        var reviewed = 0;
        var total = 0;
        if (progressState && typeof progressState === 'object') {
            reviewed = parseInt(progressState.reviewed == null ? 0 : progressState.reviewed, 10);
            total = parseInt(progressState.total == null ? 0 : progressState.total, 10);
        }
        if (!Number.isFinite(reviewed) || reviewed < 0) {
            reviewed = 0;
        }
        if (!Number.isFinite(total) || total < 0) {
            total = 0;
        }

        var queueTotal = parseInt(queueSize == null ? 0 : queueSize, 10);
        if ((!Number.isFinite(total) || total <= 0) && Number.isFinite(queueTotal) && queueTotal > 0) {
            total = queueTotal;
        }

        var pct = total > 0 ? (100.0 * reviewed / total) : 0.0;

        var startTs = null;
        if (sessionStart && typeof sessionStart === 'object' && sessionStart.ts != null) {
            startTs = Number(sessionStart.ts);
        }
        if (!Number.isFinite(startTs)) {
            startTs = Date.now() / 1000.0;
        }

        var elapsedS = Math.max(0.0, (Date.now() / 1000.0) - startTs);
        var elapsedTxt = formatHms(elapsedS);

        var pacePerMin = 0.0;
        if (elapsedS > 0 && reviewed > 0) {
            pacePerMin = reviewed / (elapsedS / 60.0);
        }

        var etaTxt = '--:--:--';
        if (pacePerMin > 0 && total > reviewed) {
            etaTxt = formatHms(((total - reviewed) / pacePerMin) * 60.0);
        }

        var paceTxt = pacePerMin > 0 ? pacePerMin.toFixed(2) + '/min' : '--/min';
        return [
            'Reviewed: ' + reviewed + '/' + total + ' (' + pct.toFixed(1) + '%) | Elapsed: ' + elapsedTxt + ' | Pace: ' + paceTxt + ' | ETA: ' + etaTxt,
            ''
        ];
    }
    """,
    [Output('review-progress-indicator', 'children'),
     Output('pace-timer-display', 'children')],
    Input('review-metrics-interval', 'n_intervals'),
    [State('review-progress-state', 'data'),
     State('queue-size-store', 'data'),
     State('review-session-start', 'data'),
     State('pace-timer-toggle', 'value')]
)


# Global keyboard listener (set up once on page load)
app.clientside_callback(
    """
    function() {
        // This runs once when the app loads
        var keyboardInput = document.getElementById('keyboard-input');

        if (!keyboardInput) {
            console.error('keyboard-input element not found!');
            return window.dash_clientside.no_update;
        }

        var valueDescriptor = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype, 'value'
        );
        var nativeInputValueSetter = valueDescriptor && valueDescriptor.set
            ? valueDescriptor.set
            : null;

        var dispatchKeyToDash = function(key) {
            if (!key) {
                return;
            }
            if (nativeInputValueSetter) {
                nativeInputValueSetter.call(
                    keyboardInput, key + '\t' + String(Date.now())
                );
            } else {
                keyboardInput.value = key + '\t' + String(Date.now());
            }
            keyboardInput.dispatchEvent(new Event('input', {bubbles: true}));
        };

        // Register once: global keyboard listener that feeds Dash callbacks.
        if (!window.__malcaKeyboardListenerAttached) {
            document.addEventListener('keydown', function(e) {
                var target = e.target;
                var tag = target && target.tagName ? target.tagName : '';
                var targetId = target && target.id ? target.id : '';
                var inFormField = (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') && targetId !== 'keyboard-input';

                // Inside a form field: keep typing behavior unchanged.
                // To trigger shortcuts while focused in inputs, use Alt+<key>.
                if (inFormField) {
                    if (e.key === 'Escape') {
                        target.blur();
                        return;
                    }
                    var allowWithFormModifier = e.altKey && !e.ctrlKey && !e.metaKey;
                    if (!allowWithFormModifier) {
                        return;
                    }
                    e.preventDefault();
                    dispatchKeyToDash(e.key);
                    return;
                }

                // Outside form fields, shortcuts are single-key only.
                if (e.ctrlKey || e.metaKey || e.altKey) {
                    return;
                }

                var key = e.key;
                if (!key || key === 'Shift' || key === 'Control' || key === 'Alt' || key === 'Meta') {
                    return;
                }

                // Prevent browser defaults for keys we use as shortcuts
                if (key === 'Backspace' || key === 'Tab' || key === 'Enter') {
                    e.preventDefault();
                }

                dispatchKeyToDash(key);
            });
            window.__malcaKeyboardListenerAttached = true;
            console.log('Global keyboard listener initialized');
        }

        return window.dash_clientside.no_update;
    }
    """,
    Output('keyboard-input', 'value', allow_duplicate=True),
    Input('keyboard-init', 'n_intervals'),
    prevent_initial_call='initial_duplicate'
)


app.clientside_callback(
    """
    function(_tick) {
        if (!window.__malcaMetadataCopyAttached) {
            document.addEventListener('click', async function(e) {
                var target = e.target;
                var button = target && target.closest ? target.closest('.metadata-copy-btn') : null;
                if (!button) {
                    return;
                }
                e.preventDefault();
                e.stopPropagation();
                var text = button.getAttribute('data-copy-text') || '';
                var originalTitle = button.getAttribute('title') || 'Copy raw value';
                try {
                    await navigator.clipboard.writeText(text);
                    button.classList.remove('copy-failed');
                    button.classList.add('copied');
                    button.setAttribute('title', 'Copied raw value');
                    setTimeout(function() {
                        button.classList.remove('copied');
                        button.setAttribute('title', originalTitle);
                    }, 900);
                } catch (err) {
                    button.classList.remove('copied');
                    button.classList.add('copy-failed');
                    button.setAttribute('title', 'Clipboard copy failed');
                    setTimeout(function() {
                        button.classList.remove('copy-failed');
                        button.setAttribute('title', originalTitle);
                    }, 1200);
                }
            });
            window.__malcaMetadataCopyAttached = true;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('metadata-copy-init', 'data'),
    Input('keyboard-init', 'n_intervals'),
    prevent_initial_call=False,
)


app.clientside_callback(
    """
    function(_savedStateTs, savedState, currentTheme, reviewScope) {
        var nu = window.dash_clientside.no_update;
        if (savedState && typeof savedState === 'object') {
            return nu;
        }
        var scope = String(reviewScope || 'default');
        var storageKey = 'malca.review.theme::' + scope;
        try {
            var saved = window.localStorage.getItem(storageKey);
            if (saved && ['black', 'gray', 'white'].includes(saved)) {
                return saved;
            }
        } catch (e) {
            // ignore storage read failures
        }
        return ['black', 'gray', 'white'].includes(currentTheme) ? currentTheme : 'black';
    }
    """,
    Output('theme-mode', 'value'),
    Input('saved-review-gui-state', 'modified_timestamp'),
    [State('saved-review-gui-state', 'data'),
     State('theme-mode', 'value'),
     State('review-db-scope', 'data')],
    prevent_initial_call=False,
)


app.clientside_callback(
    """
    function(theme, reviewScope) {
        var t = ['black', 'gray', 'white'].includes(theme) ? theme : 'black';
        var scope = String(reviewScope || 'default');
        var storageKey = 'malca.review.theme::' + scope;
        try {
            document.body.setAttribute('data-theme', t);
            window.localStorage.setItem(storageKey, t);
        } catch (e) {
            // ignore storage/document failures
        }
        return t;
    }
    """,
    Output('theme-mode-store', 'data'),
    Input('theme-mode', 'value'),
    State('review-db-scope', 'data'),
    prevent_initial_call=False,
)


# --- Sidebar plot prefs: save to localStorage on change ---
app.clientside_callback(
    """
    function(preset, overlays, mode, opacity, resHeight, externalSources, externalLayout, phasePanelMode, reviewScope) {
        var scope = String(reviewScope || 'default');
        var storageKey = 'malca.review.sidebar.plot.v3::' + scope;
        try {
            var phaseMode = ['fold', 'time'].includes(phasePanelMode) ? phasePanelMode : 'fold';
            var sourceLayout = ['overlay', 'split'].includes(externalLayout) ? externalLayout : 'overlay';
            var sources = Array.isArray(externalSources) ? externalSources : [];
            var obj = {
                preset: preset,
                overlays: overlays || [],
                mode: mode,
                opacity: opacity,
                resHeight: resHeight,
                externalSources: sources,
                externalLayout: sourceLayout,
                phasePanelMode: phaseMode
            };
            window.localStorage.setItem(storageKey, JSON.stringify(obj));
        } catch (e) {}
        return window.dash_clientside.no_update;
    }
    """,
    Output('sidebar-plot-saved', 'data'),
    [Input('plot-preset', 'value'),
     Input('plot-overlays', 'value'),
     Input('plot-mode', 'value'),
     Input('baseline-opacity-slider', 'value'),
     Input('residual-height-slider', 'value'),
     Input('external-source-values', 'value'),
     Input('external-source-layout', 'value'),
     Input('phase-panel-mode', 'value')],
    State('review-db-scope', 'data'),
    prevent_initial_call=True,
)


# --- Sidebar plot prefs: load from localStorage on init ---
app.clientside_callback(
    """
    function(_savedStateTs, savedState, curPreset, curOverlays, curMode, curOpacity, curResHeight, curExternalSources, curExternalLayout, curPhasePanelMode, reviewScope) {
        var nu = window.dash_clientside.no_update;
        if (savedState && typeof savedState === 'object') {
            return [nu, nu, nu, nu, nu, nu, nu, nu, true];
        }
        var scope = String(reviewScope || 'default');
        var storageKey = 'malca.review.sidebar.plot.v3::' + scope;
        var legacyKey = 'malca.review.sidebar.plot.v2::' + scope;
        var allowedSources = ['asassn', 'atlas', 'ztf', 'gaia_epoch', 'tess', 'neowise', 'kepler', 'aavso', 'ogle', 'stripe82', 'allwise_mep', 'vvvx_virac', 'ps1', 'crts'];
        function normalizeSources(value) {
            var raw = Array.isArray(value) ? value : (value ? [value] : []);
            var out = [];
            var seen = {};
            for (var i = 0; i < raw.length; i++) {
                var text = String(raw[i] || '').trim().toLowerCase();
                if (!text) continue;
                if (text === 'all') return allowedSources.slice();
                if (['wise', 'w1', 'w2', 'wise_w1_w2'].includes(text)) text = 'neowise';
                else if (['k2', 'kepler_k2'].includes(text)) text = 'kepler';
                else if (['sdss_s82', 's82', 'stripe_82', 'sdss_stripe82'].includes(text)) text = 'stripe82';
                else if (['allwise', 'allwise_multiepoch', 'wise_mep'].includes(text)) text = 'allwise_mep';
                else if (['vvv', 'vvvx', 'virac', 'virac2', 'vvvx_virac2'].includes(text)) text = 'vvvx_virac';
                if (!allowedSources.includes(text) || seen[text]) continue;
                out.push(text);
                seen[text] = true;
            }
            if (!Array.isArray(value) && out.length && out[0] !== 'asassn') {
                out.unshift('asassn');
            }
            return out;
        }
        try {
            var raw = window.localStorage.getItem(storageKey) || window.localStorage.getItem(legacyKey);
            if (!raw) return [nu, nu, nu, nu, nu, nu, nu, nu, false];
            var obj = JSON.parse(raw);
            var preset = (obj.preset && ['Fast Review', 'Clean', 'Diagnostics', 'Full'].includes(obj.preset))
                ? obj.preset : nu;
            var overlays = Array.isArray(obj.overlays) ? obj.overlays : nu;
            var mode = (obj.mode && ['native', 'png'].includes(obj.mode)) ? obj.mode : nu;
            var opacity = (typeof obj.opacity === 'number') ? obj.opacity : nu;
            var resHeight = (typeof obj.resHeight === 'number') ? obj.resHeight : nu;
            var sources = normalizeSources(obj.externalSources || obj.externalSource);
            var externalSources = sources.length ? sources : nu;
            var externalLayout = (obj.externalLayout && ['overlay', 'split'].includes(obj.externalLayout))
                ? obj.externalLayout : nu;
            var phasePanelMode = (obj.phasePanelMode && ['fold', 'time'].includes(obj.phasePanelMode))
                ? obj.phasePanelMode : nu;
            return [preset, overlays, mode, opacity, resHeight, externalSources, externalLayout, phasePanelMode, true];
        } catch (e) {
            return [nu, nu, nu, nu, nu, nu, nu, nu, false];
        }
    }
    """,
    [Output('plot-preset', 'value', allow_duplicate=True),
     Output('plot-overlays', 'value', allow_duplicate=True),
     Output('plot-mode', 'value', allow_duplicate=True),
     Output('baseline-opacity-slider', 'value', allow_duplicate=True),
     Output('residual-height-slider', 'value', allow_duplicate=True),
     Output('external-source-values', 'value', allow_duplicate=True),
     Output('external-source-layout', 'value', allow_duplicate=True),
     Output('phase-panel-mode', 'value', allow_duplicate=True),
     Output('plot-defaults-initialized', 'data', allow_duplicate=True)],
    Input('saved-review-gui-state', 'modified_timestamp'),
    [State('saved-review-gui-state', 'data'),
     State('plot-preset', 'value'),
     State('plot-overlays', 'value'),
     State('plot-mode', 'value'),
     State('baseline-opacity-slider', 'value'),
     State('residual-height-slider', 'value'),
     State('external-source-values', 'value'),
     State('external-source-layout', 'value'),
     State('phase-panel-mode', 'value'),
     State('review-db-scope', 'data')],
    prevent_initial_call='initial_duplicate',
)


app.clientside_callback(
    """
    function(_tick) {
        var splitter = document.getElementById('metadata-splitter');
        var leftPanel = document.getElementById('left-info-panel');
        var workspace = document.querySelector('.workspace-panels');
        if (!splitter || !leftPanel || !workspace) {
            return window.dash_clientside.no_update;
        }

        var storageKey = 'malca.review.left_panel.width.v1';
        var minWidth = 260;
        var defaultWidth = 420;

        var computeMaxWidth = function() {
            var total = workspace.clientWidth || window.innerWidth;
            var cap = Math.floor(total * 0.72);
            var floorCap = Math.max(minWidth + 40, cap);
            return floorCap;
        };

        var clampWidth = function(value) {
            var maxWidth = computeMaxWidth();
            var numeric = Number(value);
            if (!isFinite(numeric)) numeric = defaultWidth;
            if (numeric < minWidth) numeric = minWidth;
            if (numeric > maxWidth) numeric = maxWidth;
            return Math.round(numeric);
        };

        var applyWidth = function(value, persist) {
            var w = clampWidth(value);
            leftPanel.style.width = String(w) + 'px';
            leftPanel.style.flex = '0 0 ' + String(w) + 'px';
            if (persist) {
                try { window.localStorage.setItem(storageKey, String(w)); } catch (e) {}
            }
            return w;
        };

        if (!window.__malcaMetadataSplitterAttached) {
            var drag = { active: false, startX: 0, startWidth: 0, pointerId: null };

            var onPointerMove = function(e) {
                if (!drag.active) return;
                var nextWidth = drag.startWidth + (e.clientX - drag.startX);
                applyWidth(nextWidth, false);
                e.preventDefault();
            };

            var stopDrag = function(e) {
                if (!drag.active) return;
                drag.active = false;
                splitter.classList.remove('dragging');
                window.removeEventListener('pointermove', onPointerMove);
                window.removeEventListener('pointerup', stopDrag);
                window.removeEventListener('pointercancel', stopDrag);
                if (drag.pointerId !== null && splitter.releasePointerCapture) {
                    try { splitter.releasePointerCapture(drag.pointerId); } catch (err) {}
                }
                drag.pointerId = null;
                applyWidth(leftPanel.getBoundingClientRect().width, true);
                if (e) e.preventDefault();
            };

            splitter.addEventListener('pointerdown', function(e) {
                drag.active = true;
                drag.startX = e.clientX;
                drag.startWidth = leftPanel.getBoundingClientRect().width;
                drag.pointerId = (typeof e.pointerId === 'number') ? e.pointerId : null;
                splitter.classList.add('dragging');
                if (drag.pointerId !== null && splitter.setPointerCapture) {
                    try { splitter.setPointerCapture(drag.pointerId); } catch (err) {}
                }
                window.addEventListener('pointermove', onPointerMove);
                window.addEventListener('pointerup', stopDrag);
                window.addEventListener('pointercancel', stopDrag);
                e.preventDefault();
            });

            window.addEventListener('resize', function() {
                applyWidth(leftPanel.getBoundingClientRect().width, false);
            });

            window.__malcaMetadataSplitterAttached = true;
        }

        var saved = null;
        try { saved = window.localStorage.getItem(storageKey); } catch (e) { saved = null; }
        var initialWidth = defaultWidth;
        if (saved !== null && saved !== '') {
            var parsed = parseInt(saved, 10);
            if (!isNaN(parsed)) initialWidth = parsed;
        }
        applyWidth(initialWidth, false);

        return window.dash_clientside.no_update;
    }
    """,
    Output('metadata-resize-init', 'data'),
    Input('keyboard-init', 'n_intervals'),
    prevent_initial_call=False,
)


app.clientside_callback(
    """
    function(_tick) {
        var splitter = document.getElementById('status-splitter');
        var statusPanel = document.getElementById('plot-status-panel');
        if (!splitter || !statusPanel) {
            return window.dash_clientside.no_update;
        }

        var storageKey = 'malca.review.plot_status.height.v1';
        var minHeight = 16;
        var defaultHeight = 36;

        var computeMaxHeight = function() {
            return Math.max(64, Math.floor(window.innerHeight * 0.42));
        };

        var clampHeight = function(value) {
            var maxHeight = computeMaxHeight();
            var numeric = Number(value);
            if (!isFinite(numeric)) {
                numeric = defaultHeight;
            }
            if (numeric < minHeight) {
                numeric = minHeight;
            }
            if (numeric > maxHeight) {
                numeric = maxHeight;
            }
            return Math.round(numeric);
        };

        var applyHeight = function(value, persist) {
            var h = clampHeight(value);
            statusPanel.style.height = String(h) + 'px';
            statusPanel.style.flex = '0 0 auto';
            if (persist) {
                try {
                    window.localStorage.setItem(storageKey, String(h));
                } catch (e) {
                    // ignore storage failures
                }
            }
            return h;
        };

        if (!window.__malcaStatusSplitterAttached) {
            var drag = {
                active: false,
                startY: 0,
                startHeight: 0,
                pointerId: null,
            };

            var onPointerMove = function(e) {
                if (!drag.active) {
                    return;
                }
                var nextHeight = drag.startHeight - (e.clientY - drag.startY);
                applyHeight(nextHeight, false);
                e.preventDefault();
            };

            var stopDrag = function(e) {
                if (!drag.active) {
                    return;
                }
                drag.active = false;
                splitter.classList.remove('dragging');
                window.removeEventListener('pointermove', onPointerMove);
                window.removeEventListener('pointerup', stopDrag);
                window.removeEventListener('pointercancel', stopDrag);
                if (drag.pointerId !== null && splitter.releasePointerCapture) {
                    try {
                        splitter.releasePointerCapture(drag.pointerId);
                    } catch (err) {
                        // ignore capture-release failures
                    }
                }
                drag.pointerId = null;
                applyHeight(statusPanel.getBoundingClientRect().height, true);
                if (e) {
                    e.preventDefault();
                }
            };

            splitter.addEventListener('pointerdown', function(e) {
                drag.active = true;
                drag.startY = e.clientY;
                drag.startHeight = statusPanel.getBoundingClientRect().height;
                drag.pointerId = (typeof e.pointerId === 'number') ? e.pointerId : null;
                splitter.classList.add('dragging');
                if (drag.pointerId !== null && splitter.setPointerCapture) {
                    try {
                        splitter.setPointerCapture(drag.pointerId);
                    } catch (err) {
                        // ignore capture failures
                    }
                }
                window.addEventListener('pointermove', onPointerMove);
                window.addEventListener('pointerup', stopDrag);
                window.addEventListener('pointercancel', stopDrag);
                e.preventDefault();
            });

            window.addEventListener('resize', function() {
                applyHeight(statusPanel.getBoundingClientRect().height, false);
            });

            window.__malcaStatusSplitterAttached = true;
        }

        var saved = null;
        try {
            saved = window.localStorage.getItem(storageKey);
        } catch (e) {
            saved = null;
        }
        var initialHeight = defaultHeight;
        if (saved !== null && saved !== '') {
            var parsed = parseInt(saved, 10);
            if (!isNaN(parsed)) {
                initialHeight = parsed;
            }
        }
        applyHeight(initialHeight, false);

        return window.dash_clientside.no_update;
    }
    """,
    Output('status-resize-init', 'data'),
    Input('keyboard-init', 'n_intervals'),
    prevent_initial_call=False,
)


app.clientside_callback(
    """
    function(_tick, panelState) {
        var splitter = document.getElementById('eda-splitter');
        var panel = document.getElementById('eda-panel');
        var workspace = document.querySelector('.workspace-panels');
        if (!splitter || !panel || !workspace) {
            return window.dash_clientside.no_update;
        }

        var storageKey = 'malca.review.eda_panel.width.v1';
        var minWidth = 0;
        var defaultWidth = 430;
        var state = String(panelState || 'open');

        var computeMaxWidth = function() {
            var total = workspace.clientWidth || window.innerWidth;
            return Math.max(80, Math.floor(total * 0.82));
        };

        var clampWidth = function(value) {
            var maxWidth = computeMaxWidth();
            var numeric = Number(value);
            if (!isFinite(numeric)) numeric = defaultWidth;
            if (numeric < minWidth) numeric = minWidth;
            if (numeric > maxWidth) numeric = maxWidth;
            return Math.round(numeric);
        };

        var applyWidth = function(value, persist) {
            var w = clampWidth(value);
            panel.classList.remove('is-expanded');
            panel.style.width = String(w) + 'px';
            panel.style.flex = '0 0 ' + String(w) + 'px';
            if (persist) {
                try { window.localStorage.setItem(storageKey, String(w)); } catch (e) {}
            }
            return w;
        };

        var applyExpandedWidth = function() {
            var w = computeMaxWidth();
            panel.style.width = String(w) + 'px';
            panel.style.flex = '0 0 ' + String(w) + 'px';
            panel.classList.add('is-expanded');
            return w;
        };

        if (!window.__malcaEdaSplitterAttached) {
            var drag = { active: false, startX: 0, startWidth: 0, pointerId: null };

            var panelIsOpen = function() {
                return panel && !panel.classList.contains('is-collapsed');
            };

            var onPointerMove = function(e) {
                if (!drag.active) return;
                var nextWidth = drag.startWidth - (e.clientX - drag.startX);
                applyWidth(nextWidth, false);
                e.preventDefault();
            };

            var stopDrag = function(e) {
                if (!drag.active) return;
                drag.active = false;
                splitter.classList.remove('dragging');
                window.removeEventListener('pointermove', onPointerMove);
                window.removeEventListener('pointerup', stopDrag);
                window.removeEventListener('pointercancel', stopDrag);
                if (drag.pointerId !== null && splitter.releasePointerCapture) {
                    try { splitter.releasePointerCapture(drag.pointerId); } catch (err) {}
                }
                drag.pointerId = null;
                applyWidth(panel.getBoundingClientRect().width, true);
                if (e) e.preventDefault();
            };

            splitter.addEventListener('pointerdown', function(e) {
                if (!panelIsOpen()) return;
                drag.active = true;
                drag.startX = e.clientX;
                drag.startWidth = panel.getBoundingClientRect().width;
                drag.pointerId = (typeof e.pointerId === 'number') ? e.pointerId : null;
                splitter.classList.add('dragging');
                panel.classList.remove('is-expanded');
                if (drag.pointerId !== null && splitter.setPointerCapture) {
                    try { splitter.setPointerCapture(drag.pointerId); } catch (err) {}
                }
                window.addEventListener('pointermove', onPointerMove);
                window.addEventListener('pointerup', stopDrag);
                window.addEventListener('pointercancel', stopDrag);
                e.preventDefault();
            });

            window.addEventListener('resize', function() {
                if (panelIsOpen()) {
                    if (panel.classList.contains('is-expanded')) {
                        applyExpandedWidth();
                    } else {
                        applyWidth(panel.getBoundingClientRect().width, false);
                    }
                }
            });

            window.__malcaEdaSplitterAttached = true;
        }

        if (state === 'expanded') {
            applyExpandedWidth();
        } else if (state !== 'collapsed') {
            var saved = null;
            try { saved = window.localStorage.getItem(storageKey); } catch (e) { saved = null; }
            var initialWidth = defaultWidth;
            if (saved !== null && saved !== '') {
                var parsed = parseInt(saved, 10);
                if (!isNaN(parsed)) initialWidth = parsed;
            }
            applyWidth(initialWidth, false);
        }

        return window.dash_clientside.no_update;
    }
    """,
    Output('eda-resize-init', 'data'),
    Input('keyboard-init', 'n_intervals'),
    Input('eda-panel-state', 'data'),
    prevent_initial_call=False,
)
