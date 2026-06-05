# This file was mechanically split from malca.review.app; preserve behavior when editing.
# Custom OLED black CSS
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>MALCA Review</title>
    {%metas%}
    {%css%}
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
    <style>
        *, *::before, *::after {
            box-sizing: border-box;
        }
        body {
            background-color: #000;
            color: #e0e0e0;
            --review-table-cell-bg: #071016;
            --review-table-header-bg: #101b24;
            --review-table-text: #dce8f2;
            --review-table-border: rgba(84, 118, 140, 0.35);
            --review-table-header-border: rgba(84, 118, 140, 0.45);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 0;
        }
        .main-container {
            height: 100vh;
            display: flex;
            background-color: #000;
        }
        .sidebar {
            position: fixed;
            left: -280px;
            top: 0;
            width: 280px;
            height: 100vh;
            background-color: #0a0a0a;
            border-right: 1px solid #333;
            transition: left 0.2s ease;
            z-index: 1000;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 12px 14px;
            font-size: 11px;
            color: #bbb;
        }
        .sidebar.expanded {
            left: 0;
        }
        .sidebar .section-title {
            color: #0af;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin: 0 0 6px 0;
        }
        .sidebar hr {
            border: none;
            border-top: 1px solid #222;
            margin: 10px 0;
        }
        .sidebar label {
            display: block;
            color: #777;
            font-size: 11px;
            margin-bottom: 2px;
        }
        .sidebar-field-label {
            color: #8ba4b8;
            font-size: 10px;
            line-height: 1.25;
            margin-bottom: 2px;
            letter-spacing: 0.15px;
        }
        .sidebar-field-label p {
            margin: 0;
        }
        .sidebar details {
            margin-bottom: 2px;
        }
        .sidebar details summary {
            color: #0af;
            font-size: 11px;
            cursor: pointer;
            padding: 3px 0;
            user-select: none;
        }
        .sidebar details summary:hover {
            color: #4cf;
        }
        .sidebar details[open] > summary {
            margin-bottom: 4px;
        }
        .sidebar-toggle {
            appearance: none;
            position: fixed;
            left: 0;
            top: 50px;
            width: 30px;
            height: 60px;
            background-color: #1a1a1a;
            border: 1px solid #555;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1001;
            color: #0af;
            font-size: 20px;
            transition: left 0.2s ease;
            padding: 0;
        }
        .sidebar-toggle.sidebar-expanded {
            left: 280px;
        }
        .sidebar-toggle:hover {
            background-color: #2a2a2a;
        }
        .content-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 0;
            min-width: 0;
            padding-left: 0;
            transition: padding-left 0.2s ease;
        }
        /* When sidebar is open, push main content so it doesn't get covered */
        .sidebar.expanded + .content-area {
            padding-left: 280px;
        }
        .workspace-panels {
            flex: 1;
            min-height: 0;
            min-width: 0;
            display: flex;
            overflow: hidden;
            padding: 8px 10px 10px 10px;
            gap: 0;
        }
        .left-info-panel {
            flex: 0 0 340px;
            width: 340px;
            min-width: 260px;
            max-width: 72vw;
            display: flex;
            flex-direction: column;
            gap: 8px;
            height: 100%;
            min-height: 0;
            overflow: hidden;
            padding-right: 8px;
        }
        .left-info-scroll {
            flex: 1 1 auto;
            min-height: 0;
            overflow-y: auto;
            overflow-x: hidden;
            overscroll-behavior: contain;
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding-right: 2px;
            padding-bottom: 12px;
        }
        .right-plot-panel {
            flex: 1;
            min-width: 0;
            min-height: 0;
        }
        .header-bar {
            background-color: #0a0a0a;
            border-bottom: 1px solid #555;
            padding: 6px 20px;
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-start;
            align-items: center;
            gap: 14px;
            font-size: 11px;
        }
        .header-key-info {
            flex: 1;
            min-width: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header-key-info .item {
            color: #8fb1c8;
            font-size: 10px;
            white-space: nowrap;
        }
        .header-key-info .item.path {
            flex: 1;
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .plot-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            align-items: stretch;
            background-color: #000;
            overflow: hidden;
            min-height: 260px;
            padding: 0 2px 0 8px;
            gap: 8px;
        }
        .eda-panel {
            flex: 0 0 430px;
            width: 430px;
            min-width: 0;
            max-width: 82vw;
            height: 100%;
            min-height: 0;
            overflow: hidden;
            padding-left: 0;
            display: flex;
            flex-direction: column;
        }
        .eda-panel.is-expanded {
            flex-basis: min(82vw, 980px);
            width: min(82vw, 980px);
        }
        .eda-panel.is-collapsed {
            display: none;
        }
        .eda-panel-inner {
            height: 100%;
            min-height: 0;
            overflow-y: auto;
            overflow-x: hidden;
            margin-left: 8px;
            border: 1px solid rgba(84, 118, 140, 0.35);
            border-radius: 8px;
            background: linear-gradient(180deg, rgba(8, 18, 24, 0.94), rgba(3, 8, 12, 0.82));
            padding: 10px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .eda-panel-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            min-width: 0;
        }
        .eda-panel-actions {
            display: flex;
            align-items: center;
            gap: 6px;
            flex: 0 0 auto;
        }
        .eda-panel-title {
            color: #c6d7e8;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.2px;
        }
        .eda-status-line {
            color: #7d91a6;
            font-size: 10px;
            line-height: 1.35;
            min-height: 14px;
        }
        .eda-controls {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 6px;
        }
        .eda-control-full {
            grid-column: 1 / -1;
        }
        .eda-field-label {
            color: #8ba4b8;
            font-size: 10px;
            line-height: 1.25;
            margin-bottom: 2px;
        }
        .eda-graph-card,
        .eda-table-card {
            border: 1px solid rgba(84, 118, 140, 0.32);
            border-radius: 8px;
            background: rgba(3, 8, 12, 0.58);
            padding: 8px;
            min-width: 0;
        }
        .eda-graph-card {
            flex: 0 0 auto;
        }
        .eda-graph-wrap {
            height: clamp(320px, 38vh, 560px);
            min-width: 0;
        }
        .eda-panel.is-expanded .eda-graph-wrap {
            height: clamp(430px, 58vh, 760px);
        }
        .eda-table-card {
            flex: 1 1 auto;
            min-height: 260px;
            display: flex;
            flex-direction: column;
        }
        .eda-table-card .dash-table-container {
            flex: 1 1 auto;
            min-height: 0;
        }
        .eda-splitter {
            position: relative;
        }
        .eda-drag-handle {
            display: none;
        }
        .eda-panel-toggle {
            appearance: none;
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            border: 0;
            background: transparent;
            color: transparent;
            font-size: 0;
            line-height: 0;
            padding: 0;
            cursor: pointer;
            z-index: 3;
            display: none;
        }
        .eda-splitter.collapsed .eda-panel-toggle {
            display: block;
        }
        .panel-splitter-vertical {
            position: relative;
            width: 12px;
            flex: 0 0 12px;
            height: auto;
            margin: 0 2px;
            cursor: col-resize;
            user-select: none;
            touch-action: none;
        }
        .panel-splitter-vertical::before {
            content: '';
            position: absolute;
            left: 50%;
            top: 0;
            bottom: 0;
            width: 1px;
            height: auto;
            transform: translateX(-50%);
            background: rgba(126, 150, 166, 0.45);
        }
        .panel-splitter-vertical::after {
            content: '::';
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            letter-spacing: 1px;
            padding: 6px 3px;
            writing-mode: vertical-rl;
            text-orientation: mixed;
            border-radius: 999px;
            color: #8db0c8;
            font-size: 10px;
            background: rgba(8, 18, 25, 0.9);
            border: 1px solid rgba(86, 114, 132, 0.55);
            line-height: 1;
        }
        .panel-splitter-vertical:hover::after,
        .panel-splitter-vertical.dragging::after {
            color: #b5d4ea;
            border-color: rgba(133, 171, 196, 0.9);
            background: rgba(12, 26, 35, 0.96);
        }
        .plot-toolbar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(min(150px, 100%), 1fr));
            align-items: start;
            gap: 8px;
            padding: 8px 10px;
            border: 1px solid rgba(84, 118, 140, 0.35);
            background: linear-gradient(180deg, rgba(8, 18, 24, 0.9), rgba(3, 8, 12, 0.75));
            border-radius: 8px;
            font-size: 11px;
        }
        .plot-control-group {
            min-width: 0;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 6px;
        }
        .plot-control-full {
            grid-column: 1 / -1;
        }
        .plot-control-label {
            color: #9fb6cb;
            font-size: 10px;
            white-space: nowrap;
        }
        .plot-control-preset .dash-dropdown {
            flex: 1 1 130px;
            min-width: 0;
        }
        .plot-control-checks .dash-checklist,
        .plot-control-radio .dash-radioitems {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        .plot-control-checks .dash-checklist label,
        .plot-control-radio .dash-radioitems label,
        .plot-control-group label {
            margin: 0;
        }
        .plot-actions {
            align-items: center;
        }
        .plot-source-control .form-select {
            flex: 1 1 130px;
            min-width: 0;
        }
        .plot-period-controls {
            align-items: center;
        }
        .period-method-control {
            flex: 0 1 86px;
            min-width: 76px;
        }
        .period-number-input {
            width: 72px;
        }
        .period-manual-input {
            width: 92px;
        }
        .plot-control-status {
            flex: 1 1 160px;
            min-width: 0;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .compact-btn {
            background-color: #14212b;
            color: #c6d7e8;
            border: 1px solid rgba(92, 129, 154, 0.6);
            border-radius: 5px;
            padding: 2px 7px;
            font-size: 10px;
            cursor: pointer;
        }
        .compact-btn:hover {
            border-color: #7da8c4;
            background-color: #1a2b38;
        }
        .meta-toolbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            padding: 6px 10px;
            border: 1px solid rgba(84, 118, 140, 0.25);
            border-radius: 8px;
            background: rgba(6, 14, 20, 0.7);
        }
        .meta-toolbar .title {
            color: #8fb1c8;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.6px;
        }
        .plot-toolbar .label-chip {
            color: #85a7bf;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .plot-toolbar .dash-checklist label,
        .plot-toolbar label {
            color: #c9d4df !important;
            margin-right: 8px;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .plot-toolbar .dash-checklist label,
        .plot-toolbar .dash-radioitems label {
            padding: 4px 9px;
            border-radius: 4px;
            border: 1px solid rgba(60, 92, 112, 0.55);
            background: rgba(7, 16, 22, 0.9);
        }
        .sidebar .dash-checklist label,
        .sidebar .dash-radioitems label {
            padding: 4px 8px;
            border-radius: 4px;
            border: 1px solid rgba(60, 92, 112, 0.55);
            background: rgba(7, 16, 22, 0.9);
            margin-bottom: 4px;
        }
        .toolbar-slider-control {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            flex: 1 1 145px;
            min-width: 130px;
            max-width: none;
        }
        .toolbar-slider-control > div {
            flex: 1 1 90px;
            min-width: 80px;
        }
        .plot-control-status:empty {
            display: none;
        }
        .pipeline-log-panel {
            font-size: 10px;
            line-height: 1.35;
            margin-top: 6px;
            max-height: 220px;
            overflow-y: auto;
            padding: 8px;
            background: rgba(8, 16, 24, 0.75);
            border: 1px solid #284059;
            border-radius: 6px;
            white-space: pre-wrap;
            word-break: break-word;
            color: #9fc6df;
        }
        .meta-field-row {
            display: flex;
            justify-content: space-between;
            gap: 8px;
            padding: 2px 0;
            border-bottom: 1px solid #1a1a1a;
        }
        .copyable-math-field {
            display: inline-flex;
            align-items: center;
            justify-content: flex-end;
            gap: 4px;
            min-width: 0;
            flex: 1 1 auto;
        }
        .meta-field-label {
            color: #7fa3bc;
            flex-shrink: 0;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stat-field-label {
            text-transform: none;
            letter-spacing: 0.2px;
        }
        .meta-field-value {
            color: #e2edf6;
            text-align: right;
            font-weight: 600;
            word-break: break-word;
            white-space: normal;
        }
        .metadata-copy-btn {
            opacity: 0;
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 1px solid rgba(125, 145, 166, 0.42);
            background: rgba(9, 18, 26, 0.84);
            color: #9fc6df;
            font-size: 12px;
            line-height: 18px;
            padding: 0;
            cursor: pointer;
            flex: 0 0 auto;
            transition: opacity 0.12s ease, border-color 0.12s ease, color 0.12s ease;
        }
        .meta-field-row:hover .metadata-copy-btn,
        .metadata-copy-btn:focus {
            opacity: 1;
        }
        .metadata-copy-btn.copied {
            opacity: 1;
            color: #77d28f;
            border-color: rgba(100, 194, 123, 0.72);
        }
        .metadata-copy-btn.copy-failed {
            opacity: 1;
            color: #dd8080;
            border-color: rgba(221, 128, 128, 0.72);
        }
        .meta-field-label p,
        .meta-field-value p {
            margin: 0;
        }
        .lazy-panel-placeholder {
            border: 1px dashed rgba(125, 145, 166, 0.36);
            border-radius: 6px;
            padding: 8px 10px;
            color: #9fb6cb;
            background: rgba(8, 16, 24, 0.42);
            font-size: 11px;
            line-height: 1.35;
        }
        .lazy-panel-placeholder-error {
            color: #dd8080;
            border-color: rgba(221, 128, 128, 0.45);
            background: rgba(48, 18, 18, 0.26);
        }
        .survey-cutout-card {
            display: flex;
            flex-direction: column;
            gap: 6px;
            min-width: 0;
        }
        .cutout-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            min-width: 0;
        }
        .cutout-source-link {
            flex: 0 0 auto;
            color: #86c7ff;
            font-size: 10px;
            font-weight: 600;
            line-height: 1.1;
            text-decoration: none;
        }
        .cutout-source-link:hover {
            color: #b7dcff;
            text-decoration: underline;
        }
        .cutout-controls-row {
            display: grid;
            grid-template-columns: minmax(190px, 260px) minmax(160px, 1fr);
            align-items: center;
            gap: 6px 10px;
            min-width: 0;
        }
        .cutout-survey-select {
            min-width: 0;
            font-size: 11px;
        }
        .cutout-survey-select .Select-control {
            min-height: 30px;
            height: 30px;
            border-radius: 4px;
        }
        .cutout-survey-select .Select-value,
        .cutout-survey-select .Select-placeholder,
        .cutout-survey-select .Select-input {
            line-height: 28px !important;
        }
        .cutout-status {
            min-width: 0;
            overflow-wrap: anywhere;
        }
        .cutout-viewer {
            position: relative;
            width: min(100%, 512px);
            max-width: 512px;
            aspect-ratio: 1 / 1;
            overflow: hidden;
            border-radius: 5px;
            border: 1px solid rgba(125, 145, 166, 0.42);
            background: #020406;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.03);
        }
        .cutout-image {
            width: 100%;
            height: 100%;
            display: block;
            object-fit: cover;
            background: #020406;
        }
        .cutout-image[src=""] {
            display: none;
        }
        .cutout-crosshair {
            pointer-events: none;
            position: absolute;
            inset: 0;
        }
        .cutout-crosshair::before,
        .cutout-crosshair::after {
            content: "";
            position: absolute;
            background: rgba(255, 255, 255, 0.86);
            box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.28);
        }
        .cutout-crosshair::before {
            left: 50%;
            top: calc(50% - 14px);
            width: 1px;
            height: 28px;
            transform: translateX(-50%);
        }
        .cutout-crosshair::after {
            left: calc(50% - 14px);
            top: 50%;
            width: 28px;
            height: 1px;
            transform: translateY(-50%);
        }
        .cutout-empty-label {
            display: none;
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            color: #7d91a6;
            font-size: 11px;
            font-weight: 600;
            text-align: center;
        }
        .cutout-viewer-empty {
            background: rgba(8, 16, 24, 0.46);
            border-style: dashed;
        }
        .cutout-viewer-empty .cutout-crosshair {
            display: none;
        }
        .cutout-viewer-empty .cutout-empty-label {
            display: block;
        }
        @media (max-width: 760px) {
            .cutout-controls-row {
                grid-template-columns: 1fr;
            }
            .cutout-viewer {
                width: 100%;
            }
        }
        .dustycult-param-table th,
        .dustycult-param-table td {
            border-bottom: 1px solid rgba(125, 145, 166, 0.18);
        }
        .vetting-banner-empty {
            padding: 6px 12px;
            margin: 4px 0;
            border-radius: 4px;
            background: #333;
            color: #999;
            font-size: 0.85em;
            text-align: center;
        }
        .vetting-banner-shell {
            margin: 4px 0;
        }
        .vetting-banner-header {
            padding: 3px 8px;
            border-radius: 4px 4px 0 0;
            font-weight: bold;
            font-size: 11px;
            text-align: center;
        }
        .vetting-banner-header.known {
            background: #4a1111;
            color: #ff6b6b;
            border: 1px solid #ff6b6b;
            border-bottom: none;
        }
        .vetting-banner-header.new {
            background: #114a11;
            color: #6bff6b;
            border: 1px solid #6bff6b;
            border-bottom: none;
        }
        .vetting-banner-grid {
            display: flex;
            flex-direction: column;
            gap: 2px;
            padding: 5px 6px 6px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-top: none;
            border-radius: 0 0 4px 4px;
        }
        .vetting-banner-grid.with-links {
            border-radius: 0;
        }
        .vetting-banner-cell {
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            border: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            overflow: hidden;
        }
        .vetting-banner-shell.known .vetting-banner-cell {
            background: #2a1a1a;
        }
        .vetting-banner-shell.new .vetting-banner-cell {
            background: #1a2a1a;
        }
        .vetting-banner-cell.hit.known {
            background: #3a1515;
            border-color: #b94a4a;
        }
        .vetting-banner-cell.hit.new {
            background: #153a1b;
            border-color: #4aa864;
        }
        .vetting-banner-label {
            color: #888;
            font-size: 11px;
            flex-shrink: 0;
        }
        .vetting-banner-value {
            color: #e0e0e0;
            font-weight: bold;
            text-align: right;
            word-break: break-word;
            white-space: normal;
        }
        .vetting-banner-hit.known {
            color: #ff6b6b;
        }
        .vetting-banner-hit.new {
            color: #6bff6b;
        }
        .vetting-banner-links {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            padding: 6px;
            background: #161616;
            border: 1px solid #333;
            border-top: none;
            border-radius: 0 0 4px 4px;
        }
        .vetting-banner-link {
            display: inline-block;
            padding: 2px 6px;
            background: #222;
            border: 1px solid #444;
            border-radius: 3px;
            color: #8af;
            text-decoration: none;
            font-size: 10px;
            white-space: nowrap;
        }
        .vetting-banner-link:hover {
            text-decoration: none;
            border-color: #6a8ca6;
        }
        /* Blue slider theming — global override for all Dash sliders in toolbar */
        .plot-toolbar .rc-slider-rail {
            background-color: #284256 !important;
        }
        .plot-toolbar .rc-slider-track {
            background-color: #0af !important;
        }
        .plot-toolbar .rc-slider-handle {
            border-color: #0af !important;
            background-color: #0b141d !important;
            box-shadow: 0 0 0 3px rgba(0, 170, 255, 0.2) !important;
            outline: none !important;
        }
        .plot-toolbar .rc-slider-handle:hover,
        .plot-toolbar .rc-slider-handle:focus,
        .plot-toolbar .rc-slider-handle:active,
        .plot-toolbar .rc-slider-handle-dragging {
            border-color: #0af !important;
            background-color: #0b141d !important;
            box-shadow: 0 0 0 5px rgba(0, 170, 255, 0.3) !important;
        }
        .plot-toolbar .rc-slider-dot-active {
            border-color: #0af !important;
        }
        .plot-toolbar .rc-slider-tooltip-inner {
            background-color: #0af !important;
            border: 1px solid #0af !important;
            color: #fff !important;
        }
        .plot-toolbar .rc-slider-tooltip-arrow {
            border-top-color: #0af !important;
            border-bottom-color: #0af !important;
        }
        .plot-frame {
            flex: 1;
            min-height: 260px;
            border: 1px solid rgba(84, 118, 140, 0.35);
            border-radius: 10px;
            background: radial-gradient(circle at 20% 0%, rgba(17, 39, 54, 0.22), rgba(0, 0, 0, 0.05) 45%, rgba(0, 0, 0, 0));
            overflow: hidden;
            position: relative;
        }
        .plot-native {
            width: 100%;
            height: 100%;
        }
        .plot-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .plot-stats {
            display: flex;
            flex-direction: column;
            gap: 4px;
            margin-top: 2px;
        }
        .plot-status {
            border: 1px solid rgba(102, 126, 143, 0.45);
            border-radius: 8px;
            padding: 4px 8px;
            background: rgba(9, 18, 25, 0.82);
            color: #d4dfeb;
            font-size: 10px;
            line-height: 1.2;
            overflow: hidden;
        }
        .plot-status .status-line {
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .plot-status details {
            margin-top: 2px;
        }
        .plot-status summary {
            cursor: pointer;
            color: #8fb1c8;
            font-size: 10px;
            user-select: none;
        }
        .plot-status ul {
            margin: 3px 0 0 14px;
            padding: 0;
        }
        .plot-status li {
            margin: 1px 0;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .plot-status.warn {
            border-color: rgba(186, 144, 44, 0.7);
            background: rgba(41, 29, 6, 0.62);
        }
        .plot-status.error {
            border-color: rgba(192, 72, 72, 0.78);
            background: rgba(48, 12, 12, 0.58);
        }
        .camera-diag {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            font-size: 10px;
            color: #b9cad9;
        }
        .camera-diag .item {
            border: 1px solid rgba(90, 118, 138, 0.55);
            border-radius: 999px;
            padding: 2px 8px;
            background: rgba(11, 23, 31, 0.7);
        }
        .queue-provenance-panel {
            border: 1px solid rgba(77, 106, 127, 0.5);
            border-radius: 8px;
            padding: 6px 8px;
            background: rgba(9, 18, 25, 0.82);
            margin-top: 6px;
        }
        .queue-provenance-list {
            margin: 6px 0 0 16px;
            padding: 0;
            color: #cad9e5;
            font-size: 10px;
            line-height: 1.35;
        }
        .queue-provenance-list li {
            margin: 3px 0;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .queue-provenance-note {
            margin-top: 6px;
            color: #7d91a6;
            font-size: 10px;
            line-height: 1.3;
        }
        .sidebar-camera-actions {
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
            margin-bottom: 4px;
        }
        .sidebar-camera-actions .action-btn {
            margin-right: 0;
            padding: 1px 6px;
            font-size: 10px;
        }
        .sidebar-camera-checklist label {
            color: #cad9e5 !important;
            margin-right: 8px;
            margin-bottom: 2px;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .stat-card {
            padding: 4px 8px;
            border-radius: 5px;
            border: 1px solid rgba(64, 96, 116, 0.45);
            background-color: rgba(8, 17, 24, 0.75);
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
        }
        .stat-card .label {
            color: #7fa3bc;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stat-card .value {
            color: #e2edf6;
            font-size: 12px;
            font-weight: 600;
            text-align: right;
        }
        .run-config-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 8px;
        }
        .run-config-item {
            border: 1px solid rgba(78, 110, 132, 0.45);
            border-radius: 6px;
            background: rgba(7, 16, 22, 0.68);
            padding: 6px 8px;
            min-height: 52px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .run-config-item.wide {
            grid-column: 1 / -1;
            min-height: 0;
        }
        .run-config-item .k {
            color: #7fa3bc;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }
        .run-config-item .v {
            color: #dce8f2;
            font-size: 11px;
            font-weight: 600;
            margin-top: 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .run-config-item.wide .v {
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
            word-break: break-word;
        }
        .run-config-item.warning {
            border-color: rgba(186, 144, 44, 0.75);
            background: rgba(44, 30, 8, 0.2);
        }
        .repro-badge {
            border-radius: 999px;
            border: 1px solid rgba(99, 129, 153, 0.6);
            padding: 2px 8px;
            font-size: 10px;
            color: #9bc1dc;
            background: rgba(12, 25, 33, 0.8);
        }
        .repro-badge.warn {
            border-color: rgba(186, 144, 44, 0.75);
            color: #e4c16d;
            background: rgba(44, 30, 8, 0.62);
        }
        .metadata-bar {
            background-color: #0a0a0a;
            border-top: 1px solid #333;
            padding: 10px 20px;
            font-size: 11px;
            color: #aaa;
            max-height: 80px;
            overflow-y: auto;
        }
        .control-bar {
            background-color: #0a0a0a;
            padding: 6px 15px;
            border-top: 1px solid #555;
            font-size: 11px;
        }
        .review-form {
            background-color: #0a0a0a;
            padding: 4px 12px;
            border-top: 1px solid #555;
            overflow-y: auto;
            font-size: 11px;
        }
        .score-btn {
            background-color: #1a1a1a;
            border: 1px solid #444;
            color: #fff;
            padding: 2px 7px;
            font-size: 11px;
            cursor: pointer;
            border-radius: 4px;
            transition: all 0.1s;
            margin-right: 4px;
        }
        .score-btn:hover {
            border-color: #666;
            background-color: #2a2a2a;
        }
        .score-btn.active {
            border-color: #0af;
            background-color: #003366;
        }
        .badge-btn {
            background-color: transparent;
            border: 1px solid #444;
            color: #888;
            padding: 3px 8px;
            font-size: 11px;
            cursor: pointer;
            border-radius: 4px;
            margin-right: 5px;
            transition: all 0.1s;
        }
        .badge-btn:hover {
            border-color: #666;
            color: #bbb;
            background-color: #1a1a1a;
        }
        .badge-btn.active {
            border-color: #0f0;
            color: #0f0;
            background-color: #003300;
        }
        .action-btn {
            background-color: #1a1a1a;
            color: #0af;
            border: 1px solid #0af;
            padding: 2px 8px;
            font-size: 11px;
            cursor: pointer;
            border-radius: 4px;
            margin-right: 6px;
        }
        .action-btn.primary {
            background-color: #003366;
        }
        .queue-refresh-btn {
            font-size: 12px !important;
            font-weight: 600;
            min-height: 34px;
            padding: 6px 10px;
        }
        .notification {
            color: #7fd6a8;
            font-size: 11px;
            flex: 1 1 280px;
            min-width: 0;
            max-width: 100%;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
            opacity: 0.95;
        }
        .help-link {
            color: #0af;
            cursor: pointer;
            text-decoration: none;
            font-size: 11px;
        }
        .help-link:hover {
            text-decoration: underline;
        }
        .Select-control {
            background-color: #0f1418 !important;
            border-color: #2f4658 !important;
            min-height: 26px !important;
            height: 26px !important;
            box-shadow: none !important;
        }
        .Select-menu-outer {
            background-color: #0f1418 !important;
            border-color: #2f4658 !important;
        }
        input[type="text"],
        input[type="number"],
        input[type="search"],
        input[type="email"],
        input[type="password"],
        textarea {
            background-color: #1a1a1a !important;
            border: 1px solid #555 !important;
            color: #e0e0e0 !important;
        }
        /* Dropdown styling */
        .Select-control, .Select-menu-outer, .Select-menu, .Select-option {
            background-color: #0f1418 !important;
            color: #d5e3ef !important;
            border-color: #2f4658 !important;
        }
        .Select-menu-outer {
            overscroll-behavior: contain;
            -webkit-overflow-scrolling: touch;
            /* Fix scroll feel */
            scrollbar-color: #2f4658 #0f1418;
            scrollbar-width: thin;
        }
        /* Custom scrollbar for Webkit */
        .Select-menu-outer::-webkit-scrollbar {
            width: 8px;
            background-color: #0f1418;
        }
        .Select-menu-outer::-webkit-scrollbar-thumb {
            background-color: #2f4658;
            border-radius: 4px;
        }
        .Select-menu-outer::-webkit-scrollbar-thumb:hover {
            background-color: #0af;
        }
        /* NUCLEAR OPTION: Force text color on EVERYTHING inside the dropdown menu */
        body .Select-menu-outer,
        body .Select-menu-outer *,
        body .Select-menu-outer div,
        body .Select-menu-outer span,
        body .Select-menu-outer label,
        body .Select-menu-outer a,
        body .Select-menu-outer button,
        body .Select-menu-outer strong,
        body .Select-menu-outer b,
        body .Select-menu-outer i,
        body .Select-menu-outer em,
        body .Select-menu-outer small,
        body .Select-option,
        body .VirtualizedSelectOption,
        body .Select * {
            color: #dce8f2 !important;
            opacity: 1 !important;
        }

        /* Force dividers (borders) between options */
        body .Select-option,
        body .VirtualizedSelectOption,
        body [class*="Select-option"] {
            border-bottom: 1px solid #2f4658 !important;
            padding: 8px 10px !important; /* Ensure enough space for divider to be seen */
        }
        
        /* Last child should not have a border usually, but for clarity let's keep it or remove it */
        body .Select-option:last-child,
        body .VirtualizedSelectOption:last-child {
            border-bottom: none !important;
        }

        /* Ensure hover state stays readable and distinct */
        body .Select-option.is-focused,
        body .VirtualizedSelectOption.is-focused,
        body .Select-option:hover,
        body .VirtualizedSelectOption:hover {
            background-color: #1d2d3a !important;
            color: #ffffff !important;
            cursor: pointer !important;
        }

        /* Specific fix for VirtualizedSelectOption "Select All" helper text container */
        .VirtualizedSelectOption {
             display: flex !important;
             align-items: center !important;
        }

        /* Force any SVG icons (checkmarks) to be visible */
        .Select-menu-outer svg,
        .Select-menu-outer path {
            fill: #dce8f2 !important;
            stroke: #dce8f2 !important;
        }

        /* Force high specificity on the option text itself */
        .VirtualizedSelectOption, .Select-option {
            color: #dce8f2 !important;
            text-shadow: none !important;
        }
        
        /* Pseudo-elements just in case */
        body .Select-menu-outer *::before,
        body .Select-menu-outer *::after,
        body .Select *::before,
        body .Select *::after {
            color: #dce8f2 !important;
        }
        .Select-placeholder, .Select-value-label {
            color: #9db4c7 !important;
            font-size: 10px !important;
            line-height: 24px !important;
        }
        .Select-input {
            height: 24px !important;
        }
        .Select-arrow-zone {
            padding-right: 5px !important;
        }
        .Select-arrow {
            border-top-color: #9db4c7 !important;
            border-left-color: transparent !important;
            border-right-color: transparent !important;
            opacity: 1 !important;
        }
        .is-open > .Select-control .Select-arrow {
            border-top-color: transparent !important;
            border-bottom-color: #b8cede !important;
        }
        /* Checkbox and label styling */
        label, .form-label {
            color: #aaa !important;
        }
        /* DASH/REACT-SELECT COMPONENT OVERRIDES - THE FINAL HAMMER */
        
        /* 1. The Menu Container */
        .Select-menu-outer {
            background-color: #0f1418 !important;
            border-color: #2f4658 !important;
        }

        /* 2. The Options (including Select All) */
        .VirtualizedSelectOption, 
        .Select-option,
        .dash-dropdown-option,
        div[role="option"] {
            background-color: #10171d !important;
            color: #dce8f2 !important;
            border-bottom: 1px solid #2f4658 !important; /* Visible divider */
            opacity: 1 !important;
        }

        /* 3. Hover/Focused State */
        .VirtualizedSelectOption.is-focused,
        .Select-option.is-focused,
        .dash-dropdown-option:hover,
        .VirtualizedSelectOption:hover {
            background-color: #1d2d3a !important;
            color: #ffffff !important;
            cursor: pointer !important;
        }

        /* 4. "Select All" / "Deselect All" often lives in a special header or div at the top */
        .Select-menu-outer > div:first-child,
        .Select-menu-outer > div:nth-child(1) {
             color: #dce8f2 !important;
        }
        
        /* 5. Force text color on children (labels, spans) inside options */
        .VirtualizedSelectOption *,
        .Select-option * {
            color: inherit !important;
        }
        .dash-dropdown {
            background-color: #0f1418 !important;
            color: #dce8f2 !important;
            border: 1px solid #2f4658 !important;
            border-radius: 4px !important;
            min-height: 24px !important;
            height: 24px !important;
            padding: 0 6px !important;
            box-shadow: none !important;
        }
        .dash-dropdown-trigger {
            min-height: 24px !important;
            height: 24px !important;
            background-color: #0f1418 !important;
        }
        .dash-dropdown-value,
        .dash-dropdown-value-item {
            color: #dce8f2 !important;
            font-size: 10px !important;
            line-height: 20px !important;
        }
        .dash-dropdown-content,
        .dash-dropdown-options,
        .dash-options-list,
        .dash-dropdown-search-container {
            background-color: #0f1418 !important;
            border: 1px solid #2f4658 !important;
            color: #dce8f2 !important;
        }
        .dash-dropdown-search {
            background-color: #0f1418 !important;
            color: #dce8f2 !important;
            border: 1px solid #2f4658 !important;
            font-size: 10px !important;
            min-height: 22px !important;
        }
        .dash-dropdown-option,
        .dash-options-list-option {
            background-color: #10171d !important;
            color: #dce8f2 !important;
            font-size: 10px !important;
            padding: 3px 8px !important;
            min-height: 22px !important;
            border-bottom: 1px solid #2f4658 !important;
        }
        .dash-dropdown-option:hover,
        .dash-options-list-option:hover,
        .dash-dropdown-option.selected,
        .dash-options-list-option.selected {
            background-color: #1d2d3a !important;
            color: #fff !important;
        }
        /* Checklist items */
        .dash-checklist label {
            color: #e0e0e0 !important;
        }
        .dash-checklist input[type="checkbox"] {
            accent-color: #0af;
        }
        .dash-radioitems input[type="radio"],
        input[type="radio"] {
            accent-color: #0af;
        }
        /* Input placeholders */
        ::placeholder {
            color: #666 !important;
            opacity: 1;
        }
        :-ms-input-placeholder {
            color: #666 !important;
        }
        ::-ms-input-placeholder {
            color: #666 !important;
        }
        /* Collapsible metadata sections */
        .metadata-sections {
            background-color: #0a0a0a;
            border-top: 2px solid #555;
            overflow: visible;
            flex-shrink: 0;
            padding: 0 12px;
            border-radius: 8px;
        }
        .candidate-metadata {
            flex: 1;
            min-height: 100px;
            height: auto;
            max-height: none;
        }
        @media (min-width: 1600px) {
            .candidate-metadata {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
                align-items: start;
            }
            .candidate-metadata > .stats-details {
                grid-column: 1 / -1;
            }
        }
        .stats-details {
            min-width: 0;
        }
        .metadata-sections.stats-sections-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 4px 12px;
            padding: 0 8px 8px 8px;
        }
        .stats-sections-grid .stats-section {
            min-width: 0;
            border-bottom: 1px solid #222;
        }
        .metadata-sections.stats-sections-grid .meta-grid {
            padding: 2px 4px 6px 4px;
        }
        .metadata-health {
            display: flex;
            align-items: center;
            gap: 8px;
            border: 1px solid rgba(83, 113, 133, 0.5);
            border-radius: 8px;
            padding: 5px 8px;
            margin: 5px 0 6px 0;
            background: rgba(8, 16, 23, 0.72);
            font-size: 10px;
            line-height: 1.3;
        }
        .metadata-health .chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            border: 1px solid rgba(99, 129, 153, 0.6);
            padding: 1px 7px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            white-space: nowrap;
            font-weight: 600;
            color: #9bc1dc;
            background: rgba(12, 25, 33, 0.82);
        }
        .metadata-health .detail {
            color: #adbfce;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .bottom-context-bar {
            display: flex;
            align-items: center;
            gap: 22px;
            flex-wrap: wrap;
            min-width: 0;
        }
        .bottom-context-item {
            display: flex;
            align-items: baseline;
            gap: 8px;
            min-width: 0;
            flex: 1 1 320px;
        }
        .bottom-context-k {
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            white-space: nowrap;
            color: #86a7bd;
        }
        .bottom-context-v {
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 10px;
            color: #c5d5e1;
        }
        .metadata-health.metadata-health-base .chip {
            color: #e4c16d;
            border-color: rgba(186, 144, 44, 0.75);
            background: rgba(44, 30, 8, 0.62);
        }
        .metadata-health.metadata-health-partial .chip {
            color: #8dc6de;
            border-color: rgba(96, 146, 174, 0.7);
            background: rgba(10, 31, 44, 0.64);
        }
        .metadata-health.metadata-health-enriched .chip {
            color: #9fd4b7;
            border-color: rgba(72, 148, 112, 0.72);
            background: rgba(10, 35, 23, 0.62);
        }
        .metadata-sections details {
            border-bottom: 1px solid #222;
        }
        .metadata-sections summary {
            cursor: pointer;
            padding: 5px 6px;
            color: #0af;
            font-size: 11px;
            font-weight: bold;
            user-select: none;
        }
        .metadata-sections summary:hover {
            color: #4cf;
        }
        .metadata-sections .meta-grid {
            display: flex;
            flex-direction: column;
            gap: 0;
            padding: 2px 6px 6px 6px;
            font-size: 11px;
        }

        /* Theme overrides */
        body[data-theme="black"] {
            background-color: #000 !important;
            color: #e0e0e0 !important;
            --review-table-cell-bg: #071016;
            --review-table-header-bg: #101b24;
            --review-table-text: #dce8f2;
            --review-table-border: rgba(84, 118, 140, 0.35);
            --review-table-header-border: rgba(84, 118, 140, 0.45);
        }
        body[data-theme="gray"] {
            background-color: #2e3440 !important;
            color: #d8dee9 !important;
            --review-table-cell-bg: #2f3744;
            --review-table-header-bg: #3b4252;
            --review-table-text: #eceff4;
            --review-table-border: #4c566a;
            --review-table-header-border: #596778;
        }
        body[data-theme="white"] {
            background-color: #eef2f6 !important;
            color: #1c2733 !important;
            --review-table-cell-bg: #ffffff;
            --review-table-header-bg: #edf4fa;
            --review-table-text: #1c2733;
            --review-table-border: #d6e0e8;
            --review-table-header-border: #c5d0da;
        }
        body[data-theme="black"] .main-container,
        body[data-theme="gray"] .main-container,
        body[data-theme="white"] .main-container {
            background-color: inherit !important;
        }
        body[data-theme="black"] .sidebar,
        body[data-theme="black"] .header-bar,
        body[data-theme="black"] .metadata-sections,
        body[data-theme="black"] .control-bar,
        body[data-theme="black"] .review-form,
        body[data-theme="black"] .plot-status,
        body[data-theme="black"] .run-config-item,
        body[data-theme="black"] .plot-toolbar { background-color: #0a0a0a !important; border-color: #555 !important; color: #e0e0e0 !important; }
        body[data-theme="gray"] .sidebar,
        body[data-theme="gray"] .header-bar,
        body[data-theme="gray"] .metadata-sections,
        body[data-theme="gray"] .control-bar,
        body[data-theme="gray"] .review-form,
        body[data-theme="gray"] .plot-status,
        body[data-theme="gray"] .run-config-item,
        body[data-theme="gray"] .plot-toolbar { background-color: #3b4252 !important; border-color: #4c566a !important; color: #d8dee9 !important; }
        body[data-theme="white"] .sidebar,
        body[data-theme="white"] .header-bar,
        body[data-theme="white"] .metadata-sections,
        body[data-theme="white"] .control-bar,
        body[data-theme="white"] .review-form,
        body[data-theme="white"] .plot-status,
        body[data-theme="white"] .run-config-item,
        body[data-theme="white"] .plot-toolbar,
        body[data-theme="white"] .eda-panel-inner,
        body[data-theme="white"] .eda-graph-card,
        body[data-theme="white"] .eda-table-card,
        body[data-theme="white"] .queue-provenance-panel,
        body[data-theme="white"] .stat-card,
        body[data-theme="white"] .lazy-panel-placeholder,
        body[data-theme="white"] .pipeline-log-panel,
        body[data-theme="white"] .plot-frame { background-color: #ffffff !important; border-color: #c5d0da !important; color: #1c2733 !important; }
        body[data-theme="black"] .section-title,
        body[data-theme="black"] .help-link,
        body[data-theme="black"] .metadata-sections summary,
        body[data-theme="black"] #progress-text { color: #0af !important; }
        body[data-theme="gray"] .section-title,
        body[data-theme="gray"] .help-link,
        body[data-theme="gray"] .metadata-sections summary,
        body[data-theme="gray"] #progress-text { color: #88c0d0 !important; }
        body[data-theme="white"] .section-title,
        body[data-theme="white"] .help-link,
        body[data-theme="white"] .metadata-sections summary,
        body[data-theme="white"] #progress-text { color: #245f8f !important; }
        body[data-theme="black"] .action-btn.primary { background-color: #0af !important; color: #08131d !important; border-color: #0af !important; }
        body[data-theme="gray"] .action-btn.primary { background-color: #88c0d0 !important; color: #2e3440 !important; border-color: #88c0d0 !important; }
        body[data-theme="white"] .action-btn.primary { background-color: #245f8f !important; color: #f5f7fa !important; border-color: #245f8f !important; }
        body[data-theme="black"] input, body[data-theme="black"] textarea, body[data-theme="black"] select,
        body[data-theme="black"] .dash-dropdown .Select-control,
        body[data-theme="black"] .dash-dropdown .Select-menu-outer { background-color: #0a0a0a !important; color: #e0e0e0 !important; border-color: #555 !important; }
        body[data-theme="gray"] input, body[data-theme="gray"] textarea, body[data-theme="gray"] select,
        body[data-theme="gray"] .dash-dropdown .Select-control,
        body[data-theme="gray"] .dash-dropdown .Select-menu-outer { background-color: #3b4252 !important; color: #eceff4 !important; border-color: #4c566a !important; }
        body[data-theme="white"] input, body[data-theme="white"] textarea, body[data-theme="white"] select,
        body[data-theme="white"] .dash-dropdown .Select-control,
        body[data-theme="white"] .dash-dropdown .Select-menu-outer { background-color: #ffffff !important; color: #1c2733 !important; border-color: #c5d0da !important; }
        body[data-theme="white"] .plot-container,
        body[data-theme="white"] .metadata-bar,
        body[data-theme="white"] #bottom-context-info {
            background-color: #eef2f6 !important;
            border-color: #c5d0da !important;
            color: #4f6273 !important;
        }
        body[data-theme="white"] .plot-toolbar,
        body[data-theme="white"] .eda-panel-inner,
        body[data-theme="white"] .eda-graph-card,
        body[data-theme="white"] .eda-table-card,
        body[data-theme="white"] .meta-toolbar,
        body[data-theme="white"] .camera-diag .item,
        body[data-theme="white"] .run-config-item,
        body[data-theme="white"] .repro-badge,
        body[data-theme="white"] .metadata-health,
        body[data-theme="white"] .queue-provenance-panel,
        body[data-theme="white"] .stat-card,
        body[data-theme="white"] .plot-frame,
        body[data-theme="white"] .compact-btn,
        body[data-theme="white"] .score-btn,
        body[data-theme="white"] .badge-btn,
        body[data-theme="white"] .action-btn:not(.primary),
        body[data-theme="white"] .sidebar-toggle,
        body[data-theme="white"] #help-modal .modal-content {
            background: #ffffff !important;
            background-image: none !important;
            border-color: #c5d0da !important;
            color: #1c2733 !important;
        }
        body[data-theme="white"] .sidebar-toggle {
            color: #245f8f !important;
        }
        body[data-theme="white"] .sidebar-toggle:hover,
        body[data-theme="white"] .compact-btn:hover,
        body[data-theme="white"] .score-btn:hover,
        body[data-theme="white"] .badge-btn:hover,
        body[data-theme="white"] .action-btn:not(.primary):hover {
            background: #e7edf3 !important;
            border-color: #9fb1bf !important;
            color: #1c2733 !important;
        }
        body[data-theme="white"] .score-btn.active {
            background: #dbe7f1 !important;
            border-color: #245f8f !important;
            color: #163b57 !important;
        }
        body[data-theme="white"] .badge-btn.active {
            background: #e8f7ec !important;
            border-color: #2f7a57 !important;
            color: #2f7a57 !important;
        }
        body[data-theme="white"] .header-key-info .item,
        body[data-theme="white"] .notification,
        body[data-theme="white"] .camera-diag,
        body[data-theme="white"] .run-config-item .k,
        body[data-theme="white"] .plot-toolbar .label-chip,
        body[data-theme="white"] .plot-control-label,
        body[data-theme="white"] .eda-panel-title,
        body[data-theme="white"] .plot-toolbar .dash-checklist label,
        body[data-theme="white"] .plot-toolbar label,
        body[data-theme="white"] .meta-toolbar .title,
        body[data-theme="white"] .metadata-health .detail,
        body[data-theme="white"] .queue-provenance-list,
        body[data-theme="white"] .queue-provenance-note,
        body[data-theme="white"] .stat-card .label,
        body[data-theme="white"] .plot-status summary,
        body[data-theme="white"] .sidebar label,
        body[data-theme="white"] .sidebar details summary,
        body[data-theme="white"] .dash-checklist label,
        body[data-theme="white"] .sidebar-camera-checklist label,
        body[data-theme="white"] #review-progress-indicator,
        body[data-theme="white"] #pdm-result-label,
        body[data-theme="white"] #bottom-pipeline-status,
        body[data-theme="white"] #pass-indicator,
        body[data-theme="white"] #status-indicator {
            color: #4f6273 !important;
        }
        body[data-theme="white"] .sidebar details summary:hover,
        body[data-theme="white"] .metadata-sections summary:hover,
        body[data-theme="white"] .plot-status summary:hover,
        body[data-theme="white"] .help-link:hover {
            color: #245f8f !important;
        }
        body[data-theme="white"] .run-config-item .v,
        body[data-theme="white"] .plot-status,
        body[data-theme="white"] .plot-status .status-line,
        body[data-theme="white"] .eda-status-line,
        body[data-theme="white"] .eda-field-label,
        body[data-theme="white"] .plot-status li,
        body[data-theme="white"] .pipeline-log-panel,
        body[data-theme="white"] .metadata-health,
        body[data-theme="white"] .bottom-context-v,
        body[data-theme="white"] .meta-field-label,
        body[data-theme="white"] .meta-field-value,
        body[data-theme="white"] .lazy-panel-placeholder,
        body[data-theme="white"] .stat-card .value,
        body[data-theme="white"] .vetting-banner-label,
        body[data-theme="white"] .vetting-banner-value,
        body[data-theme="white"] .vetting-banner-empty,
        body[data-theme="white"] #help-modal .modal-body,
        body[data-theme="white"] #help-modal .modal-footer,
        body[data-theme="white"] #help-modal pre {
            color: #1c2733 !important;
        }
        body[data-theme="white"] .bottom-context-k {
            color: #5f7384 !important;
        }
        body[data-theme="white"] .meta-field-row {
            border-color: #d6e0e8 !important;
        }
        body[data-theme="white"] .metadata-copy-btn {
            background: #edf4fa !important;
            border-color: #b9c9d7 !important;
            color: #245f8f !important;
        }
        body[data-theme="white"] .metadata-copy-btn.copied {
            color: #2f7a57 !important;
            border-color: #a8d0ba !important;
        }
        body[data-theme="white"] .metadata-copy-btn.copy-failed {
            color: #a53a3a !important;
            border-color: #d9aaaa !important;
        }
        body[data-theme="white"] .lazy-panel-placeholder-error {
            color: #a53a3a !important;
            border-color: #d9aaaa !important;
            background: #fff0f0 !important;
        }
        body[data-theme="white"] .dustycult-param-table th,
        body[data-theme="white"] .dustycult-param-table td {
            border-color: #d6e0e8 !important;
        }
        body[data-theme="white"] .pipeline-log-panel {
            background: #f7fafc !important;
            border-color: #c5d0da !important;
            color: #1c2733 !important;
        }
        body[data-theme="white"] .eda-table-card .dash-table-container,
        body[data-theme="white"] #eda-candidate-table,
        body[data-theme="white"] #eda-candidate-table .dash-spreadsheet-container,
        body[data-theme="white"] #eda-candidate-table .dash-spreadsheet-inner,
        body[data-theme="white"] #eda-candidate-table .dash-spreadsheet,
        body[data-theme="white"] #eda-candidate-table table {
            background: #ffffff !important;
            background-image: none !important;
            color: #1c2733 !important;
            border-color: #d6e0e8 !important;
        }
        body[data-theme="white"] #eda-candidate-table input {
            background: #ffffff !important;
            color: #1c2733 !important;
            border-color: #c5d0da !important;
        }
        body[data-theme="white"] .plot-status.warn {
            border-color: rgba(186, 144, 44, 0.45) !important;
            background: rgba(255, 239, 202, 0.88) !important;
        }
        body[data-theme="white"] .plot-status.error {
            border-color: rgba(192, 72, 72, 0.45) !important;
            background: rgba(255, 226, 226, 0.92) !important;
        }
        body[data-theme="white"] .metadata-health .chip,
        body[data-theme="white"] .repro-badge {
            color: #245f8f !important;
            background: #edf4fa !important;
            border-color: #b9c9d7 !important;
        }
        body[data-theme="white"] .metadata-health.metadata-health-base .chip {
            color: #946200 !important;
            border-color: #e0c27b !important;
            background: #fff3d8 !important;
        }
        body[data-theme="white"] .metadata-health.metadata-health-partial .chip {
            color: #1f6485 !important;
            border-color: #a7cad9 !important;
            background: #e8f5fb !important;
        }
        body[data-theme="white"] .metadata-health.metadata-health-enriched .chip {
            color: #2f7a57 !important;
            border-color: #a8d0ba !important;
            background: #e9f7ef !important;
        }
        body[data-theme="white"] .repro-badge.warn {
            color: #946200 !important;
            border-color: #e0c27b !important;
            background: #fff3d8 !important;
        }
        body[data-theme="white"] .sidebar hr,
        body[data-theme="white"] .metadata-sections details,
        body[data-theme="white"] .dash-checklist label,
        body[data-theme="white"] .dash-radioitems label {
            box-shadow: none !important;
        }
        body[data-theme="white"] .panel-splitter-vertical::after {
            color: #4f6273 !important;
            background: rgba(255, 255, 255, 0.96) !important;
            border-color: rgba(159, 177, 191, 0.75) !important;
        }
        body[data-theme="white"] .panel-splitter-vertical::before {
            background: rgba(159, 177, 191, 0.6) !important;
        }
        body[data-theme="white"] .Select-control,
        body[data-theme="white"] .Select-menu-outer,
        body[data-theme="white"] .Select-menu,
        body[data-theme="white"] .Select-option,
        body[data-theme="white"] .VirtualizedSelectOption,
        body[data-theme="white"] .Select-placeholder,
        body[data-theme="white"] .Select-value,
        body[data-theme="white"] .Select-value-label,
        body[data-theme="white"] .Select-input,
        body[data-theme="white"] .Select-clear-zone,
        body[data-theme="white"] .Select-arrow-zone,
        body[data-theme="white"] .Select-menu-outer,
        body[data-theme="white"] .Select-menu-outer *,
        body[data-theme="white"] .Select * {
            background-color: #ffffff !important;
            color: #1c2733 !important;
            border-color: #c5d0da !important;
        }
        body[data-theme="white"] .Select-option,
        body[data-theme="white"] .VirtualizedSelectOption,
        body[data-theme="white"] [class*="Select-option"] {
            border-bottom: 1px solid #dde6ee !important;
        }
        body[data-theme="white"] .form-select,
        body[data-theme="white"] .form-control,
        body[data-theme="white"] .sidebar .form-select,
        body[data-theme="white"] .sidebar .form-control,
        body[data-theme="white"] .plot-toolbar .form-select,
        body[data-theme="white"] .plot-toolbar .form-control {
            background-color: #ffffff !important;
            background-image: none !important;
            color: #1c2733 !important;
            border-color: #c5d0da !important;
            box-shadow: none !important;
        }
        body[data-theme="white"] .form-select:focus,
        body[data-theme="white"] .form-control:focus,
        body[data-theme="white"] .sidebar .form-select:focus,
        body[data-theme="white"] .sidebar .form-control:focus,
        body[data-theme="white"] .plot-toolbar .form-select:focus,
        body[data-theme="white"] .plot-toolbar .form-control:focus {
            border-color: #7da8c4 !important;
            box-shadow: 0 0 0 2px rgba(36, 95, 143, 0.12) !important;
        }
        body[data-theme="white"] .dash-dropdown,
        body[data-theme="white"] .dash-dropdown-trigger,
        body[data-theme="white"] .dash-dropdown-value,
        body[data-theme="white"] .dash-dropdown-value-item,
        body[data-theme="white"] .dash-dropdown-content,
        body[data-theme="white"] .dash-dropdown-options,
        body[data-theme="white"] .dash-options-list,
        body[data-theme="white"] .dash-dropdown-search-container,
        body[data-theme="white"] .dash-dropdown-search {
            background-color: #ffffff !important;
            background-image: none !important;
            color: #1c2733 !important;
            border-color: #c5d0da !important;
        }
        body[data-theme="white"] .dash-dropdown-option,
        body[data-theme="white"] .dash-options-list-option {
            background-color: #ffffff !important;
            color: #1c2733 !important;
            border-color: #dde6ee !important;
        }
        body[data-theme="white"] .dash-dropdown-option:hover,
        body[data-theme="white"] .dash-options-list-option:hover,
        body[data-theme="white"] .dash-dropdown-option.selected,
        body[data-theme="white"] .dash-options-list-option.selected {
            background-color: #eaf1f6 !important;
            color: #1c2733 !important;
        }
        body[data-theme="white"] .Select-control .Select-input > input {
            color: #1c2733 !important;
        }
        body[data-theme="white"] .Select.has-value.Select--single > .Select-control .Select-value,
        body[data-theme="white"] .Select.has-value.is-pseudo-focused.Select--single > .Select-control .Select-value {
            background: #ffffff !important;
            background-image: none !important;
            border-color: transparent !important;
            color: #1c2733 !important;
        }
        body[data-theme="white"] .Select.has-value.Select--single > .Select-control .Select-value .Select-value-label,
        body[data-theme="white"] .Select.has-value.is-pseudo-focused.Select--single > .Select-control .Select-value .Select-value-label,
        body[data-theme="white"] .has-value.Select--single > .Select-control .Select-value a.Select-value-label,
        body[data-theme="white"] .has-value.is-pseudo-focused.Select--single > .Select-control .Select-value a.Select-value-label {
            color: #1c2733 !important;
        }
        body[data-theme="white"] .Select.is-focused:not(.is-open) > .Select-control {
            border-color: #7da8c4 !important;
            box-shadow: 0 0 0 2px rgba(36, 95, 143, 0.12) !important;
        }
        body[data-theme="white"] .sidebar .dash-checklist,
        body[data-theme="white"] .sidebar .dash-radioitems,
        body[data-theme="white"] .meta-toolbar .dash-checklist,
        body[data-theme="white"] .meta-toolbar .dash-radioitems,
        body[data-theme="white"] .plot-toolbar .dash-checklist,
        body[data-theme="white"] .plot-toolbar .dash-radioitems {
            background: transparent !important;
        }
        body[data-theme="white"] .sidebar .dash-checklist label,
        body[data-theme="white"] .sidebar .dash-radioitems label,
        body[data-theme="white"] .sidebar-camera-checklist label {
            background: #f5f8fb !important;
            border: 1px solid #d6e0e8 !important;
            color: #1c2733 !important;
        }
        body[data-theme="white"] .meta-toolbar .dash-checklist label,
        body[data-theme="white"] .meta-toolbar .dash-radioitems label,
        body[data-theme="white"] .meta-toolbar label,
        body[data-theme="white"] .plot-toolbar .dash-checklist label,
        body[data-theme="white"] .plot-toolbar .dash-radioitems label,
        body[data-theme="white"] .plot-toolbar label {
            background: #f5f8fb !important;
            border: 1px solid #d6e0e8 !important;
            color: #1c2733 !important;
        }
        body[data-theme="white"] .sidebar .dash-checklist label:hover,
        body[data-theme="white"] .sidebar .dash-radioitems label:hover,
        body[data-theme="white"] .sidebar-camera-checklist label:hover,
        body[data-theme="white"] .meta-toolbar .dash-checklist label:hover,
        body[data-theme="white"] .meta-toolbar .dash-radioitems label:hover,
        body[data-theme="white"] .meta-toolbar label:hover,
        body[data-theme="white"] .plot-toolbar .dash-checklist label:hover,
        body[data-theme="white"] .plot-toolbar .dash-radioitems label:hover,
        body[data-theme="white"] .plot-toolbar label:hover {
            background: #eaf1f6 !important;
            border-color: #b8c8d5 !important;
        }
        body[data-theme="white"] #sidebar-status {
            color: #2f7a57 !important;
        }
        body[data-theme="white"] #help-modal .modal-content,
        body[data-theme="white"] #help-modal .modal-header,
        body[data-theme="white"] #help-modal .modal-body,
        body[data-theme="white"] #help-modal .modal-footer {
            background-color: #ffffff !important;
            border-color: #c5d0da !important;
        }
        body[data-theme="white"] .vetting-banner-empty,
        body[data-theme="white"] .vetting-banner-grid,
        body[data-theme="white"] .vetting-banner-links {
            background: #ffffff !important;
            border-color: #c5d0da !important;
        }
        body[data-theme="white"] .vetting-banner-cell {
            border-color: #d6e0e8 !important;
        }
        body[data-theme="white"] .vetting-banner-shell.known .vetting-banner-cell,
        body[data-theme="white"] .vetting-banner-shell.new .vetting-banner-cell {
            background: #f7fafc !important;
        }
        body[data-theme="white"] .vetting-banner-cell.hit.known {
            background: #fbe7e7 !important;
            border-color: #d88b8b !important;
        }
        body[data-theme="white"] .vetting-banner-cell.hit.new {
            background: #e8f7ec !important;
            border-color: #95cca3 !important;
        }
        body[data-theme="white"] .vetting-banner-header.known {
            background: #fbe7e7 !important;
            color: #9f2d2d !important;
            border-color: #e4b4b4 !important;
        }
        body[data-theme="white"] .vetting-banner-header.new {
            background: #e8f7ec !important;
            color: #2f7a57 !important;
            border-color: #b7dcbf !important;
        }
        body[data-theme="white"] .vetting-banner-hit.known {
            color: #9f2d2d !important;
        }
        body[data-theme="white"] .vetting-banner-hit.new {
            color: #2f7a57 !important;
        }
        body[data-theme="white"] .vetting-banner-link {
            background: #f5f8fb !important;
            border-color: #c5d0da !important;
            color: #245f8f !important;
        }
    </style>
</head>
<body>
    {%app_entry%}
    {%config%}
    {%scripts%}
    {%renderer%}
</body>
</html>
'''
