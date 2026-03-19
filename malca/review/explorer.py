from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import glob
import json
import os
from pathlib import Path
import re
from threading import Timer
from typing import Any
import webbrowser

import dash
from dash import ALL, Input, Output, State, dash_table, dcc, html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from malca.review.app import _render_stat_cards
from malca.review.explore_data import (
    BEST_FIELDS,
    DEFAULT_COLOR,
    DEFAULT_MAIN_X,
    DEFAULT_MAIN_Y,
    DEFAULT_SYMBOL,
    CombinedCandidateData,
    add_eda_columns,
    _normalized_id,
    available_metric_columns,
    bool_series,
    cut_summary,
    discover_default_sources,
    find_candidate_key,
    get_candidate_record_by_key,
    infer_plot_dir_for_record,
    infer_plot_dir_from_source,
    load_combined_source_data,
    load_run_params,
    normalize_review_label,
    numeric_series,
    text_series,
)
from malca.review.filter_schema import SIDEBAR_GROUPS
from malca.review.handoff import build_review_command, launch_detached
from malca.review.interactive_plot import build_interactive_lightcurve_figure, resolve_lightcurve_path as review_resolve_lightcurve_path
from malca.review.keyboard import CLASS_KEY_MAP
from malca.review.period_search import has_external_period, run_period_search_for_payload
from malca.review.store import db_connect, export_review_subset_bundle, get_review, load_app_state, save_app_state, save_review


DEFAULT_THEME = "black"

APP_BG = "var(--explorer-app-bg)"
PANEL_BG = "var(--explorer-panel-bg)"
PANEL_BG_ALT = "var(--explorer-panel-bg-alt)"
PANEL_BORDER = "var(--explorer-panel-border)"
TEXT = "var(--explorer-text)"
TEXT_MUTED = "var(--explorer-text-muted)"
TEXT_FAINT = "var(--explorer-text-faint)"
ACCENT = "var(--explorer-accent)"
ACCENT_2 = "var(--explorer-accent-2)"

UI_FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

EXPLORER_PLOT_KEYS = (
    "custom",
    "lightcurve",
)

BASE_INPUT_STYLE = {
    "width": "100%",
    "padding": "8px 10px",
    "background": PANEL_BG,
    "color": TEXT,
    "border": f"1px solid {PANEL_BORDER}",
    "borderRadius": "8px",
    "fontFamily": UI_FONT_FAMILY,
    "fontSize": "12px",
}

PANEL_STYLE = {
    "background": PANEL_BG_ALT,
    "border": f"1px solid {PANEL_BORDER}",
    "borderRadius": "14px",
    "padding": "14px",
    "boxShadow": "0 12px 28px rgba(15, 23, 32, 0.08)",
}

AUTO_FILTER_EXCLUDE_COLUMNS = {
    "candidate_id",
    "asas_sn_id",
    "gaia_id",
    "candidate_key",
    "source_file",
    "source_label",
    "source_path",
    "plot_dir",
    "path",
    "lc_path",
    "payload_json",
    "notes",
    "imported_at",
}

AUTO_FILTER_TEXT_ONLY_MAX_UNIQUES = 200
_EXPLORER_GUI_STATE_APP_STATE_KEY = "dash_explorer_gui_state_v1"

ADV_FILTER_INPUTS = [
    Input({"type": "adv-bool-mode", "col": ALL}, "value"),
    Input({"type": "adv-num-min", "col": ALL}, "value"),
    Input({"type": "adv-num-max", "col": ALL}, "value"),
    Input({"type": "adv-text-value", "col": ALL}, "value"),
    Input({"type": "adv-select-exclude", "col": ALL}, "value"),
    Input("only-unreviewed", "value"),
    Input("require-failed-any-false", "value"),
]

ADV_FILTER_STATES = [
    State({"type": "adv-bool-mode", "col": ALL}, "id"),
    State({"type": "adv-num-min", "col": ALL}, "id"),
    State({"type": "adv-num-max", "col": ALL}, "id"),
    State({"type": "adv-text-value", "col": ALL}, "id"),
    State({"type": "adv-select-exclude", "col": ALL}, "id"),
]

REVIEW_EVENT_CLASSES = list(dict.fromkeys(["unclassified", *CLASS_KEY_MAP.values(), "yso"]))
REVIEWED_CLASSES = {
    "dipper",
    "yso",
    "microlensing",
    "flare",
    "instrumental",
    "unknown_interesting",
    "other",
    "ltv",
}
REVIEW_CLASS_OPTIONS = [
    {"label": "Unclassified", "value": "unclassified"},
    {"label": "Dipper", "value": "dipper"},
    {"label": "Microlensing", "value": "microlensing"},
    {"label": "Flare", "value": "flare"},
    {"label": "YSO", "value": "yso"},
    {"label": "LTV", "value": "ltv"},
    {"label": "Unknown interesting", "value": "unknown_interesting"},
    {"label": "Instrumental", "value": "instrumental"},
    {"label": "Other", "value": "other"},
]
REVIEW_SCORE_OPTIONS = [
    {"label": "1", "value": 1},
    {"label": "2", "value": 2},
    {"label": "3", "value": 3},
    {"label": "4", "value": 4},
]


def _coerce_optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if np.isfinite(numeric) else None


def _coerce_string_list(raw_value: object) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        values = [raw_value]
    elif isinstance(raw_value, (list, tuple, set, np.ndarray, pd.Series)):
        values = list(raw_value)
    else:
        values = [raw_value]
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _coerce_explorer_bool_mode_value(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    return text if text in {"Any", "True", "False", "Unset"} else "Any"


def _coerce_explorer_text_value(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    return text or "Any"


def _explorer_state_db_path(combined: CombinedCandidateData) -> Path | None:
    db_paths: list[Path] = []
    seen: set[str] = set()
    for source in combined.sources:
        if str(getattr(source, "source_kind", "")).lower() != "db":
            continue
        try:
            path = Path(str(source.source_path)).expanduser().resolve()
        except Exception:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        db_paths.append(path)
    return db_paths[0] if len(db_paths) == 1 else None


def _explorer_gui_state_from_values(
    *,
    source_filter: object,
    query_value: object,
    advanced_state: dict[str, object],
    x_metric: object,
    y_metric: object,
    color_metric: object,
    symbol_metric: object,
    log_flags: object,
    x_min: object,
    x_max: object,
    y_min: object,
    y_max: object,
    table_sort: object,
    selected_key: object,
    theme_mode: object,
    candidate_scope_state: dict[str, object] | None,
    panel_options: object,
    period_value: object,
    yaxis_mode: object,
    period_method: object,
    period_min: object,
    period_max: object,
    selected_cameras: object,
    selected_bands: object,
) -> dict[str, object]:
    return {
        "source_filter": _coerce_string_list(source_filter),
        "query_value": "" if query_value is None else str(query_value),
        "advanced": dict(advanced_state or {}),
        "x_metric": str(x_metric).strip() if x_metric not in (None, "") else None,
        "y_metric": str(y_metric).strip() if y_metric not in (None, "") else None,
        "color_metric": str(color_metric).strip() if color_metric not in (None, "") else None,
        "symbol_metric": str(symbol_metric).strip() if symbol_metric not in (None, "") else None,
        "log_flags": _coerce_string_list(log_flags),
        "x_min": _coerce_optional_float(x_min),
        "x_max": _coerce_optional_float(x_max),
        "y_min": _coerce_optional_float(y_min),
        "y_max": _coerce_optional_float(y_max),
        "table_sort": str(table_sort).strip() if table_sort not in (None, "") else None,
        "selected_key": str(selected_key).strip() if selected_key not in (None, "") else "",
        "theme_mode": "white" if str(theme_mode or "").strip() == "white" else "black",
        "candidate_scope_state": dict(candidate_scope_state or {"mode": "filtered"}),
        "panel_options": _coerce_string_list(panel_options),
        "period_value": _coerce_optional_float(period_value),
        "yaxis_mode": "flux" if str(yaxis_mode or "").strip() == "flux" else "mag",
        "period_method": str(period_method).strip().lower() if period_method not in (None, "") else "pdm",
        "period_min": _coerce_optional_float(period_min),
        "period_max": _coerce_optional_float(period_max),
        "camera_values": _coerce_string_list(selected_cameras),
        "band_values": _coerce_string_list(selected_bands),
    }


def _normalize_explorer_gui_state(raw_state: object) -> dict[str, object] | None:
    if not isinstance(raw_state, dict) or not raw_state:
        return None
    return _explorer_gui_state_from_values(
        source_filter=raw_state.get("source_filter"),
        query_value=raw_state.get("query_value"),
        advanced_state=dict(raw_state.get("advanced") or {}),
        x_metric=raw_state.get("x_metric"),
        y_metric=raw_state.get("y_metric"),
        color_metric=raw_state.get("color_metric"),
        symbol_metric=raw_state.get("symbol_metric"),
        log_flags=raw_state.get("log_flags"),
        x_min=raw_state.get("x_min"),
        x_max=raw_state.get("x_max"),
        y_min=raw_state.get("y_min"),
        y_max=raw_state.get("y_max"),
        table_sort=raw_state.get("table_sort"),
        selected_key=raw_state.get("selected_key"),
        theme_mode=raw_state.get("theme_mode"),
        candidate_scope_state=dict(raw_state.get("candidate_scope_state") or {"mode": "filtered"}),
        panel_options=raw_state.get("panel_options"),
        period_value=raw_state.get("period_value"),
        yaxis_mode=raw_state.get("yaxis_mode"),
        period_method=raw_state.get("period_method"),
        period_min=raw_state.get("period_min"),
        period_max=raw_state.get("period_max"),
        selected_cameras=raw_state.get("camera_values"),
        selected_bands=raw_state.get("band_values"),
    )


def _explorer_advanced_ui_values_from_state(
    saved_state: dict[str, object] | None,
    *,
    bool_ids: list[dict[str, object]] | None,
    num_min_ids: list[dict[str, object]] | None,
    num_max_ids: list[dict[str, object]] | None,
    text_ids: list[dict[str, object]] | None,
    select_ids: list[dict[str, object]] | None,
) -> tuple[list[object], list[object], list[object], list[object], list[object], list[str], list[str]]:
    adv = dict((saved_state or {}).get("advanced") or {})
    bool_map = dict(adv.get("bool") or {})
    num_map = dict(adv.get("num") or {})
    text_map = dict(adv.get("text") or {})
    select_map = dict(adv.get("select") or {})

    bool_values = [
        _coerce_explorer_bool_mode_value(bool_map.get(str((meta or {}).get("col") or "")))
        for meta in (bool_ids or [])
    ]
    num_min_values = [
        _coerce_optional_float(dict(num_map.get(str((meta or {}).get("col") or "")) or {}).get("min"))
        for meta in (num_min_ids or [])
    ]
    num_max_values = [
        _coerce_optional_float(dict(num_map.get(str((meta or {}).get("col") or "")) or {}).get("max"))
        for meta in (num_max_ids or [])
    ]
    text_values = [
        _coerce_explorer_text_value(text_map.get(str((meta or {}).get("col") or "")))
        for meta in (text_ids or [])
    ]
    select_values = [
        _coerce_string_list(select_map.get(str((meta or {}).get("col") or "")))
        for meta in (select_ids or [])
    ]
    only_unreviewed = ["yes"] if bool(adv.get("only_unreviewed")) else []
    require_failed = ["yes"] if bool(adv.get("require_failed_any_false")) else []
    return (
        bool_values,
        num_min_values,
        num_max_values,
        text_values,
        select_values,
        only_unreviewed,
        require_failed,
    )


def _summary_items(record: dict[str, object]) -> list[tuple[str, str]]:
    fields = [
        ("Candidate", record.get("candidate_id")),
        ("ASAS-SN", record.get("asas_sn_id")),
        ("Gaia", record.get("gaia_id")),
        ("Final class", record.get("final_class")),
        ("Known?", record.get("vetting_likely_known")),
        ("Period (d)", record.get("period_consensus_days")),
        ("N sources", record.get("period_n_sources")),
        ("Dipper score", record.get("dipper_score")),
    ]
    items: list[tuple[str, str]] = []
    for label, value in fields:
        text = _normalized_id(value)
        if text:
            items.append((label, text))
    return items


def _normalize_review_state(review: dict[str, object] | None) -> dict[str, object]:
    raw = dict(review or {})
    score = raw.get("interest_score")
    try:
        score_val = None if score in (None, "") else int(np.clip(int(score), 1, 4))
    except Exception:
        score_val = None

    event_class = str(raw.get("event_class") or "unclassified").strip() or "unclassified"
    if event_class not in REVIEW_EVENT_CLASSES:
        event_class = "other"

    try:
        review_pass = max(1, int(raw.get("review_pass") or 1))
    except Exception:
        review_pass = 1

    status = str(raw.get("status") or "unreviewed").strip() or "unreviewed"
    if status not in {"reviewed", "needs_followup", "unreviewed"}:
        status = "reviewed"

    return {
        "interest_score": score_val,
        "event_class": event_class,
        "review_pass": review_pass,
        "notes": "" if raw.get("notes") is None else str(raw.get("notes")),
        "status": status,
        "reviewer": "" if raw.get("reviewer") is None else str(raw.get("reviewer")),
        "updated_at": raw.get("updated_at"),
    }


def _record_review_state(record: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(record, dict):
        return _normalize_review_state(None)
    return _normalize_review_state({
        "interest_score": record.get("interest_score"),
        "event_class": record.get("event_class"),
        "review_pass": record.get("review_pass"),
        "notes": record.get("notes"),
        "status": record.get("status"),
        "reviewer": record.get("reviewer"),
        "updated_at": record.get("updated_at"),
    })


def _apply_review_override_to_record(
    record: dict[str, object] | None,
    review_overrides: dict[str, object] | None,
) -> dict[str, object] | None:
    if not isinstance(record, dict):
        return record
    key = str(record.get("candidate_key") or "").strip()
    if not key:
        return dict(record)
    override = dict((review_overrides or {}).get(key) or {})
    if not override:
        return dict(record)
    merged = dict(record)
    merged.update(_normalize_review_state(override))
    return merged


def _apply_review_overrides(
    frame: pd.DataFrame,
    review_overrides: dict[str, object] | None,
) -> pd.DataFrame:
    if frame.empty or "candidate_key" not in frame.columns or not review_overrides:
        return frame

    out = frame.copy()
    override_map = {
        str(key): _normalize_review_state(value)
        for key, value in dict(review_overrides or {}).items()
        if str(key).strip()
    }
    if not override_map:
        return out

    key_series = out["candidate_key"].astype(str)
    for field in ("interest_score", "event_class", "review_pass", "notes", "status", "reviewer", "updated_at"):
        if field not in out.columns:
            out[field] = pd.NA

    for key, review in override_map.items():
        mask = key_series.eq(key)
        if not bool(mask.any()):
            continue
        for field, value in review.items():
            out.loc[mask, field] = value

    review_label = normalize_review_label(text_series(out, "event_class"))
    if "review_event_class" in out.columns:
        review_label = review_label.where(
            review_label.ne(""),
            normalize_review_label(text_series(out, "review_event_class")),
        )
    out["review_label"] = review_label
    out["is_reviewed"] = review_label.isin(REVIEWED_CLASSES)
    out["is_reviewed_dipper"] = review_label.eq("dipper")
    out["is_reviewed_non_dipper"] = out["is_reviewed"] & (~out["is_reviewed_dipper"])
    return out


def _get_candidate_record_from_frame(frame: pd.DataFrame, candidate_key: str | None) -> dict[str, object] | None:
    if frame.empty or "candidate_key" not in frame.columns:
        return None
    key = str(candidate_key or "").strip()
    if not key:
        try:
            key = str(frame.iloc[0].get("candidate_key") or "")
        except Exception:
            key = ""
    if not key:
        return None
    matches = frame.loc[frame["candidate_key"].astype(str).eq(key)]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return row.to_dict() if isinstance(row, pd.Series) else None


def _render_summary(record: dict[str, object]) -> html.Div:
    items = _summary_items(record)
    if not items:
        return html.Div("No candidate summary available.", style={"color": TEXT_MUTED, "fontSize": "12px"})
    return html.Div(
        [
            html.Div(
                [
                    html.Span(f"{label}: ", style={"fontSize": "11px", "color": TEXT_FAINT, "fontWeight": "600"}),
                    html.Span(value, style={"fontSize": "11px", "color": TEXT}),
                ],
                style={"padding": "2px 0", "display": "flex", "gap": "4px", "flexWrap": "wrap"},
            )
            for label, value in items
        ],
        style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(180px, 1fr))", "columnGap": "14px", "rowGap": "2px"},
    )


def _render_stats(stat_rows: list[tuple[str, str]]) -> html.Div:
    if not stat_rows:
        return html.Div("No precomputed light-curve stats available.", style={"color": TEXT_MUTED, "fontSize": "12px"})
    return html.Div(_render_stat_cards(stat_rows), className="explorer-review-stats")


def _resolve_sources(args) -> list[Path]:
    if args.source:
        return [Path(s).expanduser().resolve() for s in args.source]
    if args.source_glob:
        return [Path(p).expanduser().resolve() for p in sorted(glob.glob(args.source_glob))]
    return discover_default_sources()


def _theme_palette(theme: str | None) -> dict[str, str]:
    if str(theme or DEFAULT_THEME).strip().lower() == "white":
        return {
            "template": "plotly_white",
            "paper_bg": "#f2f6fa",
            "plot_bg": "#ffffff",
            "text": "#182633",
            "text_muted": "#4d6476",
            "accent": "#245f8f",
            "accent_2": "#b05e2d",
            "grid": "rgba(74, 101, 122, 0.18)",
            "legend_bg": "rgba(255, 255, 255, 0.96)",
            "legend_border": "rgba(133, 157, 177, 0.55)",
            "hover_bg": "#ffffff",
            "marker_line": "rgba(24, 38, 51, 0.22)",
        }
    return {
        "template": "plotly_dark",
        "paper_bg": "#0c1218",
        "plot_bg": "#111a23",
        "text": "#f4f8fb",
        "text_muted": "#a6bac9",
        "accent": "#6fd4ff",
        "accent_2": "#ffb36f",
        "grid": "rgba(140, 170, 192, 0.18)",
        "legend_bg": "rgba(15, 23, 32, 0.78)",
        "legend_border": "rgba(120, 150, 170, 0.22)",
        "hover_bg": "#0f1720",
        "marker_line": "rgba(230, 240, 248, 0.24)",
    }


def _default_plot_reset_data() -> dict[str, int]:
    return {key: 0 for key in EXPLORER_PLOT_KEYS}


def _plot_uirevision(reset_data: dict[str, object] | None, key: str) -> str:
    try:
        token = int(dict(reset_data or {}).get(key, 0) or 0)
    except Exception:
        token = 0
    return f"explorer:{key}:{token}"


def _slugify_token(value: object, *, fallback: str = "selection") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    return text[:80] or fallback


def _journal_export_figure(figure: go.Figure | dict[str, object]) -> go.Figure:
    export_fig = go.Figure(figure)
    export_fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#111111", "family": "Helvetica, Arial, sans-serif", "size": 12},
        title_font={"color": "#111111", "family": "Helvetica, Arial, sans-serif", "size": 16},
        margin={"t": 82, "l": 84, "r": 30, "b": 72},
        legend={
            "bgcolor": "rgba(255,255,255,0.96)",
            "bordercolor": "rgba(40,40,40,0.20)",
            "borderwidth": 1,
            "font": {"color": "#111111", "family": "Helvetica, Arial, sans-serif", "size": 10},
        },
        autosize=False,
        width=1400,
        height=900,
    )
    export_fig.update_xaxes(
        color="#111111",
        title_font_color="#111111",
        tickfont_color="#111111",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.12)",
        zeroline=False,
    )
    export_fig.update_yaxes(
        color="#111111",
        title_font_color="#111111",
        tickfont_color="#111111",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.12)",
        zeroline=False,
    )
    return export_fig


def _axis_window_from_relayout(relayout_data: dict[str, object] | None, axis: str, *, log_axis: bool) -> tuple[float, float] | None:
    if not isinstance(relayout_data, dict):
        return None
    if relayout_data.get(f"{axis}.autorange"):
        return None

    lo = relayout_data.get(f"{axis}.range[0]")
    hi = relayout_data.get(f"{axis}.range[1]")
    if lo is None or hi is None:
        window = relayout_data.get(f"{axis}.range")
        if isinstance(window, (list, tuple)) and len(window) == 2:
            lo, hi = window[0], window[1]
    try:
        lo_f = float(lo)
        hi_f = float(hi)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(lo_f) or not np.isfinite(hi_f):
        return None
    if log_axis:
        lo_f = float(pow(10.0, lo_f))
        hi_f = float(pow(10.0, hi_f))
    if hi_f < lo_f:
        lo_f, hi_f = hi_f, lo_f
    return (lo_f, hi_f)


def _candidate_scope_from_plot(
    relayout_data: dict[str, object] | None,
    *,
    x_metric: object,
    y_metric: object,
    log_flags: list[str] | None,
) -> dict[str, object]:
    x_name = str(x_metric or "").strip()
    y_name = str(y_metric or "").strip()
    log_values = set(str(v) for v in (log_flags or []))
    scope: dict[str, object] = {
        "mode": "filtered",
        "x_metric": x_name,
        "y_metric": y_name,
        "log_x": "logx" in log_values,
        "log_y": "logy" in log_values,
        "x_range": None,
        "y_range": None,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    if not x_name or not y_name:
        return scope

    x_window = _axis_window_from_relayout(relayout_data, "xaxis", log_axis=bool(scope["log_x"]))
    y_window = _axis_window_from_relayout(relayout_data, "yaxis", log_axis=bool(scope["log_y"]))
    if x_window is None and y_window is None:
        return scope
    scope["mode"] = "view"
    scope["x_range"] = list(x_window) if x_window is not None else None
    scope["y_range"] = list(y_window) if y_window is not None else None
    return scope


def _apply_candidate_scope(
    frame: pd.DataFrame,
    *,
    scope_state: dict[str, object] | None,
    x_metric: object,
    y_metric: object,
    log_flags: list[str] | None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    state = dict(scope_state or {})
    if state.get("mode") != "view":
        return frame

    x_name = str(x_metric or "").strip()
    y_name = str(y_metric or "").strip()
    if state.get("x_metric") != x_name or state.get("y_metric") != y_name:
        return frame

    log_values = set(str(v) for v in (log_flags or []))
    if bool(state.get("log_x")) != ("logx" in log_values):
        return frame
    if bool(state.get("log_y")) != ("logy" in log_values):
        return frame

    if x_name not in frame.columns or y_name not in frame.columns:
        return frame

    mask = frame[x_name].notna() & frame[y_name].notna()
    x_window = state.get("x_range")
    if isinstance(x_window, (list, tuple)) and len(x_window) == 2:
        x_series = numeric_series(frame, x_name)
        mask &= x_series >= float(x_window[0])
        mask &= x_series <= float(x_window[1])
    y_window = state.get("y_range")
    if isinstance(y_window, (list, tuple)) and len(y_window) == 2:
        y_series = numeric_series(frame, y_name)
        mask &= y_series >= float(y_window[0])
        mask &= y_series <= float(y_window[1])
    return frame.loc[mask].copy()


def _candidate_scope_status(
    filtered_count: int,
    working_count: int,
    scope_state: dict[str, object] | None,
) -> str:
    state = dict(scope_state or {})
    if state.get("mode") == "view" and working_count < filtered_count:
        return f"Candidates list follows captured plot view: {working_count:,} / {filtered_count:,} filtered"
    return f"Candidates list uses all filtered rows: {working_count:,}"


def _prepare_export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    export_df = frame.copy()
    if export_df.empty:
        return export_df
    if "lc_path" not in export_df.columns:
        export_df["lc_path"] = None
    for idx, row in export_df.iterrows():
        current_lc = str(row.get("lc_path") or "").strip()
        if current_lc:
            continue
        plot_dir_value = row.get("plot_dir")
        plot_dir = None
        if plot_dir_value not in (None, ""):
            try:
                plot_dir = Path(str(plot_dir_value)).expanduser().resolve()
            except Exception:
                plot_dir = None
        payload = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        resolved = review_resolve_lightcurve_path(payload, plot_dir)
        if resolved is not None:
            export_df.at[idx, "lc_path"] = str(resolved)
    return export_df


def _selection_meta(
    *,
    source_filter: list[str] | None,
    query_value: object,
    advanced_state: dict[str, object],
    x_metric: object,
    y_metric: object,
    color_metric: object,
    symbol_metric: object,
    log_flags: list[str] | None,
    x_min: object,
    x_max: object,
    y_min: object,
    y_max: object,
    table_sort: object,
    selected_key: object,
    scope_state: dict[str, object] | None,
    filtered_count: int,
    working_count: int,
    working_frame: pd.DataFrame,
) -> dict[str, object]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": int(working_count),
        "filtered_count": int(filtered_count),
        "source_labels": sorted(str(v) for v in (source_filter or [])),
        "source_files": sorted(str(v) for v in working_frame["source_file"].dropna().astype(str).unique().tolist()) if "source_file" in working_frame.columns else [],
        "source_paths": sorted(str(v) for v in working_frame["source_path"].dropna().astype(str).unique().tolist()) if "source_path" in working_frame.columns else [],
        "plot": {
            "x_metric": str(x_metric or ""),
            "y_metric": str(y_metric or ""),
            "color_metric": str(color_metric or ""),
            "symbol_metric": str(symbol_metric or ""),
            "log_flags": [str(v) for v in (log_flags or [])],
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "candidate_scope": dict(scope_state or {}),
        },
        "filters": {
            "query": str(query_value or ""),
            "advanced": advanced_state,
            "table_sort": str(table_sort or ""),
        },
        "selected_candidate_key": str(selected_key or ""),
    }


def _resolve_initial_candidate_key(
    combined: CombinedCandidateData,
    *,
    candidate_key: str | None = None,
    candidate: str | None = None,
) -> str:
    """Resolve the initial explorer selection from CLI inputs."""
    key_text = str(candidate_key or "").strip()
    if key_text and key_text in combined.key_lookup:
        return key_text
    candidate_text = str(candidate or "").strip()
    resolved = find_candidate_key(combined, candidate_text) if candidate_text else None
    if resolved:
        return resolved
    return str(getattr(combined, "default_candidate_key", "") or "")


def _record_review_target(record: dict[str, object]) -> tuple[Path | None, Path | None]:
    """Return an existing review DB target for a selected record when possible."""
    source_file = str(record.get("source_file") or "").strip()
    if not source_file:
        return None, None
    source_path = Path(source_file).expanduser().resolve()
    if source_path.suffix.lower() != ".db" or not source_path.exists():
        return None, None
    return source_path, infer_plot_dir_from_source(source_path)


def _graph_layout(height: int | None, *, theme: str = DEFAULT_THEME, uirevision: str | None = None) -> dict[str, object]:
    colors = _theme_palette(theme)
    layout: dict[str, object] = {
        "template": colors["template"],
        "autosize": True,
        "paper_bgcolor": colors["paper_bg"],
        "plot_bgcolor": colors["plot_bg"],
        "font": {"color": colors["text"], "family": UI_FONT_FAMILY, "size": 12},
        "title": {"x": 0.01, "xanchor": "left", "y": 0.98, "yanchor": "top", "pad": {"t": 2, "b": 8}},
        "title_automargin": True,
        "title_font": {"color": colors["text"], "size": 14, "family": UI_FONT_FAMILY},
        "margin": {"l": 46, "r": 18, "t": 92, "b": 38},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.14,
            "xanchor": "left",
            "x": 0,
            "bgcolor": colors["legend_bg"],
            "bordercolor": colors["legend_border"],
            "borderwidth": 1,
            "font": {"size": 11, "color": colors["text"], "family": UI_FONT_FAMILY},
        },
        "hoverlabel": {"bgcolor": colors["hover_bg"], "font": {"color": colors["text"], "family": UI_FONT_FAMILY}},
        "uirevision": uirevision or "explorer:static",
    }
    if height is not None:
        layout["height"] = int(height)
    return layout


def _style_plot(fig: go.Figure, *, height: int | None, theme: str = DEFAULT_THEME, uirevision: str | None = None) -> go.Figure:
    colors = _theme_palette(theme)
    fig.update_layout(**_graph_layout(height, theme=theme, uirevision=uirevision))
    fig.update_xaxes(showgrid=True, gridcolor=colors["grid"], zeroline=False, color=colors["text"])
    fig.update_yaxes(showgrid=True, gridcolor=colors["grid"], zeroline=False, color=colors["text"])
    return fig


def _style_native_explorer_figure(fig: go.Figure) -> go.Figure:
    margin = fig.layout.margin.to_plotly_json() if fig.layout.margin else {}
    fig.update_layout(
        title={"x": 0.01, "xanchor": "left", "y": 0.98, "yanchor": "top", "pad": {"t": 2, "b": 8}},
        title_automargin=True,
        margin={
            "l": int(margin.get("l", 55) or 55),
            "r": int(margin.get("r", 20) or 20),
            "t": max(int(margin.get("t", 0) or 0), 84),
            "b": int(margin.get("b", 44) or 44),
        },
    )
    return fig


def _status_figure(message: str, *, height: int | None = 320, theme: str = DEFAULT_THEME, uirevision: str | None = None) -> go.Figure:
    colors = _theme_palette(theme)
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": colors["text"], "family": UI_FONT_FAMILY},
    )
    _style_plot(fig, height=height, theme=theme, uirevision=uirevision)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _distinct_values(frame: pd.DataFrame, col: str, *, max_values: int = 250) -> list[str]:
    if col not in frame.columns:
        return []
    values = frame[col].dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return []
    unique = sorted(values.unique().tolist())
    return unique[:max_values]


def _label(text: str) -> html.Div:
    return html.Div(text, className="explorer-field-label")


def _section_title(text: str) -> html.Div:
    return html.Div(text, className="explorer-section-title")


def _reset_button(button_id: str) -> html.Button:
    return html.Button("Reset view", id=button_id, n_clicks=0, className="explorer-reset-btn")


def _graph_card(graph_id: str, reset_button_id: str, *, height: str, class_name: str = "") -> html.Div:
    card_class = f"explorer-graph-card {class_name}".strip()
    return html.Div(
        [
            html.Div([_reset_button(reset_button_id)], className="explorer-graph-toolbar"),
            dcc.Graph(
                id=graph_id,
                mathjax=True,
                config={"displaylogo": False, "scrollZoom": True, "doubleClick": False},
                style={"height": height},
                className="explorer-graph",
            ),
        ],
        className=card_class,
    )


def _bool_filter_control(col: str) -> html.Div:
    return html.Div(
        [
            _label(col),
            dcc.Dropdown(
                id={"type": "adv-bool-mode", "col": col},
                options=[{"label": "Any", "value": "Any"}, {"label": "True", "value": "True"}, {"label": "False", "value": "False"}],
                value="Any",
                clearable=False,
                persistence=True,
                persistence_type="local",
            ),
        ],
        style={"marginBottom": "8px"},
    )


def _num_filter_control(col: str) -> html.Div:
    return html.Div(
        [
            _label(col),
            html.Div(
                [
                    dcc.Input(id={"type": "adv-num-min", "col": col}, type="number", debounce=True, placeholder="min", style={**BASE_INPUT_STYLE, "padding": "6px 8px"}, persistence=True, persistence_type="local"),
                    dcc.Input(id={"type": "adv-num-max", "col": col}, type="number", debounce=True, placeholder="max", style={**BASE_INPUT_STYLE, "padding": "6px 8px"}, persistence=True, persistence_type="local"),
                ],
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "8px"},
            ),
        ],
        style={"marginBottom": "8px"},
    )


def _text_filter_control(frame: pd.DataFrame, col: str) -> html.Div:
    values = _distinct_values(frame, col)
    options = [{"label": "Any", "value": "Any"}] + [{"label": v, "value": v} for v in values]
    return html.Div(
        [
            _label(col),
            dcc.Dropdown(id={"type": "adv-text-value", "col": col}, options=options, value="Any", clearable=False, persistence=True, persistence_type="local"),
        ],
        style={"marginBottom": "8px"},
    )


def _select_filter_control(frame: pd.DataFrame, col: str) -> html.Div:
    values = _distinct_values(frame, col)
    options = [{"label": v, "value": v} for v in values]
    return html.Div(
        [
            _label(f"{col} (exclude)"),
            dcc.Dropdown(id={"type": "adv-select-exclude", "col": col}, options=options, value=[], multi=True, placeholder="None excluded", persistence=True, persistence_type="local"),
        ],
        style={"marginBottom": "8px"},
    )


def _make_filter_group(frame: pd.DataFrame, name: str, items: list[tuple[str, str]]) -> html.Details:
    children: list[html.Div] = []
    for ftype, col in items:
        if ftype == "bool":
            children.append(_bool_filter_control(col))
        elif ftype == "num":
            children.append(_num_filter_control(col))
        elif ftype == "text":
            children.append(_text_filter_control(frame, col))
        elif ftype == "select":
            children.append(_select_filter_control(frame, col))
    return html.Details(
        [
            html.Summary(name, style={"color": TEXT, "cursor": "pointer", "fontSize": "12px", "fontWeight": "600"}),
            html.Div(children, style={"padding": "8px 2px 2px 2px"}),
        ],
        style={"border": f"1px solid {PANEL_BORDER}", "borderRadius": "8px", "padding": "6px 8px", "background": PANEL_BG, "marginBottom": "8px"},
    )


def _covered_filter_columns() -> set[str]:
    return {str(col) for _group_name, items in SIDEBAR_GROUPS for _kind, col in items}


def _infer_auto_filter_kind(frame: pd.DataFrame, col: str) -> str | None:
    if col in AUTO_FILTER_EXCLUDE_COLUMNS or col.startswith("_"):
        return None
    if col not in frame.columns:
        return None

    series = frame[col]
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_numeric_dtype(series):
        return "num"

    raw = frame[col]
    try:
        values = raw.astype("string").fillna("").str.strip()
    except Exception:
        values = raw.astype(str).replace({"nan": "", "None": ""}).str.strip()
    nonempty = values[values != ""]
    if nonempty.empty:
        return None

    unique_values = {str(v).strip().lower() for v in nonempty.drop_duplicates().tolist()}
    if unique_values and unique_values.issubset({"0", "1", "true", "false", "t", "f", "yes", "no", "y", "n"}):
        return "bool"

    if len(unique_values) <= AUTO_FILTER_TEXT_ONLY_MAX_UNIQUES:
        return "select"
    return "text"


def _build_auto_filter_groups(frame: pd.DataFrame) -> list[tuple[str, list[tuple[str, str]]]]:
    covered = _covered_filter_columns()
    groups: dict[str, list[tuple[str, str]]] = {
        "Additional Flags": [],
        "Additional Numeric": [],
        "Additional Categorical": [],
        "Additional Text": [],
    }

    for col in sorted(str(c) for c in frame.columns):
        if col in covered:
            continue
        kind = _infer_auto_filter_kind(frame, col)
        if kind == "bool":
            groups["Additional Flags"].append(("bool", col))
        elif kind == "num":
            groups["Additional Numeric"].append(("num", col))
        elif kind == "select":
            groups["Additional Categorical"].append(("select", col))
        elif kind == "text":
            groups["Additional Text"].append(("text", col))

    return [(name, items) for name, items in groups.items() if items]


def _build_advanced_filter_sections(frame: pd.DataFrame) -> list[html.Details]:
    groups = list(SIDEBAR_GROUPS) + _build_auto_filter_groups(frame)
    return [_make_filter_group(frame, name, items) for name, items in groups]


def _build_advanced_filter_state(
    bool_values,
    num_min_values,
    num_max_values,
    text_values,
    select_values,
    only_unreviewed_value,
    require_failed_value,
    bool_ids,
    num_min_ids,
    num_max_ids,
    text_ids,
    select_ids,
) -> dict[str, object]:
    state: dict[str, object] = {
        "bool": {},
        "num": {},
        "text": {},
        "select": {},
        "only_unreviewed": bool(only_unreviewed_value and "yes" in only_unreviewed_value),
        "require_failed_any_false": bool(require_failed_value and "yes" in require_failed_value),
    }

    bool_map: dict[str, str] = {}
    for meta, value in zip(bool_ids or [], bool_values or []):
        col = str((meta or {}).get("col") or "")
        if col and value and value != "Any":
            bool_map[col] = str(value)
    state["bool"] = bool_map

    num_map: dict[str, dict[str, float]] = {}
    for meta, value in zip(num_min_ids or [], num_min_values or []):
        col = str((meta or {}).get("col") or "")
        if not col or value in (None, ""):
            continue
        try:
            num_map.setdefault(col, {})["min"] = float(value)
        except Exception:
            continue
    for meta, value in zip(num_max_ids or [], num_max_values or []):
        col = str((meta or {}).get("col") or "")
        if not col or value in (None, ""):
            continue
        try:
            num_map.setdefault(col, {})["max"] = float(value)
        except Exception:
            continue
    state["num"] = num_map

    text_map: dict[str, str] = {}
    for meta, value in zip(text_ids or [], text_values or []):
        col = str((meta or {}).get("col") or "")
        if col and value and value != "Any":
            text_map[col] = str(value)
    state["text"] = text_map

    select_map: dict[str, list[str]] = {}
    for meta, value in zip(select_ids or [], select_values or []):
        col = str((meta or {}).get("col") or "")
        chosen = [str(v) for v in (value or []) if str(v).strip()]
        if col and chosen:
            select_map[col] = chosen
    state["select"] = select_map

    active_count = int(state["only_unreviewed"]) + int(state["require_failed_any_false"])
    active_count += len(bool_map) + len(text_map) + len(select_map)
    active_count += sum(1 for cfg in num_map.values() if cfg)
    state["active_count"] = active_count
    return state


def _apply_advanced_filters(frame: pd.DataFrame, state: dict[str, object]) -> pd.DataFrame:
    out = frame

    if bool(state.get("only_unreviewed")):
        status = text_series(out, "status").str.strip().str.lower()
        if "status" in out.columns:
            out = out[(status == "") | (status == "unreviewed")].copy()

    if bool(state.get("require_failed_any_false")) and "failed_any" in out.columns:
        out = out[~bool_series(out, "failed_any")].copy()

    for col, mode in dict(state.get("bool") or {}).items():
        if col not in out.columns:
            continue
        mask = bool_series(out, col)
        if str(mode) == "True":
            out = out[mask].copy()
        elif str(mode) == "False":
            out = out[~mask].copy()

    for col, cfg in dict(state.get("num") or {}).items():
        if col not in out.columns:
            continue
        series = numeric_series(out, col)
        min_value = dict(cfg).get("min")
        max_value = dict(cfg).get("max")
        if min_value is not None:
            out = out[series >= float(min_value)].copy()
            series = numeric_series(out, col)
        if max_value is not None:
            out = out[series <= float(max_value)].copy()

    for col, value in dict(state.get("text") or {}).items():
        if col not in out.columns:
            continue
        out = out[text_series(out, col) == str(value)].copy()

    for col, values in dict(state.get("select") or {}).items():
        if col not in out.columns:
            continue
        out = out[~text_series(out, col).isin([str(v) for v in values])].copy()

    return out


def _filter_frame(
    frame: pd.DataFrame,
    selected_sources: list[str] | None,
    query: str | None,
    advanced_state: dict[str, object],
) -> tuple[pd.DataFrame, str, int]:
    out = frame
    if selected_sources:
        out = out[out["source_label"].isin(selected_sources)].copy()
    else:
        out = out.iloc[0:0].copy()

    out = _apply_advanced_filters(out, advanced_state)
    query_error = ""
    if query:
        try:
            out = out.query(query, engine="python").copy()
        except Exception as exc:
            query_error = f"Query error: {exc}"
    return out, query_error, int(advanced_state.get("active_count") or 0)


def _series_for_plot(frame: pd.DataFrame, col: str) -> pd.Series:
    s = frame[col]
    if pd.api.types.is_bool_dtype(s):
        return s.map({True: "True", False: "False"}).fillna("False")
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s.astype(str)
    return s.astype(str) if s.dtype == object else s


def _sample_frame(frame: pd.DataFrame, max_points: int, selected_key: str | None = None) -> pd.DataFrame:
    if max_points <= 0 or len(frame) <= max_points:
        return frame.copy()
    sampled = frame.sample(max_points, random_state=42).copy()
    if selected_key and "candidate_key" in frame.columns and selected_key not in set(sampled["candidate_key"]):
        selected = frame[frame["candidate_key"] == selected_key].head(1)
        if not selected.empty:
            sampled = pd.concat([sampled, selected], ignore_index=True).drop_duplicates(subset=["candidate_key"], keep="last")
    return sampled


def _metric_available(frame: pd.DataFrame, col: str) -> bool:
    return bool(col in frame.columns and frame[col].notna().any())


def _resolve_metric(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if _metric_available(frame, col):
            return col
    return None


def _scatter_figure(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str | None,
    symbol: str | None,
    title: str,
    selected_key: str | None,
    max_points: int,
    log_x: bool = False,
    log_y: bool = False,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    height: int | None = 360,
    theme: str = DEFAULT_THEME,
    uirevision: str | None = None,
) -> go.Figure:
    colors = _theme_palette(theme)
    if x not in frame.columns or y not in frame.columns:
        return _status_figure(
            f"Missing columns: {', '.join(col for col in (x, y) if col not in frame.columns)}",
            height=height,
            theme=theme,
            uirevision=uirevision,
        )

    data = frame.loc[frame[x].notna() & frame[y].notna()].copy()
    if data.empty:
        return _status_figure(f"No rows with both `{x}` and `{y}`", height=height, theme=theme, uirevision=uirevision)
    data = _sample_frame(data, max_points=max_points, selected_key=selected_key)

    plot_data = data.copy()
    if color and color in plot_data.columns:
        plot_data[color] = _series_for_plot(plot_data, color)
    if symbol and symbol in plot_data.columns:
        plot_data[symbol] = _series_for_plot(plot_data, symbol)

    hover_cols = [col for col in ["candidate_id", "asas_sn_id", "gaia_id", "source_label", "final_class", "dipper_score"] if col in plot_data.columns]
    fig = px.scatter(
        plot_data,
        x=x,
        y=y,
        color=color if color and color in plot_data.columns else None,
        symbol=symbol if symbol and symbol in plot_data.columns else None,
        hover_data=hover_cols,
        custom_data=["candidate_key"],
        opacity=0.72,
        title=title,
    )
    _style_plot(fig, height=height, theme=theme, uirevision=uirevision)
    fig.update_traces(marker={"size": 8, "line": {"width": 0.7, "color": colors["marker_line"]}})
    fig.update_xaxes(type="log" if log_x else "linear")
    fig.update_yaxes(type="log" if log_y else "linear")

    if selected_key:
        selected = data[data["candidate_key"] == selected_key]
        if not selected.empty:
            fig.add_trace(
                go.Scatter(
                    x=selected[x],
                    y=selected[y],
                    mode="markers",
                    marker={"size": 16, "symbol": "diamond-open", "color": colors["text"], "line": {"width": 2.2, "color": colors["accent"]}},
                    name="selected",
                    customdata=np.column_stack([selected["candidate_key"]]),
                    hovertemplate="selected<extra></extra>",
                    showlegend=True,
                )
            )

    if any(v is not None for v in (x_min, x_max, y_min, y_max)):
        x0 = x_min if x_min is not None else float(data[x].min())
        x1 = x_max if x_max is not None else float(data[x].max())
        y0 = y_min if y_min is not None else float(data[y].min())
        y1 = y_max if y_max is not None else float(data[y].max())
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, line={"color": colors["accent_2"], "width": 2}, fillcolor="rgba(0,0,0,0)")
    return fig


def _metric_pair_figure(
    frame: pd.DataFrame,
    *,
    x_candidates: list[str],
    y_candidates: list[str],
    title: str,
    color: str | None,
    symbol: str | None,
    selected_key: str | None,
    max_points: int,
    log_x: bool = False,
    log_y: bool = False,
    height: int | None = 340,
    theme: str = DEFAULT_THEME,
    uirevision: str | None = None,
) -> go.Figure:
    x = _resolve_metric(frame, x_candidates)
    y = _resolve_metric(frame, y_candidates)
    if x is None or y is None:
        missing = "/".join(x_candidates if x is None else y_candidates)
        return _status_figure(f"No data for {title} ({missing})", height=height, theme=theme, uirevision=uirevision)
    return _scatter_figure(
        frame,
        x=x,
        y=y,
        color=color,
        symbol=symbol,
        title=title,
        selected_key=selected_key,
        max_points=max_points,
        log_x=log_x,
        log_y=log_y,
        height=height,
        theme=theme,
        uirevision=uirevision,
    )


def _hist_figure(
    frame: pd.DataFrame,
    metric: str,
    color: str | None,
    *,
    height: int | None = 340,
    theme: str = DEFAULT_THEME,
    uirevision: str | None = None,
) -> go.Figure:
    if metric not in frame.columns:
        return _status_figure(f"Missing column: {metric}", height=height, theme=theme, uirevision=uirevision)
    data = frame.loc[frame[metric].notna()].copy()
    if data.empty:
        return _status_figure(f"No rows with `{metric}`", height=height, theme=theme, uirevision=uirevision)
    if color and color in data.columns:
        data[color] = _series_for_plot(data, color)
    fig = px.histogram(
        data,
        x=metric,
        color=color if color and color in data.columns else None,
        histnorm="probability density",
        barmode="overlay",
        nbins=40,
        opacity=0.7,
        title=f"Distribution of {metric}",
    )
    _style_plot(fig, height=height, theme=theme, uirevision=uirevision)
    return fig


def _default_native_plot(frame: pd.DataFrame) -> tuple[str, str, str, bool, bool]:
    if {"phase_quality_score", "periodicity_score"}.issubset(frame.columns):
        non_null = frame[["phase_quality_score", "periodicity_score"]].notna().all(axis=1).sum()
        if int(non_null) > 0:
            return "phase_quality_score", "periodicity_score", "Native periodicity: phase quality vs periodicity score", False, False
    return "period_consensus_days", "dipper_score", "Catalog period vs dipper score", True, False


def _table_rows(frame: pd.DataFrame, sort_by: str) -> list[dict[str, object]]:
    cols = [
        "candidate_key",
        "candidate_id",
        "asas_sn_id",
        "gaia_id",
        "source_label",
        "final_class",
        "status",
        "event_class",
        "interest_score",
        "dipper_score",
        "period_n_sources",
        "period_consensus_days",
        "dip_run_count",
        "known_periodic_catalog",
        "oneoff_like",
    ]
    cols = [col for col in cols if col in frame.columns]
    data = frame.copy()
    if sort_by in data.columns:
        data = data.sort_values(sort_by, ascending=False, na_position="last")
    else:
        data = data.sort_values("dipper_score", ascending=False, na_position="last")
    return data.loc[:, cols].to_dict("records")


def _selection_status(selected_record: dict | None, filtered_count: int, query_error: str, active_filters: int) -> str:
    bits = [f"Filtered rows: {filtered_count:,}", f"Active filters: {active_filters}"]
    if selected_record is not None:
        summary = _summary_items(selected_record)
        if summary:
            bits.append(f"Selected: {summary[0][1]}")
    if query_error:
        bits.append(query_error)
    return " | ".join(bits)


def build_explorer_app(
    combined: CombinedCandidateData,
    *,
    host: str,
    port: int,
    initial_candidate_key: str | None = None,
) -> dash.Dash:
    all_source_labels = sorted({src.source_label for src in combined.sources})
    metric_options = available_metric_columns(combined.df) if not combined.df.empty else BEST_FIELDS
    default_hist_metric = "dipper_score" if "dipper_score" in metric_options else (metric_options[0] if metric_options else DEFAULT_MAIN_X)
    advanced_sections = _build_advanced_filter_sections(combined.df)

    app = dash.Dash(
        __name__,
        title="MALCA Explorer",
        assets_folder=str(Path(__file__).resolve().parent / "assets"),
    )
    app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>MALCA Explorer</title>
    {%css%}
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
</head>
<body data-explorer-theme="black">
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
"""
    app.layout = html.Div(
        [
            dcc.Store(id="selected-key-store", data=(initial_candidate_key or combined.default_candidate_key), storage_type="local"),
            dcc.Store(id="period-search-store", data=None),
            dcc.Store(id="period-cache-store", data={}, storage_type="session"),
            dcc.Store(id="theme-mode-store", data=DEFAULT_THEME),
            dcc.Store(id="plot-reset-store", data=_default_plot_reset_data()),
            dcc.Store(id="candidate-scope-store", data={"mode": "filtered"}),
            dcc.Store(id="saved-explorer-gui-state", data=None),
            dcc.Store(id="explorer-review-overrides", data={}, storage_type="session"),
            dcc.Store(id="explorer-resize-init", data=0),
            dcc.Store(id="explorer-sidebar-open", data=True, storage_type="local"),
            dcc.Interval(id="explorer-init", interval=200, n_intervals=0, max_intervals=1),
            dcc.Download(id="explorer-plot-download"),
            dcc.Download(id="explorer-native-plot-download"),
            html.Div(
                [
                    html.Button("Hide Sidebar", id="explorer-sidebar-toggle", n_clicks=0, className="explorer-action-btn explorer-sidebar-toggle"),
                    html.Div(
                        [
                            html.Div("MALCA Explorer", className="explorer-main-title"),
                            html.Div(f"Sources: {len(combined.sources)} | Rows: {len(combined.df):,}", className="explorer-main-subtitle"),
                        ],
                        className="explorer-main-header",
                    ),
                ],
                className="explorer-topbar",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            _section_title("Filters"),
                            _label("Sources"),
                            dcc.Dropdown(id="source-filter", options=[{"label": label, "value": label} for label in all_source_labels], value=all_source_labels, multi=True, persistence=True, persistence_type="local"),
                            _label("Query"),
                            dcc.Input(id="query-input", debounce=True, placeholder="pandas query, e.g. dipper_score >= 5 and period_n_sources <= 1", style=BASE_INPUT_STYLE, persistence=True, persistence_type="local"),
                            dcc.Checklist(id="only-unreviewed", options=[{"label": " Only unreviewed", "value": "yes"}], value=[], className="explorer-stack-checklist", persistence=True, persistence_type="local"),
                            dcc.Checklist(id="require-failed-any-false", options=[{"label": " Require failed_any=False", "value": "yes"}], value=[], className="explorer-stack-checklist", persistence=True, persistence_type="local"),
                            html.Div(
                                [
                                    html.Button("Save GUI State", id="save-explorer-gui-state-btn", n_clicks=0, className="explorer-action-btn"),
                                    html.Div(id="save-explorer-gui-state-status", className="explorer-status-line"),
                                ],
                                style={"marginBottom": "8px"},
                            ),
                            html.Div(advanced_sections, className="explorer-advanced-sections"),
                            html.Hr(className="explorer-rule"),
                            _section_title("Plot Maker"),
                            _label("X metric"),
                            dcc.Dropdown(id="x-metric", options=[{"label": col, "value": col} for col in metric_options], value=None, clearable=True, placeholder="Choose x metric", className="explorer-plotmaker-dropdown"),
                            _label("Y metric"),
                            dcc.Dropdown(id="y-metric", options=[{"label": col, "value": col} for col in metric_options], value=None, clearable=True, placeholder="Choose y metric", className="explorer-plotmaker-dropdown"),
                            _label("Color metric"),
                            dcc.Dropdown(id="color-metric", options=[{"label": col, "value": col} for col in metric_options], value=None, clearable=True, placeholder="Optional", className="explorer-plotmaker-dropdown"),
                            _label("Symbol metric"),
                            dcc.Dropdown(id="symbol-metric", options=[{"label": col, "value": col} for col in metric_options], value=None, clearable=True, placeholder="Optional", className="explorer-plotmaker-dropdown"),
                            html.Div(
                                [
                                    html.Div([_label("X min"), dcc.Input(id="x-min", type="number", placeholder="x min", style=BASE_INPUT_STYLE)], className="explorer-two-up-item"),
                                    html.Div([_label("X max"), dcc.Input(id="x-max", type="number", placeholder="x max", style=BASE_INPUT_STYLE)], className="explorer-two-up-item"),
                                ],
                                className="explorer-two-up",
                            ),
                            html.Div(
                                [
                                    html.Div([_label("Y min"), dcc.Input(id="y-min", type="number", placeholder="y min", style=BASE_INPUT_STYLE)], className="explorer-two-up-item"),
                                    html.Div([_label("Y max"), dcc.Input(id="y-max", type="number", placeholder="y max", style=BASE_INPUT_STYLE)], className="explorer-two-up-item"),
                                ],
                                className="explorer-two-up",
                            ),
                            _label("Axis scaling"),
                            dcc.Checklist(id="log-flags", options=[{"label": " log x", "value": "logx"}, {"label": " log y", "value": "logy"}], value=[], className="explorer-inline-checklist"),
                            _label("Table sort"),
                            dcc.Dropdown(id="table-sort", options=[{"label": col, "value": col} for col in metric_options], value="dipper_score" if "dipper_score" in metric_options else (metric_options[0] if metric_options else None), clearable=False),
                            html.Hr(className="explorer-rule"),
                            _section_title("Open Existing"),
                            _label("Find candidate"),
                            dcc.Input(id="candidate-search", debounce=True, placeholder="candidate_id / asas_sn_id / gaia_id / LC stem", style=BASE_INPUT_STYLE),
                            html.Div(id="explorer-status", className="explorer-status-line"),
                            html.Hr(className="explorer-rule"),
                            _section_title("Native Cameras"),
                            html.Div(
                                [
                                    html.Button("All", id="cams-all-btn", n_clicks=0, className="explorer-action-btn"),
                                    html.Button("Clear", id="cams-clear-btn", n_clicks=0, className="explorer-action-btn"),
                                    html.Button("Invert", id="cams-invert-btn", n_clicks=0, className="explorer-action-btn"),
                                ],
                                className="explorer-button-row",
                            ),
                            dcc.Checklist(id="camera-checklist", options=[], value=[], className="explorer-camera-checklist"),
                            _section_title("Native Bands"),
                            dcc.Checklist(id="band-checklist", options=[{"label": " g", "value": "g"}, {"label": " V", "value": "V"}], value=["g", "V"], className="explorer-camera-checklist"),
                            html.Hr(className="explorer-rule"),
                            _section_title("Light Curve Panels"),
                            dcc.Checklist(
                                id="panel-options",
                                options=[
                                    {"label": " Raw", "value": "raw"},
                                    {"label": " Residuals", "value": "resid"},
                                    {"label": " Phase", "value": "phase"},
                                    {"label": " Baseline", "value": "baseline"},
                                    {"label": " Events", "value": "events"},
                                    {"label": " Diagnostics", "value": "diagnostics"},
                                    {"label": " Filter bad cameras", "value": "filter_bad_cameras"},
                                ],
                                value=["raw", "resid", "phase", "baseline", "events", "filter_bad_cameras"],
                                className="explorer-stack-checklist",
                            ),
                            html.Div(
                                [
                                    html.Div([_label("Manual period"), dcc.Input(id="period-input", type="number", debounce=True, placeholder="days", style=BASE_INPUT_STYLE)], className="explorer-two-up-item"),
                                    html.Div([_label("Y-axis"), dcc.RadioItems(id="yaxis-mode", options=[{"label": " mag", "value": "mag"}, {"label": " flux", "value": "flux"}], value="mag", className="explorer-inline-radio")], className="explorer-two-up-item"),
                                ],
                                className="explorer-two-up",
                            ),
                            _section_title("Period Search"),
                            _label("Method"),
                            dcc.Dropdown(id="period-method", options=[{"label": "PDM", "value": "pdm"}, {"label": "CE", "value": "ce"}, {"label": "LSP", "value": "lsp"}], value="pdm", clearable=False),
                            html.Div(
                                [
                                    html.Div([_label("Min period"), dcc.Input(id="period-min", type="number", debounce=True, placeholder="min d", value=0.1, style=BASE_INPUT_STYLE)], className="explorer-two-up-item"),
                                    html.Div([_label("Max period"), dcc.Input(id="period-max", type="number", debounce=True, placeholder="max d", value=100.0, style=BASE_INPUT_STYLE)], className="explorer-two-up-item"),
                                ],
                                className="explorer-two-up",
                            ),
                            html.Div(
                                [
                                    html.Button("Auto", id="period-auto-btn", n_clicks=0, className="explorer-action-btn"),
                                    html.Button("Find Period", id="period-search-btn", n_clicks=0, className="explorer-action-btn explorer-primary-btn"),
                                ],
                                className="explorer-button-row",
                            ),
                            html.Div(id="period-search-label", className="explorer-status-line"),
                            html.Hr(className="explorer-rule"),
                            _section_title("Theme"),
                            dcc.RadioItems(id="theme-mode", options=[{"label": " Dark", "value": "black"}, {"label": " Light", "value": "white"}], value=DEFAULT_THEME, className="explorer-inline-radio"),
                        ],
                        id="explorer-sidebar",
                        className="explorer-sidebar",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div("Custom plot", className="explorer-card-title"),
                                            dcc.Graph(
                                                id="custom-graph",
                                                mathjax=True,
                                                config={"displaylogo": False, "scrollZoom": True, "doubleClick": False, "responsive": True},
                                                responsive=True,
                                                style={"height": "clamp(360px, 56vh, 760px)", "width": "100%", "minWidth": "0"},
                                                className="explorer-graph",
                                            ),
                                            html.Div(
                                                [
                                                    html.Button("Refresh Candidates From View", id="refresh-candidates-view-btn", n_clicks=0, className="explorer-action-btn"),
                                                    html.Button("Export Review Bundle", id="export-review-bundle-btn", n_clicks=0, className="explorer-action-btn"),
                                                    html.Button("Open Selection In Review", id="open-selection-in-review-btn", n_clicks=0, className="explorer-action-btn"),
                                                    html.Button("Export PDF", id="export-custom-pdf-btn", n_clicks=0, className="explorer-action-btn explorer-primary-btn"),
                                                    _reset_button("custom-reset-btn"),
                                                ],
                                                className="explorer-graph-actions",
                                            ),
                                            html.Div(id="plot-summary", className="explorer-status-line"),
                                            html.Div(id="candidate-scope-status", className="explorer-status-line"),
                                            html.Div(id="bundle-status", className="explorer-status-line"),
                                        ],
                                        className="explorer-graph-card",
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Candidates", className="explorer-card-title"),
                                            dash_table.DataTable(
                                                id="candidate-table",
                                                columns=[
                                                    {"name": "candidate_id", "id": "candidate_id"},
                                                    {"name": "source_label", "id": "source_label"},
                                                    {"name": "final_class", "id": "final_class"},
                                                    {"name": "status", "id": "status"},
                                                    {"name": "event_class", "id": "event_class"},
                                                    {"name": "interest_score", "id": "interest_score"},
                                                    {"name": "dipper_score", "id": "dipper_score"},
                                                    {"name": "period_n_sources", "id": "period_n_sources"},
                                                    {"name": "period_consensus_days", "id": "period_consensus_days"},
                                                    {"name": "dip_run_count", "id": "dip_run_count"},
                                                    {"name": "known_periodic_catalog", "id": "known_periodic_catalog"},
                                                    {"name": "oneoff_like", "id": "oneoff_like"},
                                                    {"name": "candidate_key", "id": "candidate_key"},
                                                ],
                                                hidden_columns=["candidate_key"],
                                                data=[],
                                                page_action="native",
                                                page_size=12,
                                                style_table={"overflowX": "auto"},
                                                style_cell={"backgroundColor": PANEL_BG, "color": TEXT, "border": f"1px solid {PANEL_BORDER}", "fontFamily": UI_FONT_FAMILY, "fontSize": "12px", "padding": "7px 8px"},
                                                style_header={"backgroundColor": PANEL_BG_ALT, "color": TEXT, "fontWeight": "600", "border": f"1px solid {PANEL_BORDER}"},
                                                row_selectable="single",
                                            ),
                                        ],
                                        className="explorer-card explorer-table-card",
                                    ),
                                ],
                                id="explorer-left-panel",
                                className="explorer-left-panel",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Div("Selected candidate", className="explorer-card-title"),
                                                    html.Button("Open in Review", id="open-current-in-review-btn", n_clicks=0, className="explorer-action-btn"),
                                                ],
                                                className="explorer-graph-toolbar",
                                            ),
                                            html.Div(id="selected-status", className="explorer-status-line"),
                                            html.Div(id="review-launch-status", className="explorer-status-line"),
                                            html.Div(id="viewer-summary", className="explorer-summary"),
                                            html.Hr(className="explorer-rule"),
                                            html.Div("Explorer grading", className="explorer-card-title"),
                                            html.Div(id="explorer-review-target-status", className="explorer-status-line"),
                                            html.Div(id="explorer-review-save-status", className="explorer-status-line"),
                                            _label("Class"),
                                            dcc.Dropdown(
                                                id="explorer-review-class",
                                                options=REVIEW_CLASS_OPTIONS,
                                                value="unclassified",
                                                clearable=False,
                                            ),
                                            _label("Confidence"),
                                            dcc.RadioItems(
                                                id="explorer-review-confidence",
                                                options=REVIEW_SCORE_OPTIONS,
                                                value=None,
                                                className="explorer-inline-radio",
                                            ),
                                            dcc.Checklist(
                                                id="explorer-review-followup",
                                                options=[{"label": " Needs follow-up", "value": "followup"}],
                                                value=[],
                                                className="explorer-inline-checklist",
                                            ),
                                            _label("Notes"),
                                            dcc.Textarea(
                                                id="explorer-review-notes",
                                                value="",
                                                placeholder="Review notes",
                                                style={"width": "100%", "minHeight": "88px", "resize": "vertical"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Button(
                                                        "Save Grade",
                                                        id="explorer-review-save-btn",
                                                        n_clicks=0,
                                                        className="explorer-action-btn explorer-primary-btn",
                                                    ),
                                                ],
                                                className="explorer-button-row",
                                            ),
                                        ],
                                        className="explorer-card",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Div("Native light curve", className="explorer-card-title"),
                                                    html.Button("Export PDF", id="export-native-pdf-btn", n_clicks=0, className="explorer-action-btn explorer-primary-btn"),
                                                    _reset_button("lightcurve-reset-btn"),
                                                ],
                                                className="explorer-graph-toolbar",
                                            ),
                                            dcc.Graph(
                                                id="lightcurve-graph",
                                                mathjax=True,
                                                config={"displaylogo": False, "scrollZoom": True, "doubleClick": False, "responsive": True},
                                                responsive=True,
                                                style={"height": "clamp(420px, 72vh, 1100px)", "width": "100%", "minWidth": "0"},
                                                className="explorer-graph",
                                            ),
                                        ],
                                        className="explorer-graph-card explorer-lightcurve-card",
                                    ),
                                    html.Div(
                                        [
                                            html.Div("Precomputed stats", className="explorer-card-title"),
                                            html.Div(id="viewer-stats"),
                                        ],
                                        className="explorer-card",
                                    ),
                                ],
                                className="explorer-right-panel",
                            ),
                        ],
                        className="explorer-workspace",
                    ),
                ],
                id="explorer-shell",
                className="explorer-shell",
            ),
        ],
        className="review-explorer-app",
        style={
            "background": APP_BG,
            "height": "100vh",
            "padding": "12px",
            "fontFamily": UI_FONT_FAMILY,
            "overflow": "hidden",
        },
    )

    @app.callback(
        Output("selected-key-store", "data", allow_duplicate=True),
        Input("explorer-init", "n_intervals"),
        prevent_initial_call="initial_duplicate",
    )
    def apply_initial_candidate_selection(_tick):
        if not initial_candidate_key:
            return dash.no_update
        return initial_candidate_key

    app.clientside_callback(
        """
        function(_tick, currentTheme) {
            try {
                var saved = window.localStorage.getItem('malca.explorer.theme');
                if (saved && ['black', 'white'].includes(saved)) {
                    return saved;
                }
            } catch (e) {
                // ignore storage failures
            }
            return ['black', 'white'].includes(currentTheme) ? currentTheme : 'black';
        }
        """,
        Output("theme-mode", "value"),
        Input("explorer-init", "n_intervals"),
        State("theme-mode", "value"),
        prevent_initial_call=False,
    )

    app.clientside_callback(
        """
        function(theme) {
            var resolved = ['black', 'white'].includes(theme) ? theme : 'black';
            try {
                document.body.setAttribute('data-explorer-theme', resolved);
                window.localStorage.setItem('malca.explorer.theme', resolved);
            } catch (e) {
                // ignore storage/document failures
            }
            return resolved;
        }
        """,
        Output("theme-mode-store", "data"),
        Input("theme-mode", "value"),
        prevent_initial_call=False,
    )

    @app.callback(
        Output("saved-explorer-gui-state", "data"),
        Input("explorer-init", "n_intervals"),
        prevent_initial_call=False,
    )
    def load_saved_explorer_gui_state(_tick):
        state_db_path = _explorer_state_db_path(combined)
        if state_db_path is None:
            return None
        try:
            with closing(db_connect(state_db_path)) as conn:
                raw = str(load_app_state(conn, _EXPLORER_GUI_STATE_APP_STATE_KEY, "") or "").strip()
        except Exception:
            return None
        if not raw:
            return None
        try:
            return _normalize_explorer_gui_state(json.loads(raw))
        except Exception:
            return None

    @app.callback(
        Output("save-explorer-gui-state-status", "children"),
        Output("saved-explorer-gui-state", "data", allow_duplicate=True),
        Input("save-explorer-gui-state-btn", "n_clicks"),
        [
            State("source-filter", "value"),
            State("query-input", "value"),
            State("x-metric", "value"),
            State("y-metric", "value"),
            State("color-metric", "value"),
            State("symbol-metric", "value"),
            State("log-flags", "value"),
            State("x-min", "value"),
            State("x-max", "value"),
            State("y-min", "value"),
            State("y-max", "value"),
            State("table-sort", "value"),
            State("selected-key-store", "data"),
            State("theme-mode", "value"),
            State("candidate-scope-store", "data"),
            State("panel-options", "value"),
            State("period-input", "value"),
            State("yaxis-mode", "value"),
            State("period-method", "value"),
            State("period-min", "value"),
            State("period-max", "value"),
            State("camera-checklist", "value"),
            State("band-checklist", "value"),
            *ADV_FILTER_STATES,
        ],
        prevent_initial_call=True,
    )
    def save_explorer_gui_state(
        n_clicks,
        source_filter,
        query_value,
        x_metric,
        y_metric,
        color_metric,
        symbol_metric,
        log_flags,
        x_min,
        x_max,
        y_min,
        y_max,
        table_sort,
        selected_key,
        theme_mode,
        candidate_scope_state,
        panel_options,
        period_value,
        yaxis_mode,
        period_method,
        period_min,
        period_max,
        camera_values,
        band_values,
        bool_values,
        num_min_values,
        num_max_values,
        text_values,
        select_values,
        only_unreviewed_value,
        require_failed_value,
        bool_ids,
        num_min_ids,
        num_max_ids,
        text_ids,
        select_ids,
    ):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        state_db_path = _explorer_state_db_path(combined)
        if state_db_path is None:
            return "GUI state save is available only when explorer is opened on a single review DB.", dash.no_update
        advanced_state = _build_advanced_filter_state(
            bool_values,
            num_min_values,
            num_max_values,
            text_values,
            select_values,
            only_unreviewed_value,
            require_failed_value,
            bool_ids,
            num_min_ids,
            num_max_ids,
            text_ids,
            select_ids,
        )
        gui_state = _explorer_gui_state_from_values(
            source_filter=source_filter,
            query_value=query_value,
            advanced_state=advanced_state,
            x_metric=x_metric,
            y_metric=y_metric,
            color_metric=color_metric,
            symbol_metric=symbol_metric,
            log_flags=log_flags,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            table_sort=table_sort,
            selected_key=selected_key,
            theme_mode=theme_mode,
            candidate_scope_state=dict(candidate_scope_state or {"mode": "filtered"}),
            panel_options=panel_options,
            period_value=period_value,
            yaxis_mode=yaxis_mode,
            period_method=period_method,
            period_min=period_min,
            period_max=period_max,
            selected_cameras=camera_values,
            selected_bands=band_values,
        )
        try:
            with closing(db_connect(state_db_path)) as conn:
                save_app_state(conn, _EXPLORER_GUI_STATE_APP_STATE_KEY, json.dumps(gui_state, default=str))
        except Exception as exc:
            return f"Failed to save GUI state: {exc}", dash.no_update
        return f"Saved explorer GUI state to {state_db_path}.", gui_state

    @app.callback(
        [
            Output("source-filter", "value", allow_duplicate=True),
            Output("query-input", "value", allow_duplicate=True),
            Output("only-unreviewed", "value", allow_duplicate=True),
            Output("require-failed-any-false", "value", allow_duplicate=True),
            Output("x-metric", "value", allow_duplicate=True),
            Output("y-metric", "value", allow_duplicate=True),
            Output("color-metric", "value", allow_duplicate=True),
            Output("symbol-metric", "value", allow_duplicate=True),
            Output("log-flags", "value", allow_duplicate=True),
            Output("x-min", "value", allow_duplicate=True),
            Output("x-max", "value", allow_duplicate=True),
            Output("y-min", "value", allow_duplicate=True),
            Output("y-max", "value", allow_duplicate=True),
            Output("table-sort", "value", allow_duplicate=True),
            Output("selected-key-store", "data", allow_duplicate=True),
            Output("theme-mode", "value", allow_duplicate=True),
            Output("candidate-scope-store", "data", allow_duplicate=True),
            Output("panel-options", "value", allow_duplicate=True),
            Output("period-input", "value", allow_duplicate=True),
            Output("yaxis-mode", "value", allow_duplicate=True),
            Output("period-method", "value", allow_duplicate=True),
            Output("period-min", "value", allow_duplicate=True),
            Output("period-max", "value", allow_duplicate=True),
            Output("camera-checklist", "value", allow_duplicate=True),
            Output("band-checklist", "value", allow_duplicate=True),
            Output({"type": "adv-bool-mode", "col": ALL}, "value", allow_duplicate=True),
            Output({"type": "adv-num-min", "col": ALL}, "value", allow_duplicate=True),
            Output({"type": "adv-num-max", "col": ALL}, "value", allow_duplicate=True),
            Output({"type": "adv-text-value", "col": ALL}, "value", allow_duplicate=True),
            Output({"type": "adv-select-exclude", "col": ALL}, "value", allow_duplicate=True),
        ],
        Input("saved-explorer-gui-state", "data"),
        [
            State({"type": "adv-bool-mode", "col": ALL}, "id"),
            State({"type": "adv-num-min", "col": ALL}, "id"),
            State({"type": "adv-num-max", "col": ALL}, "id"),
            State({"type": "adv-text-value", "col": ALL}, "id"),
            State({"type": "adv-select-exclude", "col": ALL}, "id"),
        ],
        prevent_initial_call="initial_duplicate",
    )
    def restore_saved_explorer_gui_state(saved_state, bool_ids, num_min_ids, num_max_ids, text_ids, select_ids):
        state = _normalize_explorer_gui_state(saved_state)
        if state is None:
            return tuple([dash.no_update] * 30)

        valid_sources = [label for label in _coerce_string_list(state.get("source_filter")) if label in all_source_labels]
        source_filter = valid_sources or list(all_source_labels)
        valid_metrics = set(metric_options)
        x_metric = state.get("x_metric") if state.get("x_metric") in valid_metrics else None
        y_metric = state.get("y_metric") if state.get("y_metric") in valid_metrics else None
        color_metric = state.get("color_metric") if state.get("color_metric") in valid_metrics else None
        symbol_metric = state.get("symbol_metric") if state.get("symbol_metric") in valid_metrics else None
        table_sort = state.get("table_sort") if state.get("table_sort") in valid_metrics else (
            "dipper_score" if "dipper_score" in valid_metrics else (metric_options[0] if metric_options else None)
        )
        bool_values, num_min_values, num_max_values, text_values, select_values, only_unreviewed, require_failed = _explorer_advanced_ui_values_from_state(
            state,
            bool_ids=bool_ids,
            num_min_ids=num_min_ids,
            num_max_ids=num_max_ids,
            text_ids=text_ids,
            select_ids=select_ids,
        )
        return (
            source_filter,
            str(state.get("query_value") or ""),
            only_unreviewed,
            require_failed,
            x_metric,
            y_metric,
            color_metric,
            symbol_metric,
            _coerce_string_list(state.get("log_flags")),
            _coerce_optional_float(state.get("x_min")),
            _coerce_optional_float(state.get("x_max")),
            _coerce_optional_float(state.get("y_min")),
            _coerce_optional_float(state.get("y_max")),
            table_sort,
            str(state.get("selected_key") or ""),
            "white" if state.get("theme_mode") == "white" else "black",
            dict(state.get("candidate_scope_state") or {"mode": "filtered"}),
            _coerce_string_list(state.get("panel_options")),
            _coerce_optional_float(state.get("period_value")),
            "flux" if state.get("yaxis_mode") == "flux" else "mag",
            state.get("period_method") if state.get("period_method") in {"pdm", "ce", "lsp", "auto"} else "pdm",
            _coerce_optional_float(state.get("period_min")),
            _coerce_optional_float(state.get("period_max")),
            _coerce_string_list(state.get("camera_values")),
            _coerce_string_list(state.get("band_values")),
            bool_values,
            num_min_values,
            num_max_values,
            text_values,
            select_values,
        )

    app.clientside_callback(
        """
        function(_tick) {
            var workspace = document.querySelector('.explorer-workspace');
            if (!workspace) {
                return window.dash_clientside.no_update;
            }

            var scheduleResize = function() {
                if (window.__malcaExplorerResizeFrame) {
                    window.cancelAnimationFrame(window.__malcaExplorerResizeFrame);
                }
                window.__malcaExplorerResizeFrame = window.requestAnimationFrame(function() {
                    window.dispatchEvent(new Event('resize'));
                    window.__malcaExplorerResizeFrame = null;
                });
            };

            var resizePlots = function() {
                scheduleResize();
                if (window.Plotly && window.Plotly.Plots && window.Plotly.Plots.resize) {
                    ['custom-graph', 'lightcurve-graph'].forEach(function(graphId) {
                        var container = document.getElementById(graphId);
                        if (!container) {
                            return;
                        }
                        var plotNode = container.querySelector('.js-plotly-plot');
                        if (plotNode) {
                            try { window.Plotly.Plots.resize(plotNode); } catch (err) {}
                        }
                    });
                }
            };

            if (!window.__malcaExplorerResizeObserver) {
                window.addEventListener('resize', resizePlots);
                if (window.ResizeObserver) {
                    var observer = new window.ResizeObserver(function() {
                        resizePlots();
                    });
                    observer.observe(workspace);
                    var leftPanel = document.getElementById('explorer-left-panel');
                    var rightPanel = workspace.querySelector('.explorer-right-panel');
                    if (leftPanel) {
                        observer.observe(leftPanel);
                    }
                    if (rightPanel) {
                        observer.observe(rightPanel);
                    }
                    window.__malcaExplorerResizeObserver = observer;
                } else {
                    window.__malcaExplorerResizeObserver = true;
                }
            }

            resizePlots();
            return window.dash_clientside.no_update;
        }
        """,
        Output("explorer-resize-init", "data"),
        Input("explorer-init", "n_intervals"),
        prevent_initial_call=False,
    )

    @app.callback(
        Output("explorer-sidebar", "className"),
        Output("explorer-shell", "className"),
        Output("explorer-sidebar-open", "data"),
        Output("explorer-sidebar-toggle", "children"),
        Input("explorer-sidebar-toggle", "n_clicks"),
        State("explorer-sidebar-open", "data"),
        prevent_initial_call=False,
    )
    def toggle_sidebar(n_clicks, is_open):
        sidebar_open = bool(is_open) if is_open is not None else True
        if dash.callback_context.triggered_id == "explorer-sidebar-toggle" and n_clicks:
            sidebar_open = not sidebar_open

        sidebar_class = "explorer-sidebar" if sidebar_open else "explorer-sidebar is-hidden"
        shell_class = "explorer-shell" if sidebar_open else "explorer-shell sidebar-collapsed"
        button_label = "Hide Sidebar" if sidebar_open else "Show Sidebar"
        return sidebar_class, shell_class, sidebar_open, button_label

    @app.callback(
        Output("plot-reset-store", "data"),
        [
            Input("custom-reset-btn", "n_clicks"),
            Input("lightcurve-reset-btn", "n_clicks"),
        ],
        State("plot-reset-store", "data"),
        prevent_initial_call=True,
    )
    def update_plot_resets(_custom, _lightcurve, current):
        triggered = dash.callback_context.triggered_id
        key_map = {
            "custom-reset-btn": "custom",
            "lightcurve-reset-btn": "lightcurve",
        }
        key = key_map.get(str(triggered or ""))
        if key is None:
            return dash.no_update
        updated = _default_plot_reset_data()
        updated.update(dict(current or {}))
        updated[key] = int(updated.get(key, 0) or 0) + 1
        return updated

    @app.callback(
        Output("candidate-scope-store", "data"),
        Input("refresh-candidates-view-btn", "n_clicks"),
        State("custom-graph", "relayoutData"),
        State("x-metric", "value"),
        State("y-metric", "value"),
        State("log-flags", "value"),
        prevent_initial_call=True,
    )
    def refresh_candidate_scope(_n_clicks, relayout_data, x_metric, y_metric, log_flags):
        return _candidate_scope_from_plot(
            relayout_data,
            x_metric=x_metric,
            y_metric=y_metric,
            log_flags=log_flags,
        )

    @app.callback(
        Output("explorer-plot-download", "data"),
        Output("bundle-status", "children", allow_duplicate=True),
        Input("export-custom-pdf-btn", "n_clicks"),
        State("custom-graph", "figure"),
        State("x-metric", "value"),
        State("y-metric", "value"),
        prevent_initial_call=True,
    )
    def export_custom_plot_pdf(n_clicks, figure, x_metric, y_metric):
        if not n_clicks:
            return dash.no_update, dash.no_update
        if not figure:
            return dash.no_update, "No custom plot is available to export."
        try:
            image_bytes = pio.to_image(_journal_export_figure(figure), format="pdf", width=1400, height=900)
        except Exception as exc:
            return dash.no_update, f"PDF export failed: {exc}"
        slug_x = _slugify_token(x_metric or "x")
        slug_y = _slugify_token(y_metric or "y")
        fname = f"explorer_{slug_y}_vs_{slug_x}.pdf"
        return dcc.send_bytes(image_bytes, fname), f"Exported {fname}"

    @app.callback(
        Output("explorer-native-plot-download", "data"),
        Output("bundle-status", "children", allow_duplicate=True),
        Input("export-native-pdf-btn", "n_clicks"),
        State("lightcurve-graph", "figure"),
        State("selected-key-store", "data"),
        prevent_initial_call=True,
    )
    def export_native_plot_pdf(n_clicks, figure, selected_key):
        if not n_clicks:
            return dash.no_update, dash.no_update
        if not figure:
            return dash.no_update, "No native light curve plot is available to export."
        try:
            image_bytes = pio.to_image(_journal_export_figure(figure), format="pdf", width=1400, height=900)
        except Exception as exc:
            return dash.no_update, f"Native PDF export failed: {exc}"
        fname = f"explorer_native_lightcurve_{_slugify_token(selected_key or 'candidate')}.pdf"
        return dcc.send_bytes(image_bytes, fname), f"Exported {fname}"

    @app.callback(
        Output("bundle-status", "children"),
        Input("export-review-bundle-btn", "n_clicks"),
        Input("open-selection-in-review-btn", "n_clicks"),
        [
            State("custom-graph", "figure"),
            State("source-filter", "value"),
            State("query-input", "value"),
            State("x-metric", "value"),
            State("y-metric", "value"),
            State("color-metric", "value"),
            State("symbol-metric", "value"),
            State("log-flags", "value"),
            State("x-min", "value"),
            State("x-max", "value"),
            State("y-min", "value"),
            State("y-max", "value"),
            State("table-sort", "value"),
            State("selected-key-store", "data"),
            State("candidate-scope-store", "data"),
            State("explorer-review-overrides", "data"),
            *ADV_FILTER_STATES,
        ],
        prevent_initial_call=True,
    )
    def export_review_bundle(*args):
        (
            n_clicks,
            open_review_clicks,
            figure,
            source_filter,
            query_value,
            x_metric,
            y_metric,
            color_metric,
            symbol_metric,
            log_flags,
            x_min,
            x_max,
            y_min,
            y_max,
            table_sort,
            selected_key,
            candidate_scope_state,
            review_overrides,
            bool_values,
            num_min_values,
            num_max_values,
            text_values,
            select_values,
            only_unreviewed_value,
            require_failed_value,
            bool_ids,
            num_min_ids,
            num_max_ids,
            text_ids,
            select_ids,
        ) = args
        if not n_clicks and not open_review_clicks:
            return dash.no_update
        triggered = dash.callback_context.triggered_id

        source_frame = _apply_review_overrides(combined.df, review_overrides)
        advanced_state = _build_advanced_filter_state(
            bool_values,
            num_min_values,
            num_max_values,
            text_values,
            select_values,
            only_unreviewed_value,
            require_failed_value,
            bool_ids,
            num_min_ids,
            num_max_ids,
            text_ids,
            select_ids,
        )
        filtered, query_error, _active_filters = _filter_frame(source_frame, source_filter, query_value, advanced_state)
        working = _apply_candidate_scope(
            filtered,
            scope_state=candidate_scope_state,
            x_metric=x_metric,
            y_metric=y_metric,
            log_flags=log_flags,
        )
        if query_error:
            return f"Bundle export blocked by invalid query: {query_error}"
        if working.empty:
            return "No candidates are available in the current filtered/view selection."

        export_df = _prepare_export_frame(working)
        slug_x = _slugify_token(x_metric or "x")
        slug_y = _slugify_token(y_metric or "y")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        bundle_dir = Path("output") / "explorer_exports" / f"{ts}_{slug_y}_vs_{slug_x}"
        selection_meta = _selection_meta(
            source_filter=source_filter,
            query_value=query_value,
            advanced_state=advanced_state,
            x_metric=x_metric,
            y_metric=y_metric,
            color_metric=color_metric,
            symbol_metric=symbol_metric,
            log_flags=log_flags,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            table_sort=table_sort,
            selected_key=selected_key,
            scope_state=candidate_scope_state,
            filtered_count=len(filtered),
            working_count=len(export_df),
            working_frame=export_df,
        )
        try:
            result = export_review_subset_bundle(bundle_dir, export_df, selection_meta=selection_meta)
            if figure:
                pdf_path = Path(result["bundle_dir"]) / "selection_plot.pdf"
                pdf_path.write_bytes(pio.to_image(_journal_export_figure(figure), format="pdf", width=1400, height=900))
            review_db = Path(result["review_db"])
            if triggered == "open-selection-in-review-btn":
                selected_record = _get_candidate_record_from_frame(source_frame, selected_key) or get_candidate_record_by_key(combined, selected_key)
                candidate_hint = None
                if selected_record is not None:
                    candidate_hint = str(
                        selected_record.get("candidate_id")
                        or selected_record.get("asas_sn_id")
                        or selected_record.get("gaia_id")
                        or ""
                    ).strip() or None
                command, url = build_review_command(db_path=review_db, candidate=candidate_hint)
                launch_detached(command)
                return f"Wrote review bundle to {bundle_dir} ({len(export_df):,} candidates) and opened review at {url}."
            return f"Wrote review bundle to {bundle_dir} ({len(export_df):,} candidates). Open with `malca review --db {review_db}`."
        except Exception as exc:
            return f"Review bundle export failed: {exc}"

    @app.callback(
        Output("review-launch-status", "children"),
        Input("open-current-in-review-btn", "n_clicks"),
        State("selected-key-store", "data"),
        State("explorer-review-overrides", "data"),
        prevent_initial_call=True,
    )
    def open_current_in_review(n_clicks, selected_key, review_overrides):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate
        source_frame = _apply_review_overrides(combined.df, review_overrides)
        record = _get_candidate_record_from_frame(source_frame, selected_key) or get_candidate_record_by_key(combined, selected_key)
        if record is None:
            return "No candidate selected for review."

        review_db, plot_dir = _record_review_target(record)
        candidate_hint = str(
            record.get("candidate_id")
            or record.get("asas_sn_id")
            or record.get("gaia_id")
            or selected_key
            or ""
        ).strip() or None

        if review_db is None:
            export_df = _prepare_export_frame(pd.DataFrame([record]))
            slug = _slugify_token(candidate_hint or "candidate")
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            bundle_dir = Path("output") / "explorer_exports" / f"{ts}_{slug}"
            selection_meta = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "candidate_count": 1,
                "filtered_count": 1,
                "source_labels": [str(record.get("source_label") or "")],
                "source_files": [str(record.get("source_file") or "")],
                "source_paths": [str(record.get("source_path") or "")],
                "plot": {},
                "filters": {"query": "", "advanced": {}, "table_sort": ""},
                "selected_candidate_key": str(selected_key or ""),
            }
            result = export_review_subset_bundle(bundle_dir, export_df, selection_meta=selection_meta)
            review_db = Path(result["review_db"])
            plot_dir = None

        command, url = build_review_command(db_path=review_db, candidate=candidate_hint, plot_dir=plot_dir)
        launch_detached(command)
        return f"Opened review at {url}."

    @app.callback(
        Output("explorer-review-class", "value"),
        Output("explorer-review-confidence", "value"),
        Output("explorer-review-followup", "value"),
        Output("explorer-review-notes", "value"),
        Output("explorer-review-target-status", "children"),
        Output("explorer-review-save-btn", "disabled"),
        Input("selected-key-store", "data"),
        Input("explorer-review-overrides", "data"),
        prevent_initial_call=False,
    )
    def load_explorer_review_form(selected_key, review_overrides):
        source_frame = _apply_review_overrides(combined.df, review_overrides)
        record = _get_candidate_record_from_frame(source_frame, selected_key) or get_candidate_record_by_key(combined, selected_key)
        if record is None:
            return "unclassified", None, [], "", "No candidate selected.", True

        review_state = _record_review_state(record)
        review_db, _plot_dir = _record_review_target(record)
        candidate_id = str(record.get("candidate_id") or "").strip()
        if review_db is not None and candidate_id:
            try:
                with closing(db_connect(review_db)) as conn:
                    review_state = _normalize_review_state(get_review(conn, candidate_id))
            except Exception as exc:
                return (
                    review_state.get("event_class", "unclassified"),
                    review_state.get("interest_score"),
                    ["followup"] if review_state.get("status") == "needs_followup" else [],
                    str(review_state.get("notes") or ""),
                    f"Inline grading unavailable: {exc}",
                    True,
                )
            target_status = f"Inline grading saves to {review_db}"
            save_disabled = False
        else:
            target_status = "Inline grading is available only for DB-backed candidates."
            save_disabled = True

        return (
            str(review_state.get("event_class") or "unclassified"),
            review_state.get("interest_score"),
            ["followup"] if review_state.get("status") == "needs_followup" else [],
            str(review_state.get("notes") or ""),
            target_status,
            save_disabled,
        )

    @app.callback(
        Output("explorer-review-save-status", "children"),
        Input("selected-key-store", "data"),
        prevent_initial_call=False,
    )
    def clear_explorer_review_save_status(_selected_key):
        return ""

    @app.callback(
        Output("explorer-review-save-status", "children", allow_duplicate=True),
        Output("explorer-review-overrides", "data"),
        Input("explorer-review-save-btn", "n_clicks"),
        State("selected-key-store", "data"),
        State("explorer-review-class", "value"),
        State("explorer-review-confidence", "value"),
        State("explorer-review-followup", "value"),
        State("explorer-review-notes", "value"),
        State("explorer-review-overrides", "data"),
        prevent_initial_call=True,
    )
    def save_explorer_grade(
        n_clicks,
        selected_key,
        event_class,
        interest_score,
        followup_value,
        notes,
        review_overrides,
    ):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        source_frame = _apply_review_overrides(combined.df, review_overrides)
        record = _get_candidate_record_from_frame(source_frame, selected_key) or get_candidate_record_by_key(combined, selected_key)
        if record is None:
            return "No candidate selected to save.", dash.no_update

        review_db, _plot_dir = _record_review_target(record)
        candidate_id = str(record.get("candidate_id") or "").strip()
        if review_db is None or not candidate_id:
            return "Inline grading is available only for DB-backed candidates.", dash.no_update

        try:
            score = None if interest_score in (None, "") else int(interest_score)
        except Exception:
            score = None
        event_class_value = str(event_class or "unclassified").strip() or "unclassified"
        if event_class_value not in REVIEW_EVENT_CLASSES:
            event_class_value = "other"
        status = "needs_followup" if followup_value and "followup" in followup_value else "reviewed"
        reviewer = str(os.environ.get("USER") or "explorer")

        try:
            with closing(db_connect(review_db)) as conn:
                current_review = get_review(conn, candidate_id)
                save_review(
                    conn,
                    candidate_id=candidate_id,
                    interest_score=score,
                    event_class=event_class_value,
                    review_pass=max(1, int(current_review.get("review_pass", 1) or 1)),
                    notes=str(notes or ""),
                    status=status,
                    reviewer=reviewer,
                    event_type="explorer_save",
                )
                saved_review = _normalize_review_state(get_review(conn, candidate_id))
        except Exception as exc:
            return f"Save failed: {exc}", dash.no_update

        updated_overrides = dict(review_overrides or {})
        updated_overrides[str(record.get("candidate_key") or candidate_id)] = saved_review
        return f"Saved {candidate_id} to {review_db}.", updated_overrides

    @app.callback(
        Output("selected-key-store", "data"),
        [
            Input("custom-graph", "clickData"),
            Input("candidate-table", "selected_rows"),
            Input("candidate-search", "value"),
            Input("source-filter", "value"),
            Input("query-input", "value"),
            Input("x-metric", "value"),
            Input("y-metric", "value"),
            Input("log-flags", "value"),
            Input("candidate-scope-store", "data"),
            Input("explorer-review-overrides", "data"),
            *ADV_FILTER_INPUTS,
        ],
        [State("candidate-table", "data"), State("selected-key-store", "data"), *ADV_FILTER_STATES],
        prevent_initial_call=False,
    )
    def update_selected_key(*args):
        (
            custom_click,
            table_rows,
            search_value,
            source_filter,
            query_value,
            x_metric,
            y_metric,
            log_flags,
            candidate_scope_state,
            review_overrides,
            bool_values,
            num_min_values,
            num_max_values,
            text_values,
            select_values,
            only_unreviewed_value,
            require_failed_value,
            table_data,
            current_key,
            bool_ids,
            num_min_ids,
            num_max_ids,
            text_ids,
            select_ids,
        ) = args

        source_frame = _apply_review_overrides(combined.df, review_overrides)
        advanced_state = _build_advanced_filter_state(
            bool_values,
            num_min_values,
            num_max_values,
            text_values,
            select_values,
            only_unreviewed_value,
            require_failed_value,
            bool_ids,
            num_min_ids,
            num_max_ids,
            text_ids,
            select_ids,
        )
        filtered, _, _ = _filter_frame(source_frame, source_filter, query_value, advanced_state)
        working = _apply_candidate_scope(
            filtered,
            scope_state=candidate_scope_state,
            x_metric=x_metric,
            y_metric=y_metric,
            log_flags=log_flags,
        )
        ctx = dash.callback_context
        triggered = ctx.triggered_id

        def _key_from_click(click_data: dict | None) -> str | None:
            if not click_data or not click_data.get("points"):
                return None
            custom = click_data["points"][0].get("customdata")
            if isinstance(custom, (list, tuple)) and custom:
                return str(custom[0])
            if isinstance(custom, str):
                return custom
            return None

        new_key = None
        if triggered == "custom-graph":
            new_key = _key_from_click(custom_click)
        elif triggered == "candidate-table" and table_rows and table_data:
            row_idx = int(table_rows[0])
            if 0 <= row_idx < len(table_data):
                new_key = str(table_data[row_idx].get("candidate_key") or "")
        elif triggered == "candidate-search":
            new_key = find_candidate_key(combined, search_value, subset=working)

        if not new_key:
            active_frame = working if not working.empty else filtered
            if current_key and current_key in set(active_frame.get("candidate_key", [])):
                new_key = current_key
            elif not active_frame.empty:
                new_key = str(active_frame.iloc[0].get("candidate_key") or "")
            else:
                new_key = ""
        return new_key

    @app.callback(Output("candidate-search", "value"), Input("selected-key-store", "data"), prevent_initial_call=False)
    def sync_search_value(selected_key):
        record = get_candidate_record_by_key(combined, selected_key)
        if record is None:
            return ""
        return str(record.get("candidate_id") or record.get("asas_sn_id") or record.get("gaia_id") or selected_key or "")

    @app.callback(
        [
            Output("explorer-status", "children"),
            Output("custom-graph", "figure"),
            Output("plot-summary", "children"),
            Output("candidate-scope-status", "children"),
            Output("candidate-table", "data"),
        ],
        [
            Input("source-filter", "value"),
            Input("query-input", "value"),
            Input("x-metric", "value"),
            Input("y-metric", "value"),
            Input("color-metric", "value"),
            Input("symbol-metric", "value"),
            Input("log-flags", "value"),
            Input("x-min", "value"),
            Input("x-max", "value"),
            Input("y-min", "value"),
            Input("y-max", "value"),
            Input("table-sort", "value"),
            Input("selected-key-store", "data"),
            Input("theme-mode-store", "data"),
            Input("plot-reset-store", "data"),
            Input("candidate-scope-store", "data"),
            Input("explorer-review-overrides", "data"),
            *ADV_FILTER_INPUTS,
        ],
        ADV_FILTER_STATES,
    )
    def update_explorer_views(*args):
        (
            source_filter,
            query_value,
            x_metric,
            y_metric,
            color_metric,
            symbol_metric,
            log_flags,
            x_min,
            x_max,
            y_min,
            y_max,
            table_sort,
            selected_key,
            theme_mode,
            plot_reset_data,
            candidate_scope_state,
            review_overrides,
            bool_values,
            num_min_values,
            num_max_values,
            text_values,
            select_values,
            only_unreviewed_value,
            require_failed_value,
            bool_ids,
            num_min_ids,
            num_max_ids,
            text_ids,
            select_ids,
        ) = args

        source_frame = _apply_review_overrides(combined.df, review_overrides)
        advanced_state = _build_advanced_filter_state(
            bool_values,
            num_min_values,
            num_max_values,
            text_values,
            select_values,
            only_unreviewed_value,
            require_failed_value,
            bool_ids,
            num_min_ids,
            num_max_ids,
            text_ids,
            select_ids,
        )
        filtered, query_error, active_filters = _filter_frame(source_frame, source_filter, query_value, advanced_state)
        working = _apply_candidate_scope(
            filtered,
            scope_state=candidate_scope_state,
            x_metric=x_metric,
            y_metric=y_metric,
            log_flags=log_flags,
        )
        theme_name = str(theme_mode or DEFAULT_THEME)

        custom_fig: go.Figure
        plot_summary = ""
        if not x_metric or not y_metric:
            custom_fig = _status_figure(
                "Choose X and Y metrics in the sidebar to build a plot.",
                height=None,
                theme=theme_name,
                uirevision=_plot_uirevision(plot_reset_data, "custom"),
            )
        else:
            custom_fig = _scatter_figure(
                filtered,
                x=str(x_metric),
                y=str(y_metric),
                color=str(color_metric) if color_metric else None,
                symbol=str(symbol_metric) if symbol_metric else None,
                title=f"{y_metric} vs {x_metric}",
                selected_key=selected_key,
                max_points=0,
                log_x="logx" in (log_flags or []),
                log_y="logy" in (log_flags or []),
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                height=None,
                theme=theme_name,
                uirevision=_plot_uirevision(plot_reset_data, "custom"),
            )

        if any(v is not None for v in (x_min, x_max, y_min, y_max)) and x_metric in filtered.columns and y_metric in filtered.columns:
            mask = pd.Series(True, index=filtered.index, dtype="bool")
            if x_min is not None:
                mask &= numeric_series(filtered, x_metric) >= float(x_min)
            if x_max is not None:
                mask &= numeric_series(filtered, x_metric) <= float(x_max)
            if y_min is not None:
                mask &= numeric_series(filtered, y_metric) >= float(y_min)
            if y_max is not None:
                mask &= numeric_series(filtered, y_metric) <= float(y_max)
            target_col = filtered.attrs.get("default_target_col", "proxy_oneoff_dipper")
            reject_col = filtered.attrs.get("default_reject_col", "proxy_periodic_contaminant")
            summary = cut_summary(filtered, mask, target_col=target_col, reject_col=reject_col, name="box")
            purity = summary.get("purity")
            recall = summary.get("target_recall")
            leak = summary.get("reject_leakage")
            plot_summary = (
                f"Box cut selected {int(summary['selected']):,}/{int(summary['eligible']):,} rows"
                f" | purity={float(purity):.3f}" if pd.notna(purity) else f"Box cut selected {int(summary['selected']):,}/{int(summary['eligible']):,} rows"
            )
            if pd.notna(recall):
                plot_summary += f" | recall={float(recall):.3f}"
            if pd.notna(leak):
                plot_summary += f" | reject leakage={float(leak):.3f}"

        selected_record = _get_candidate_record_from_frame(source_frame, selected_key) or get_candidate_record_by_key(combined, selected_key)
        status = _selection_status(selected_record, len(filtered), query_error, active_filters)
        scope_status = _candidate_scope_status(len(filtered), len(working), candidate_scope_state)
        table_data = _table_rows(working, table_sort or "dipper_score")
        return (
            status,
            custom_fig,
            plot_summary,
            scope_status,
            table_data,
        )

    @app.callback(
        Output("period-search-store", "data"),
        Output("period-search-label", "children"),
        Output("period-cache-store", "data"),
        Input("selected-key-store", "data"),
        Input("period-search-btn", "n_clicks"),
        Input("period-auto-btn", "n_clicks"),
        State("period-min", "value"),
        State("period-max", "value"),
        State("period-method", "value"),
        State("period-cache-store", "data"),
        prevent_initial_call=False,
    )
    def update_period_search(selected_key, search_clicks, auto_clicks, min_period, max_period, method, cache):
        _ = search_clicks, auto_clicks
        record = get_candidate_record_by_key(combined, selected_key)
        if record is None:
            return None, "", cache or {}

        cache_map = dict(cache or {})
        cache_key = str(selected_key or "")
        trigger = dash.callback_context.triggered_id

        try:
            min_p = float(min_period) if min_period not in (None, "") else 0.1
            max_p = float(max_period) if max_period not in (None, "") else 100.0
        except (TypeError, ValueError):
            min_p, max_p = 0.1, 100.0
        if min_p <= 0:
            min_p = 0.01
        if max_p <= min_p:
            max_p = min_p + 1.0

        if trigger == "selected-key-store" and cache_key in cache_map:
            cached = dict(cache_map.get(cache_key) or {})
            return cached.get("result"), str(cached.get("label", "")), cache_map

        if trigger == "period-search-btn":
            chosen_method = str(method or "pdm").lower()
        else:
            chosen_method = "auto"
            if has_external_period(record):
                label = "Catalog/pipeline period"
                cache_map[cache_key] = {"result": None, "label": label}
                return None, label, cache_map

        plot_dir = infer_plot_dir_for_record(record, Path(str(record.get("plot_dir"))).expanduser().resolve() if record.get("plot_dir") else None)
        result, label = run_period_search_for_payload(
            record,
            plot_dir=plot_dir,
            min_period=min_p,
            max_period=max_p,
            method=chosen_method,
        )
        if trigger != "period-search-btn":
            if not str(label or "").lower().startswith("auto"):
                label = f"Auto {label}" if result is not None else f"Auto search: {label}"
            if isinstance(result, dict):
                result = dict(result)
                result["auto"] = True
        cache_map[cache_key] = {"result": result, "label": label}
        return result, label, cache_map

    @app.callback(
        Output("camera-checklist", "value", allow_duplicate=True),
        Input("cams-all-btn", "n_clicks"),
        Input("cams-clear-btn", "n_clicks"),
        Input("cams-invert-btn", "n_clicks"),
        State("camera-checklist", "options"),
        State("camera-checklist", "value"),
        prevent_initial_call=True,
    )
    def update_camera_selection(_all_clicks, _clear_clicks, _invert_clicks, camera_options, current_values):
        triggered = dash.callback_context.triggered_id
        all_values = [str(opt.get("value")) for opt in (camera_options or []) if str(opt.get("value") or "").strip()]
        selected = [str(v) for v in (current_values or []) if str(v) in all_values]
        if triggered == "cams-all-btn":
            return all_values
        if triggered == "cams-clear-btn":
            return []
        if triggered == "cams-invert-btn":
            selected_set = set(selected)
            return [value for value in all_values if value not in selected_set]
        return dash.no_update

    @app.callback(
        Output("lightcurve-graph", "figure"),
        Output("camera-checklist", "options"),
        Output("camera-checklist", "value"),
        Output("selected-status", "children"),
        Output("viewer-summary", "children"),
        Output("viewer-stats", "children"),
        Input("selected-key-store", "data"),
        Input("camera-checklist", "value"),
        Input("band-checklist", "value"),
        Input("panel-options", "value"),
        Input("period-input", "value"),
        Input("yaxis-mode", "value"),
        Input("period-search-store", "data"),
        Input("theme-mode-store", "data"),
        Input("plot-reset-store", "data"),
        Input("explorer-review-overrides", "data"),
    )
    def render_candidate(selected_key, selected_cameras, selected_bands, panel_options, period_value, yaxis_mode, period_search_result, theme_mode, plot_reset_data, review_overrides):
        source_frame = _apply_review_overrides(combined.df, review_overrides)
        record = _get_candidate_record_from_frame(source_frame, selected_key) or get_candidate_record_by_key(combined, selected_key)
        if record is None:
            theme_name = str(theme_mode or DEFAULT_THEME)
            return (
                _status_figure("No candidate selected", height=560, theme=theme_name, uirevision=_plot_uirevision(plot_reset_data, "lightcurve")),
                [],
                [],
                "No candidate selected",
                html.Div(),
                html.Div(),
            )

        plot_dir = infer_plot_dir_for_record(record, Path(str(record.get("plot_dir"))).expanduser().resolve() if record.get("plot_dir") else None)
        run_params = load_run_params(plot_dir)
        options = set(panel_options or [])
        theme_name = str(theme_mode or DEFAULT_THEME)

        override_period = None
        if period_value not in (None, ""):
            try:
                candidate_period = float(period_value)
                if candidate_period > 0:
                    override_period = candidate_period
            except Exception:
                override_period = None
        if override_period is None and isinstance(period_search_result, dict):
            try:
                period_candidate = float(period_search_result.get("best_period"))
                if np.isfinite(period_candidate) and period_candidate > 0:
                    override_period = period_candidate
            except Exception:
                override_period = None

        yaxis_literal = "flux" if str(yaxis_mode or "mag") == "flux" else "mag"
        native = build_interactive_lightcurve_figure(
            record,
            plot_dir=plot_dir,
            selected_cameras=selected_cameras or [],
            filter_bad_cameras="filter_bad_cameras" in options,
            show_baseline="baseline" in options,
            show_event_markers="events" in options,
            show_residuals="resid" in options,
            show_phase_fold="phase" in options,
            show_raw_mag="raw" in options,
            override_period=override_period,
            show_diagnostics="diagnostics" in options,
            confidence_colors=True,
            run_params=run_params,
            uirevision_key=_plot_uirevision(plot_reset_data, "lightcurve"),
            theme=theme_name,
            yaxis_mode=yaxis_literal,
            selected_bands=selected_bands or ["g", "V"],
        )
        native["figure"] = _style_native_explorer_figure(native["figure"])

        camera_options = list(native.get("camera_options", []))
        available_cameras = {str(opt.get("value")) for opt in camera_options if str(opt.get("value") or "").strip()}
        preserved = [str(value) for value in (selected_cameras or []) if str(value) in available_cameras]
        camera_values = preserved or list(native.get("camera_values", []))

        status_bits = []
        if native.get("status_message"):
            status_bits.append(str(native["status_message"]))
        if plot_dir is not None:
            status_bits.append(f"Plot dir: {plot_dir}")
        if native.get("warnings"):
            status_bits.append("Warnings: " + "; ".join(str(w) for w in native["warnings"]))
        status = " | ".join(status_bits)
        return (
            native["figure"],
            camera_options,
            camera_values,
            status,
            _render_summary(record),
            _render_stats(native.get("stat_rows", [])),
        )

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Dash explorer for MALCA EDA and click-through light-curve viewing.")
    parser.add_argument("--source", action="append", default=None, help="Review DB, parquet, or CSV source. May be passed multiple times.")
    parser.add_argument("--source-glob", default=None, help="Glob pattern for multiple sources, e.g. 'output/runs/*/review/review.db'")
    parser.add_argument("--source-kind", default=None, choices=["db", "parquet", "csv"], help="Optional explicit source kind")
    parser.add_argument("--plot-dir", default=None, help="Optional plot-dir override for all sources")
    parser.add_argument("--candidate", default=None, help="Candidate ID / ASAS-SN ID / Gaia ID / LC stem to select on startup")
    parser.add_argument("--candidate-key", default=None, help="Explicit candidate_key to select on startup")
    parser.add_argument("--host", default="127.0.0.1", help="Dash host")
    parser.add_argument("--port", type=int, default=8062, help="Dash port")
    parser.add_argument("--debug", action="store_true", help="Run Dash with debug enabled")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    source_paths = _resolve_sources(args)
    combined = load_combined_source_data(sources=source_paths, source_kind=args.source_kind, plot_dir=args.plot_dir)
    combined.df = add_eda_columns(combined.df)
    initial_candidate_key = _resolve_initial_candidate_key(
        combined,
        candidate_key=getattr(args, "candidate_key", None),
        candidate=getattr(args, "candidate", None),
    )
    app = build_explorer_app(
        combined,
        host=str(args.host),
        port=int(args.port),
        initial_candidate_key=initial_candidate_key,
    )
    url = f"http://{args.host}:{args.port}"
    print(f"Starting MALCA Explorer on {url}")
    print(f"  Sources: {len(combined.sources)}")
    for source in combined.sources[:10]:
        print(f"   - {source.source_label}: {source.source_path}")
    if len(combined.sources) > 10:
        print(f"   ... and {len(combined.sources) - 10} more")

    Timer(0.1, lambda: webbrowser.open(url)).start()
    app.run(host=str(args.host), port=int(args.port), debug=bool(args.debug))


if __name__ == "__main__":
    main()
