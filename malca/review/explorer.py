from __future__ import annotations

import argparse
import glob
from pathlib import Path
from threading import Timer
from typing import Any
import webbrowser

import dash
from dash import ALL, Input, Output, State, dash_table, dcc, html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from malca.review.explore_data import (
    BEST_FIELDS,
    DEFAULT_COLOR,
    DEFAULT_MAIN_X,
    DEFAULT_MAIN_Y,
    DEFAULT_SYMBOL,
    CombinedCandidateData,
    add_eda_columns,
    available_metric_columns,
    bool_series,
    cut_summary,
    discover_default_sources,
    find_candidate_key,
    get_candidate_record_by_key,
    infer_plot_dir_for_record,
    load_combined_source_data,
    load_run_params,
    numeric_series,
    text_series,
)
from malca.review.filter_schema import SIDEBAR_GROUPS
from malca.review.interactive_plot import build_interactive_lightcurve_figure
from malca.review.mini_viewer import _render_stats, _render_summary, _summary_items
from malca.review.period_search import has_external_period, run_period_search_for_payload


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


def build_explorer_app(combined: CombinedCandidateData, *, host: str, port: int) -> dash.Dash:
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
            dcc.Store(id="selected-key-store", data=combined.default_candidate_key, storage_type="local"),
            dcc.Store(id="period-search-store", data=None),
            dcc.Store(id="period-cache-store", data={}, storage_type="session"),
            dcc.Store(id="theme-mode-store", data=DEFAULT_THEME),
            dcc.Store(id="plot-reset-store", data=_default_plot_reset_data()),
            dcc.Store(id="explorer-split-init", data=0),
            dcc.Store(id="explorer-sidebar-open", data=True, storage_type="local"),
            dcc.Interval(id="explorer-init", interval=200, n_intervals=0, max_intervals=1),
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
                                            html.Div(
                                                [
                                                    html.Div("Custom plot", className="explorer-card-title"),
                                                    _reset_button("custom-reset-btn"),
                                                ],
                                                className="explorer-graph-toolbar",
                                            ),
                                            dcc.Graph(
                                                id="custom-graph",
                                                mathjax=True,
                                                config={"displaylogo": False, "scrollZoom": True, "doubleClick": False, "responsive": True},
                                                responsive=True,
                                                style={"height": "clamp(360px, 56vh, 760px)", "width": "100%"},
                                                className="explorer-graph",
                                            ),
                                            html.Div(id="plot-summary", className="explorer-status-line"),
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
                            html.Div(id="explorer-main-splitter", className="explorer-panel-splitter", title="Drag to resize panels"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div("Selected candidate", className="explorer-card-title"),
                                            html.Div(id="selected-status", className="explorer-status-line"),
                                            html.Div(id="viewer-summary", className="explorer-summary"),
                                        ],
                                        className="explorer-card",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Div("Native light curve", className="explorer-card-title"),
                                                    _reset_button("lightcurve-reset-btn"),
                                                ],
                                                className="explorer-graph-toolbar",
                                            ),
                                            dcc.Graph(
                                                id="lightcurve-graph",
                                                mathjax=True,
                                                config={"displaylogo": False, "scrollZoom": True, "doubleClick": False, "responsive": True},
                                                responsive=True,
                                                style={"height": "clamp(420px, 72vh, 1100px)", "width": "100%"},
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

    app.clientside_callback(
        """
        function(_tick) {
            var splitter = document.getElementById('explorer-main-splitter');
            var leftPanel = document.getElementById('explorer-left-panel');
            if (!splitter || !leftPanel) {
                return window.dash_clientside.no_update;
            }

            var workspace = splitter.parentElement;
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

            var storageKey = 'malca.explorer.left_panel.width.v1';
            var defaultWidth = Math.min(Math.max(window.innerWidth * 0.46, 760), 1080);
            var minWidth = 620;

            var computeMaxWidth = function() {
                var workspaceWidth = workspace.getBoundingClientRect().width;
                return Math.max(minWidth, Math.floor(workspaceWidth - 480));
            };

            var clampWidth = function(value) {
                var maxWidth = computeMaxWidth();
                var numeric = Number(value);
                if (!isFinite(numeric)) {
                    numeric = defaultWidth;
                }
                if (numeric < minWidth) {
                    numeric = minWidth;
                }
                if (numeric > maxWidth) {
                    numeric = maxWidth;
                }
                return Math.round(numeric);
            };

            var applyWidth = function(value, persist) {
                var w = clampWidth(value);
                leftPanel.style.width = String(w) + 'px';
                leftPanel.style.flex = '0 0 auto';
                leftPanel.style.minWidth = String(minWidth) + 'px';
                if (persist) {
                    try {
                        window.localStorage.setItem(storageKey, String(w));
                    } catch (e) {
                        // ignore storage failures
                    }
                }
                scheduleResize();
                return w;
            };

            if (!window.__malcaExplorerSplitterAttached) {
                var drag = {active: false, startX: 0, startWidth: 0, pointerId: null};

                var onPointerMove = function(e) {
                    if (!drag.active) {
                        return;
                    }
                    applyWidth(drag.startWidth + (e.clientX - drag.startX), false);
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
                        try { splitter.releasePointerCapture(drag.pointerId); } catch (err) {}
                    }
                    drag.pointerId = null;
                    applyWidth(leftPanel.getBoundingClientRect().width, true);
                    if (e) {
                        e.preventDefault();
                    }
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

                if (window.ResizeObserver) {
                    var observer = new window.ResizeObserver(function() {
                        scheduleResize();
                    });
                    observer.observe(workspace);
                    observer.observe(leftPanel);
                    window.__malcaExplorerResizeObserver = observer;
                }

                window.__malcaExplorerSplitterAttached = true;
            }

            var saved = null;
            try { saved = window.localStorage.getItem(storageKey); } catch (e) { saved = null; }
            var initialWidth = defaultWidth;
            if (saved !== null && saved !== '') {
                var parsed = parseInt(saved, 10);
                if (!isNaN(parsed)) {
                    initialWidth = parsed;
                }
            }
            applyWidth(initialWidth, false);
            return window.dash_clientside.no_update;
        }
        """,
        Output("explorer-split-init", "data"),
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
        Output("selected-key-store", "data"),
        [
            Input("custom-graph", "clickData"),
            Input("candidate-table", "selected_rows"),
            Input("candidate-search", "value"),
            Input("source-filter", "value"),
            Input("query-input", "value"),
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
        filtered, _, _ = _filter_frame(combined.df, source_filter, query_value, advanced_state)
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
            new_key = find_candidate_key(combined, search_value, subset=filtered)

        if not new_key:
            if current_key and current_key in set(filtered.get("candidate_key", [])):
                new_key = current_key
            elif not filtered.empty:
                new_key = str(filtered.iloc[0].get("candidate_key") or "")
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
        filtered, query_error, active_filters = _filter_frame(combined.df, source_filter, query_value, advanced_state)
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

        selected_record = get_candidate_record_by_key(combined, selected_key)
        status = _selection_status(selected_record, len(filtered), query_error, active_filters)
        table_data = _table_rows(filtered, table_sort or "dipper_score")
        return (
            status,
            custom_fig,
            plot_summary,
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
            chosen_method = "pdm"
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
    )
    def render_candidate(selected_key, selected_cameras, selected_bands, panel_options, period_value, yaxis_mode, period_search_result, theme_mode, plot_reset_data):
        record = get_candidate_record_by_key(combined, selected_key)
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
    app = build_explorer_app(combined, host=str(args.host), port=int(args.port))
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
