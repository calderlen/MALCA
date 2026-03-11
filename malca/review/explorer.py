from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

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


APP_BG = "#0a1016"
PANEL_BG = "#101923"
PANEL_BG_ALT = "#0f1720"
PANEL_BORDER = "#243645"
TEXT = "#f4f8fb"
TEXT_MUTED = "#a6bac9"
TEXT_FAINT = "#88a4b7"
ACCENT = "#6fd4ff"
ACCENT_2 = "#ffb36f"
GRID = "rgba(140, 170, 192, 0.18)"
PLOT_BG = "#111a23"
PLOT_PAPER = "#0c1218"

BASE_INPUT_STYLE = {
    "width": "100%",
    "padding": "8px 10px",
    "background": PANEL_BG,
    "color": TEXT,
    "border": f"1px solid {PANEL_BORDER}",
    "borderRadius": "6px",
    "fontFamily": "Monaco, Courier New, monospace",
    "fontSize": "11px",
}

PANEL_STYLE = {
    "background": PANEL_BG_ALT,
    "border": f"1px solid {PANEL_BORDER}",
    "borderRadius": "12px",
    "padding": "12px",
    "boxShadow": "0 0 0 1px rgba(255,255,255,0.015) inset",
}

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


def _graph_layout(height: int) -> dict[str, object]:
    return {
        "template": "plotly_dark",
        "height": height,
        "paper_bgcolor": PLOT_PAPER,
        "plot_bgcolor": PLOT_BG,
        "font": {"color": TEXT, "family": "Monaco, Courier New, monospace", "size": 11},
        "title_font": {"color": TEXT, "size": 13},
        "margin": {"l": 46, "r": 18, "t": 46, "b": 38},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(15, 23, 32, 0.65)",
            "bordercolor": "rgba(120, 150, 170, 0.22)",
            "borderwidth": 1,
            "font": {"size": 10, "color": TEXT},
        },
        "hoverlabel": {"bgcolor": "#0f1720", "font": {"color": TEXT}},
    }


def _style_plot(fig: go.Figure, *, height: int) -> go.Figure:
    fig.update_layout(**_graph_layout(height))
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, color=TEXT)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, color=TEXT)
    return fig


def _status_figure(message: str, *, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": TEXT},
    )
    _style_plot(fig, height=height)
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
    return html.Div(text, style={"fontSize": "10px", "color": TEXT_FAINT, "textTransform": "uppercase", "marginBottom": "4px"})


def _bool_filter_control(col: str) -> html.Div:
    return html.Div(
        [
            _label(col),
            dcc.Dropdown(
                id={"type": "adv-bool-mode", "col": col},
                options=[{"label": "Any", "value": "Any"}, {"label": "True", "value": "True"}, {"label": "False", "value": "False"}],
                value="Any",
                clearable=False,
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
                    dcc.Input(id={"type": "adv-num-min", "col": col}, type="number", debounce=True, placeholder="min", style={**BASE_INPUT_STYLE, "padding": "6px 8px"}),
                    dcc.Input(id={"type": "adv-num-max", "col": col}, type="number", debounce=True, placeholder="max", style={**BASE_INPUT_STYLE, "padding": "6px 8px"}),
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
            dcc.Dropdown(id={"type": "adv-text-value", "col": col}, options=options, value="Any", clearable=False),
        ],
        style={"marginBottom": "8px"},
    )


def _select_filter_control(frame: pd.DataFrame, col: str) -> html.Div:
    values = _distinct_values(frame, col)
    options = [{"label": v, "value": v} for v in values]
    return html.Div(
        [
            _label(f"{col} (exclude)"),
            dcc.Dropdown(id={"type": "adv-select-exclude", "col": col}, options=options, value=[], multi=True, placeholder="None excluded"),
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


def _build_advanced_filter_sections(frame: pd.DataFrame) -> list[html.Details]:
    return [_make_filter_group(frame, name, items) for name, items in SIDEBAR_GROUPS]


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
    height: int = 360,
) -> go.Figure:
    if x not in frame.columns or y not in frame.columns:
        return _status_figure(f"Missing columns: {', '.join(col for col in (x, y) if col not in frame.columns)}", height=height)

    data = frame.loc[frame[x].notna() & frame[y].notna()].copy()
    if data.empty:
        return _status_figure(f"No rows with both `{x}` and `{y}`", height=height)
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
    _style_plot(fig, height=height)
    fig.update_traces(marker={"size": 8, "line": {"width": 0.7, "color": "rgba(230,240,248,0.24)"}})
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
                    marker={"size": 16, "symbol": "diamond-open", "color": TEXT, "line": {"width": 2.2, "color": ACCENT}},
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
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, line={"color": ACCENT_2, "width": 2}, fillcolor="rgba(0,0,0,0)")
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
    height: int = 340,
) -> go.Figure:
    x = _resolve_metric(frame, x_candidates)
    y = _resolve_metric(frame, y_candidates)
    if x is None or y is None:
        missing = "/".join(x_candidates if x is None else y_candidates)
        return _status_figure(f"No data for {title} ({missing})", height=height)
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
    )


def _hist_figure(frame: pd.DataFrame, metric: str, color: str | None, *, height: int = 340) -> go.Figure:
    if metric not in frame.columns:
        return _status_figure(f"Missing column: {metric}", height=height)
    data = frame.loc[frame[metric].notna()].copy()
    if data.empty:
        return _status_figure(f"No rows with `{metric}`", height=height)
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
    _style_plot(fig, height=height)
    return fig


def _default_native_plot(frame: pd.DataFrame) -> tuple[str, str, str, bool, bool]:
    if {"phase_quality_score", "periodicity_score"}.issubset(frame.columns):
        non_null = frame[["phase_quality_score", "periodicity_score"]].notna().all(axis=1).sum()
        if int(non_null) > 0:
            return "phase_quality_score", "periodicity_score", "Native periodicity: phase quality vs periodicity score", False, False
    return "period_consensus_days", "dipper_score", "Catalog period vs dipper score", True, False


def _table_rows(frame: pd.DataFrame, sort_by: str, limit: int = 40) -> list[dict[str, object]]:
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
    return data.loc[:, cols].head(limit).to_dict("records")


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
    app.layout = html.Div(
        [
            dcc.Store(id="selected-key-store", data=combined.default_candidate_key),
            dcc.Store(id="period-search-store", data=None),
            dcc.Store(id="period-cache-store", data={}, storage_type="session"),
            html.Div(
                [
                    html.Div("MALCA Explorer", style={"fontSize": "24px", "fontWeight": "600", "color": TEXT}),
                    html.Div(f"Sources: {len(combined.sources)} | Rows: {len(combined.df):,}", style={"fontSize": "12px", "color": TEXT_MUTED}),
                ],
                style={"paddingBottom": "10px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            _label("Sources"),
                            dcc.Dropdown(id="source-filter", options=[{"label": label, "value": label} for label in all_source_labels], value=all_source_labels, multi=True),
                        ],
                        style={"minWidth": "280px", "flex": "1 1 280px"},
                    ),
                    html.Div(
                        [
                            _label("Query"),
                            dcc.Input(id="query-input", debounce=True, placeholder="pandas query, e.g. dipper_score >= 5 and period_n_sources <= 1", style=BASE_INPUT_STYLE),
                        ],
                        style={"minWidth": "340px", "flex": "2 1 420px"},
                    ),
                    html.Div(
                        [
                            _label("Find candidate"),
                            dcc.Input(id="candidate-search", debounce=True, placeholder="candidate_id / asas_sn_id / gaia_id", style=BASE_INPUT_STYLE),
                        ],
                        style={"minWidth": "240px", "flex": "1 1 260px"},
                    ),
                ],
                style={**PANEL_STYLE, "display": "flex", "gap": "10px", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Diagnostics", style={"fontSize": "13px", "fontWeight": "600", "color": TEXT, "paddingBottom": "8px"}),
                            html.Div(
                                [
                                    dcc.Dropdown(id="x-metric", options=[{"label": col, "value": col} for col in metric_options], value=DEFAULT_MAIN_X if DEFAULT_MAIN_X in metric_options else (metric_options[0] if metric_options else None), clearable=False),
                                    dcc.Dropdown(id="y-metric", options=[{"label": col, "value": col} for col in metric_options], value=DEFAULT_MAIN_Y if DEFAULT_MAIN_Y in metric_options else (metric_options[0] if metric_options else None), clearable=False),
                                    dcc.Dropdown(id="color-metric", options=[{"label": col, "value": col} for col in metric_options], value=DEFAULT_COLOR if DEFAULT_COLOR in metric_options else None, clearable=True),
                                    dcc.Dropdown(id="symbol-metric", options=[{"label": col, "value": col} for col in metric_options], value=DEFAULT_SYMBOL if DEFAULT_SYMBOL in metric_options else None, clearable=True),
                                    dcc.Input(id="sample-size", type="number", min=200, step=200, value=4000, placeholder="sample", style=BASE_INPUT_STYLE),
                                    dcc.Checklist(id="log-flags", options=[{"label": " log x", "value": "logx"}, {"label": " log y", "value": "logy"}], value=[], inline=True, style={"color": TEXT, "fontSize": "12px", "display": "flex", "alignItems": "center"}),
                                ],
                                style={"display": "grid", "gridTemplateColumns": "repeat(6, minmax(120px, 1fr))", "gap": "8px", "marginBottom": "10px"},
                            ),
                            html.Div(
                                [
                                    dcc.Input(id="x-min", type="number", placeholder="x min", style=BASE_INPUT_STYLE),
                                    dcc.Input(id="x-max", type="number", placeholder="x max", style=BASE_INPUT_STYLE),
                                    dcc.Input(id="y-min", type="number", placeholder="y min", style=BASE_INPUT_STYLE),
                                    dcc.Input(id="y-max", type="number", placeholder="y max", style=BASE_INPUT_STYLE),
                                    dcc.Dropdown(id="hist-metric", options=[{"label": col, "value": col} for col in metric_options], value=default_hist_metric, clearable=False),
                                    dcc.Dropdown(id="table-sort", options=[{"label": col, "value": col} for col in metric_options], value="dipper_score" if "dipper_score" in metric_options else default_hist_metric, clearable=False),
                                ],
                                style={"display": "grid", "gridTemplateColumns": "repeat(6, minmax(120px, 1fr))", "gap": "8px", "marginBottom": "10px"},
                            ),
                            html.Div(id="explorer-status", style={"fontSize": "12px", "color": TEXT_MUTED, "padding": "2px 0 10px 0"}),
                            html.Details(
                                [
                                    html.Summary("Advanced Filters (all regular GUI filters)", style={"color": TEXT, "cursor": "pointer", "fontSize": "12px", "fontWeight": "600"}),
                                    html.Div(
                                        [
                                            dcc.Checklist(id="only-unreviewed", options=[{"label": " Only unreviewed", "value": "yes"}], value=[], style={"color": TEXT, "marginBottom": "6px", "fontSize": "11px"}),
                                            dcc.Checklist(id="require-failed-any-false", options=[{"label": " Require failed_any=False", "value": "yes"}], value=[], style={"color": TEXT, "marginBottom": "10px", "fontSize": "11px"}),
                                            *advanced_sections,
                                        ],
                                        style={"paddingTop": "10px"},
                                    ),
                                ],
                                style={**PANEL_STYLE, "marginBottom": "12px"},
                            ),
                            dcc.Graph(id="main-graph", mathjax=True, config={"displaylogo": False, "scrollZoom": True}, style={"height": "470px"}),
                            html.Div(id="cut-summary", style={"fontSize": "12px", "color": TEXT_MUTED, "padding": "6px 0 4px 0"}),
                            html.Div(
                                [
                                    dcc.Graph(id="catalog-graph", mathjax=True, config={"displaylogo": False}, style={"height": "340px"}),
                                    dcc.Graph(id="regularity-graph", mathjax=True, config={"displaylogo": False}, style={"height": "340px"}),
                                    dcc.Graph(id="repeatability-graph", mathjax=True, config={"displaylogo": False}, style={"height": "340px"}),
                                    dcc.Graph(id="score-variability-graph", mathjax=True, config={"displaylogo": False}, style={"height": "340px"}),
                                    dcc.Graph(id="stetson-graph", mathjax=True, config={"displaylogo": False}, style={"height": "340px"}),
                                    dcc.Graph(id="shape-graph", mathjax=True, config={"displaylogo": False}, style={"height": "340px"}),
                                    dcc.Graph(id="native-graph", mathjax=True, config={"displaylogo": False}, style={"height": "340px"}),
                                    dcc.Graph(id="hist-graph", mathjax=True, config={"displaylogo": False}, style={"height": "340px"}),
                                ],
                                style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(280px, 1fr))", "gap": "10px", "marginTop": "10px"},
                            ),
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
                                page_size=12,
                                style_table={"overflowX": "auto", "marginTop": "12px"},
                                style_cell={"backgroundColor": PANEL_BG, "color": TEXT, "border": f"1px solid {PANEL_BORDER}", "fontFamily": "Monaco, Courier New, monospace", "fontSize": "11px", "padding": "6px"},
                                style_header={"backgroundColor": "#172430", "color": TEXT, "fontWeight": "600", "border": f"1px solid {PANEL_BORDER}"},
                                row_selectable="single",
                            ),
                        ],
                        style={"minHeight": 0, "overflowY": "auto", "paddingRight": "10px"},
                    ),
                    html.Div(
                        [
                            html.Div("Selected candidate", style={"fontSize": "13px", "fontWeight": "600", "color": TEXT, "paddingBottom": "6px"}),
                            html.Div(id="selected-status", style={"fontSize": "12px", "color": TEXT_MUTED, "paddingBottom": "6px"}),
                            html.Div(id="viewer-summary", style={"paddingBottom": "10px"}),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dcc.Dropdown(id="camera-dropdown", multi=True, placeholder="All cameras"),
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
                                                inline=True,
                                                style={"color": TEXT, "fontSize": "12px", "display": "flex", "flexWrap": "wrap", "gap": "8px"},
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Input(id="period-input", type="number", debounce=True, placeholder="Manual period override (days)", style={**BASE_INPUT_STYLE, "width": "220px"}),
                                                    dcc.RadioItems(id="yaxis-mode", options=[{"label": " mag", "value": "mag"}, {"label": " flux", "value": "flux"}], value="mag", inline=True, style={"color": TEXT, "fontSize": "12px"}),
                                                ],
                                                style={"display": "flex", "gap": "10px", "alignItems": "center", "paddingTop": "8px", "flexWrap": "wrap"},
                                            ),
                                        ],
                                        style={"paddingBottom": "10px"},
                                    ),
                                    html.Div(
                                        [
                                            _label("Period search"),
                                            html.Div(
                                                [
                                                    dcc.Dropdown(id="period-method", options=[{"label": "PDM", "value": "pdm"}, {"label": "CE", "value": "ce"}, {"label": "LSP", "value": "lsp"}], value="pdm", clearable=False, style={"minWidth": "120px"}),
                                                    dcc.Input(id="period-min", type="number", debounce=True, placeholder="min d", value=0.1, style={**BASE_INPUT_STYLE, "width": "100px"}),
                                                    dcc.Input(id="period-max", type="number", debounce=True, placeholder="max d", value=100.0, style={**BASE_INPUT_STYLE, "width": "100px"}),
                                                    html.Button("Auto", id="period-auto-btn", n_clicks=0, style={"padding": "8px 12px", "background": "#173246", "color": TEXT, "border": f"1px solid {PANEL_BORDER}", "borderRadius": "6px", "cursor": "pointer"}),
                                                    html.Button("Find Period", id="period-search-btn", n_clicks=0, style={"padding": "8px 12px", "background": "#264b2a", "color": TEXT, "border": f"1px solid {PANEL_BORDER}", "borderRadius": "6px", "cursor": "pointer"}),
                                                ],
                                                style={"display": "flex", "gap": "8px", "alignItems": "center", "flexWrap": "wrap"},
                                            ),
                                            html.Div(id="period-search-label", style={"fontSize": "12px", "color": TEXT_MUTED, "paddingTop": "8px"}),
                                        ],
                                        style={"paddingTop": "4px"},
                                    ),
                                ],
                                style={**PANEL_STYLE, "marginBottom": "10px"},
                            ),
                            dcc.Graph(id="lightcurve-graph", mathjax=True, config={"displaylogo": False, "scrollZoom": True}, style={"height": "72vh", "border": f"1px solid {PANEL_BORDER}", "borderRadius": "10px", "background": "#0c1218"}),
                            html.Div(
                                [
                                    html.Div("Precomputed stats", style={"fontSize": "13px", "fontWeight": "600", "color": TEXT, "paddingBottom": "6px"}),
                                    html.Div(id="viewer-stats"),
                                ],
                                style={**PANEL_STYLE, "marginTop": "10px"},
                            ),
                        ],
                        style={"minHeight": 0, "overflowY": "auto", "paddingLeft": "10px"},
                    ),
                ],
                style={"display": "grid", "gridTemplateColumns": "minmax(760px, 1.6fr) minmax(520px, 1fr)", "gap": "10px", "minHeight": 0, "paddingTop": "12px"},
            ),
        ],
        className="review-explorer-app",
        style={
            "background": APP_BG,
            "height": "100vh",
            "display": "grid",
            "gridTemplateRows": "auto auto minmax(0, 1fr)",
            "padding": "16px",
            "gap": "10px",
            "fontFamily": "Monaco, Courier New, monospace",
            "overflow": "hidden",
        },
    )

    @app.callback(
        Output("selected-key-store", "data"),
        [
            Input("main-graph", "clickData"),
            Input("catalog-graph", "clickData"),
            Input("regularity-graph", "clickData"),
            Input("repeatability-graph", "clickData"),
            Input("score-variability-graph", "clickData"),
            Input("stetson-graph", "clickData"),
            Input("shape-graph", "clickData"),
            Input("native-graph", "clickData"),
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
            main_click,
            catalog_click,
            regularity_click,
            repeatability_click,
            score_var_click,
            stetson_click,
            shape_click,
            native_click,
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

        click_map = {
            "main-graph": main_click,
            "catalog-graph": catalog_click,
            "regularity-graph": regularity_click,
            "repeatability-graph": repeatability_click,
            "score-variability-graph": score_var_click,
            "stetson-graph": stetson_click,
            "shape-graph": shape_click,
            "native-graph": native_click,
        }

        new_key = None
        if triggered in click_map:
            new_key = _key_from_click(click_map.get(triggered))
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
            Output("main-graph", "figure"),
            Output("catalog-graph", "figure"),
            Output("regularity-graph", "figure"),
            Output("repeatability-graph", "figure"),
            Output("score-variability-graph", "figure"),
            Output("stetson-graph", "figure"),
            Output("shape-graph", "figure"),
            Output("native-graph", "figure"),
            Output("hist-graph", "figure"),
            Output("cut-summary", "children"),
            Output("candidate-table", "data"),
        ],
        [
            Input("source-filter", "value"),
            Input("query-input", "value"),
            Input("x-metric", "value"),
            Input("y-metric", "value"),
            Input("color-metric", "value"),
            Input("symbol-metric", "value"),
            Input("sample-size", "value"),
            Input("log-flags", "value"),
            Input("x-min", "value"),
            Input("x-max", "value"),
            Input("y-min", "value"),
            Input("y-max", "value"),
            Input("hist-metric", "value"),
            Input("table-sort", "value"),
            Input("selected-key-store", "data"),
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
            sample_size,
            log_flags,
            x_min,
            x_max,
            y_min,
            y_max,
            hist_metric,
            table_sort,
            selected_key,
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
        sample_n = max(int(sample_size or 4000), 200)

        main_fig = _scatter_figure(
            filtered,
            x=x_metric or DEFAULT_MAIN_X,
            y=y_metric or DEFAULT_MAIN_Y,
            color=color_metric,
            symbol=symbol_metric,
            title="Main metric scatter",
            selected_key=selected_key,
            max_points=sample_n,
            log_x="logx" in (log_flags or []),
            log_y="logy" in (log_flags or []),
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            height=460,
        )
        catalog_fig = _scatter_figure(
            filtered,
            x="period_n_sources",
            y="dip_run_count",
            color="known_periodic_catalog",
            symbol="oneoff_like",
            title="Catalog support vs dip recurrence",
            selected_key=selected_key,
            max_points=min(sample_n, 3000),
        )
        regularity_fig = _scatter_figure(
            filtered,
            x="dip_inter_event_spacing_median",
            y="dip_inter_event_spacing_std",
            color="periodic_evidence_bucket",
            symbol=None,
            title="Recurrence regularity",
            selected_key=selected_key,
            max_points=min(sample_n, 3000),
            log_x=True,
            log_y=True,
        )
        repeatability_fig = _metric_pair_figure(
            filtered,
            x_candidates=["dip_amplitude_consistency"],
            y_candidates=["dip_duration_consistency"],
            title="Dip repeatability: amplitude vs duration consistency",
            color="oneoff_like",
            symbol=None,
            selected_key=selected_key,
            max_points=min(sample_n, 3000),
        )
        score_variability_fig = _metric_pair_figure(
            filtered,
            x_candidates=["stats_photometry_robust_sigma_mag", "stats_amplitude", "stats_photometry_IQR_mag"],
            y_candidates=["dipper_score"],
            title="Dipper score vs variability strength",
            color="oneoff_like",
            symbol=None,
            selected_key=selected_key,
            max_points=min(sample_n, 3000),
        )
        stetson_fig = _metric_pair_figure(
            filtered,
            x_candidates=["stats_photometry_robust_sigma_mag", "stats_amplitude", "stats_percent_amplitude"],
            y_candidates=["stats_variability_stetson_J", "stats_variability_stetson_K"],
            title="Scatter vs Stetson variability",
            color="periodic_evidence_bucket",
            symbol=None,
            selected_key=selected_key,
            max_points=min(sample_n, 3000),
        )
        shape_fig = _metric_pair_figure(
            filtered,
            x_candidates=["stats_skew", "stats_gskew"],
            y_candidates=["stats_max_slope", "stats_percent_amplitude", "stats_amplitude"],
            title="Shape and impulsiveness diagnostics",
            color="oneoff_like",
            symbol=None,
            selected_key=selected_key,
            max_points=min(sample_n, 3000),
        )
        native_x, native_y, native_title, native_log_x, native_log_y = _default_native_plot(filtered)
        native_fig = _scatter_figure(
            filtered,
            x=native_x,
            y=native_y,
            color="periodic_evidence_bucket",
            symbol=None,
            title=native_title,
            selected_key=selected_key,
            max_points=min(sample_n, 3000),
            log_x=native_log_x,
            log_y=native_log_y,
        )
        hist_fig = _hist_figure(filtered, hist_metric or "dipper_score", color_metric or "periodic_evidence_bucket")

        cut_text = ""
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
            cut_text = (
                f"Box cut selected {int(summary['selected']):,}/{int(summary['eligible']):,} rows"
                f" | purity={float(purity):.3f}" if pd.notna(purity) else f"Box cut selected {int(summary['selected']):,}/{int(summary['eligible']):,} rows"
            )
            if pd.notna(recall):
                cut_text += f" | recall={float(recall):.3f}"
            if pd.notna(leak):
                cut_text += f" | reject leakage={float(leak):.3f}"

        selected_record = get_candidate_record_by_key(combined, selected_key)
        status = _selection_status(selected_record, len(filtered), query_error, active_filters)
        table_data = _table_rows(filtered, table_sort or "dipper_score")
        return (
            status,
            main_fig,
            catalog_fig,
            regularity_fig,
            repeatability_fig,
            score_variability_fig,
            stetson_fig,
            shape_fig,
            native_fig,
            hist_fig,
            cut_text,
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
        Output("lightcurve-graph", "figure"),
        Output("camera-dropdown", "options"),
        Output("camera-dropdown", "value"),
        Output("selected-status", "children"),
        Output("viewer-summary", "children"),
        Output("viewer-stats", "children"),
        Input("selected-key-store", "data"),
        Input("camera-dropdown", "value"),
        Input("panel-options", "value"),
        Input("period-input", "value"),
        Input("yaxis-mode", "value"),
        Input("period-search-store", "data"),
    )
    def render_candidate(selected_key, selected_cameras, panel_options, period_value, yaxis_mode, period_search_result):
        record = get_candidate_record_by_key(combined, selected_key)
        if record is None:
            return _status_figure("No candidate selected", height=560), [], [], "No candidate selected", html.Div(), html.Div()

        plot_dir = infer_plot_dir_for_record(record, Path(str(record.get("plot_dir"))).expanduser().resolve() if record.get("plot_dir") else None)
        run_params = load_run_params(plot_dir)
        options = set(panel_options or [])

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
            uirevision_key=f"explore:{selected_key}",
            theme="black",
            yaxis_mode=yaxis_literal,
        )

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
            native.get("camera_options", []),
            native.get("camera_values", []),
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
    print(f"Starting MALCA Explorer on http://{args.host}:{args.port}")
    print(f"  Sources: {len(combined.sources)}")
    for source in combined.sources[:10]:
        print(f"   - {source.source_label}: {source.source_path}")
    if len(combined.sources) > 10:
        print(f"   ... and {len(combined.sources) - 10} more")
    app.run(host=str(args.host), port=int(args.port), debug=bool(args.debug))


if __name__ == "__main__":
    main()
