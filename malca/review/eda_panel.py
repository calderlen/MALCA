from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from malca.plotting.lightcurve_publication import PUBLICATION_PLOTLY_FONT
from malca.review.eda_data import (
    DEFAULT_COLOR,
    DEFAULT_MAIN_X,
    DEFAULT_MAIN_Y,
    DEFAULT_SYMBOL,
    add_eda_columns,
    available_metric_columns,
    load_review_db,
)
from malca.review.publication import publication_figure


EDA_TABLE_COLUMNS = [
    {"name": "candidate_id", "id": "candidate_id"},
    {"name": "asas_sn_id", "id": "asas_sn_id"},
    {"name": "gaia_id", "id": "gaia_id"},
    {"name": "status", "id": "status"},
    {"name": "event_class", "id": "event_class"},
    {"name": "interest_score", "id": "interest_score"},
    {"name": "dipper_score", "id": "dipper_score"},
    {"name": "period_n_sources", "id": "period_n_sources"},
    {"name": "period_consensus_days", "id": "period_consensus_days"},
    {"name": "dip_run_count", "id": "dip_run_count"},
    {"name": "periodic_evidence_bucket", "id": "periodic_evidence_bucket"},
]

EDA_COLOR_SEQUENCE = [
    "#2f80ed",
    "#27ae60",
    "#f2994a",
    "#eb5757",
    "#9b51e0",
    "#00a6a6",
    "#b7791f",
    "#6b7280",
]
EDA_SYMBOL_SEQUENCE = [
    "circle",
    "square",
    "diamond",
    "triangle-up",
    "triangle-down",
    "cross",
    "x",
]


def _source_label_for_db(path: Path) -> str:
    if path.suffix.lower() == ".db" and path.parent.name == "review":
        return path.parent.parent.name
    return path.stem


@lru_cache(maxsize=8)
def _load_review_eda_frame_cached(db_path_text: str, db_signature: str) -> pd.DataFrame:
    _ = db_signature
    path = Path(db_path_text).expanduser().resolve()
    frame = load_review_db(path)
    if "candidate_id" not in frame.columns:
        frame["candidate_id"] = pd.Series(dtype="object")
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    if "candidate_key" not in frame.columns:
        frame["candidate_key"] = frame["candidate_id"]
    if "source_label" not in frame.columns:
        frame["source_label"] = _source_label_for_db(path)
    if "source_file" not in frame.columns:
        frame["source_file"] = str(path)
    return add_eda_columns(frame)


def load_review_eda_frame(db_path: str | Path, db_signature: str) -> pd.DataFrame:
    """Load EDA-ready candidate/review data, invalidated by DB signature."""
    path = Path(db_path).expanduser().resolve()
    return _load_review_eda_frame_cached(str(path), str(db_signature or "")).copy()


def queue_eda_frame(frame: pd.DataFrame, candidate_ids: list[str] | tuple[str, ...] | None) -> pd.DataFrame:
    """Restrict an EDA frame to queue candidate ids while preserving queue order."""
    ids = [str(candidate_id).strip() for candidate_id in (candidate_ids or []) if str(candidate_id).strip()]
    if not ids or "candidate_id" not in frame.columns:
        return frame.iloc[0:0].copy()

    order = pd.Series(range(len(ids)), index=ids)
    out = frame[frame["candidate_id"].astype(str).isin(order.index)].copy()
    if out.empty:
        return out
    out["_queue_order"] = out["candidate_id"].astype(str).map(order)
    out = out.sort_values("_queue_order", kind="stable").drop(columns=["_queue_order"])
    return out.reset_index(drop=True)


def _eda_option_columns(frame: pd.DataFrame) -> list[str]:
    columns = available_metric_columns(frame)
    for col in (
        DEFAULT_COLOR,
        DEFAULT_SYMBOL,
        "status",
        "event_class",
        "final_class",
        "review_label",
        "known_periodic_catalog",
        "oneoff_like",
        "source_label",
    ):
        if col in frame.columns and col not in columns:
            columns.append(col)
    return columns


def eda_metric_options(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [{"label": col, "value": col} for col in _eda_option_columns(frame)]


def resolve_eda_metric_values(
    frame: pd.DataFrame,
    *,
    x_metric: object = None,
    y_metric: object = None,
    color_metric: object = None,
    symbol_metric: object = None,
) -> tuple[str | None, str | None, str | None, str | None]:
    metrics = available_metric_columns(frame)
    option_set = set(_eda_option_columns(frame))
    metric_set = set(metrics)

    x = str(x_metric) if x_metric in option_set else (DEFAULT_MAIN_X if DEFAULT_MAIN_X in metric_set else None)
    if x is None and metrics:
        x = metrics[0]

    y = str(y_metric) if y_metric in option_set else (DEFAULT_MAIN_Y if DEFAULT_MAIN_Y in metric_set else None)
    if y is None:
        y = next((metric for metric in metrics if metric != x), None)

    color = str(color_metric) if color_metric in frame.columns else (DEFAULT_COLOR if DEFAULT_COLOR in frame.columns else None)
    symbol = str(symbol_metric) if symbol_metric in frame.columns else (DEFAULT_SYMBOL if DEFAULT_SYMBOL in frame.columns else None)
    return x, y, color, symbol


def _theme_palette(theme: str | None) -> dict[str, str]:
    if str(theme or "").lower() == "white":
        return {
            "paper": "#ffffff",
            "plot": "#ffffff",
            "grid": "#d5dee8",
            "text": "#18242f",
            "muted": "#5f6f7f",
            "accent": "#245f8f",
            "selected": "#111827",
            "marker_line": "#ffffff",
        }
    if str(theme or "").lower() == "gray":
        return {
            "paper": "#111820",
            "plot": "#111820",
            "grid": "#304354",
            "text": "#dce8f2",
            "muted": "#9db4c7",
            "accent": "#49b7ff",
            "selected": "#ffffff",
            "marker_line": "#0a0f14",
        }
    return {
        "paper": "#000000",
        "plot": "#000000",
        "grid": "#233544",
        "text": "#dce8f2",
        "muted": "#9db4c7",
        "accent": "#0aa7ff",
        "selected": "#ffffff",
        "marker_line": "#050505",
    }


def _style_eda_figure(fig: go.Figure, *, theme: str | None, height: int = 360) -> go.Figure:
    colors = _theme_palette(theme)
    fig.update_layout(
        template=None,
        paper_bgcolor=colors["paper"],
        plot_bgcolor=colors["plot"],
        font={"color": colors["text"], "family": PUBLICATION_PLOTLY_FONT, "size": 11},
        margin={"l": 58, "r": 20, "t": 42, "b": 58},
        height=height,
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 10, "family": PUBLICATION_PLOTLY_FONT},
        },
    )
    fig.update_xaxes(gridcolor=colors["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=colors["grid"], zeroline=False)
    return fig


def eda_status_figure(message: str, *, theme: str | None = None, height: int = 360) -> go.Figure:
    colors = _theme_palette(theme)
    fig = go.Figure()
    fig.add_annotation(
        text=str(message),
        showarrow=False,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        font={"color": colors["muted"], "family": PUBLICATION_PLOTLY_FONT, "size": 13},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style_eda_figure(fig, theme=theme, height=height)


def _plot_data_and_counts(
    frame: pd.DataFrame,
    *,
    x_metric: str | None,
    y_metric: str | None,
    log_x: bool = False,
    log_y: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts: dict[str, Any] = {
        "queue_rows": int(len(frame)),
        "plottable_rows": 0,
        "dropped_missing": 0,
        "dropped_nonpositive": 0,
        "dropped_total": int(len(frame)),
        "missing_metrics": [],
    }
    if frame.empty or not x_metric or not y_metric:
        return frame.iloc[0:0].copy(), counts

    missing = [metric for metric in (x_metric, y_metric) if metric not in frame.columns]
    counts["missing_metrics"] = missing
    if missing:
        return frame.iloc[0:0].copy(), counts

    x_values = pd.to_numeric(frame[x_metric], errors="coerce")
    y_values = pd.to_numeric(frame[y_metric], errors="coerce")
    nonmissing = x_values.notna() & y_values.notna()
    nonpositive = pd.Series(False, index=frame.index)
    if log_x:
        nonpositive |= nonmissing & (x_values <= 0)
    if log_y:
        nonpositive |= nonmissing & (y_values <= 0)
    valid = nonmissing & (~nonpositive)

    counts["dropped_missing"] = int((~nonmissing).sum())
    counts["dropped_nonpositive"] = int(nonpositive.sum())
    counts["plottable_rows"] = int(valid.sum())
    counts["dropped_total"] = int(len(frame) - valid.sum())

    data = frame.loc[valid].copy()
    if not data.empty:
        data[x_metric] = x_values.loc[valid].astype(float)
        data[y_metric] = y_values.loc[valid].astype(float)
    return data.reset_index(drop=True), counts


def eda_plot_row_counts(
    frame: pd.DataFrame,
    *,
    x_metric: str | None,
    y_metric: str | None,
    log_x: bool = False,
    log_y: bool = False,
) -> dict[str, Any]:
    """Return row-count diagnostics for the current EDA metric pair."""
    _data, counts = _plot_data_and_counts(frame, x_metric=x_metric, y_metric=y_metric, log_x=log_x, log_y=log_y)
    return counts


def _plot_status_message(counts: dict[str, Any], x_metric: str | None, y_metric: str | None) -> str:
    missing = counts.get("missing_metrics") or []
    if missing:
        return f"Missing metric: {', '.join(str(metric) for metric in missing)}"
    if not x_metric or not y_metric:
        return "Choose X and Y metrics."
    if counts.get("queue_rows", 0) <= 0:
        return "The current review queue is empty."
    if counts.get("dropped_nonpositive", 0):
        return f"No plottable rows remain after log-axis filtering for {y_metric} vs {x_metric}."
    return f"No queue rows have plottable {x_metric} and {y_metric} values."


def _current_marker_trace(
    *,
    x: list[float] | tuple[float, ...] | None,
    y: list[float] | tuple[float, ...] | None,
    customdata: list[list[str]] | tuple[tuple[str, ...], ...] | None,
    colors: dict[str, str],
    use_webgl: bool,
):
    scatter_cls = go.Scattergl if use_webgl else go.Scatter
    return scatter_cls(
        x=list(x or []),
        y=list(y or []),
        mode="markers",
        marker={
            "size": 15,
            "symbol": "diamond-open",
            "color": colors["selected"],
            "line": {"width": 2.2, "color": colors["accent"]},
        },
        name="current",
        customdata=list(customdata or []),
        hovertemplate="current %{customdata[0]}<extra></extra>",
        showlegend=False,
    )


def _categorical_label(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        if pd.isna(value):
            return "missing"
    except Exception:
        pass
    text = str(value).strip()
    return text or "missing"


def _hover_text(row: pd.Series, hover_cols: list[str]) -> str:
    parts = []
    for col in hover_cols:
        value = row.get(col)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        parts.append(f"{col}: {value}")
    return "<br>".join(parts)


def eda_scatter_figure(
    frame: pd.DataFrame,
    *,
    x_metric: str | None,
    y_metric: str | None,
    color_metric: str | None = None,
    symbol_metric: str | None = None,
    selected_candidate_id: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    theme: str | None = None,
    height: int = 360,
    use_webgl: bool = True,
) -> go.Figure:
    data, counts = _plot_data_and_counts(frame, x_metric=x_metric, y_metric=y_metric, log_x=log_x, log_y=log_y)
    if data.empty:
        colors = _theme_palette(theme)
        fig = eda_status_figure(_plot_status_message(counts, x_metric, y_metric), theme=theme, height=height)
        fig.add_trace(
            _current_marker_trace(
                x=[],
                y=[],
                customdata=[],
                colors=colors,
                use_webgl=use_webgl,
            )
        )
        return fig

    colors = _theme_palette(theme)
    hover_cols = [
        col
        for col in ("candidate_id", "asas_sn_id", "gaia_id", "status", "event_class", "dipper_score")
        if col in data.columns
    ]
    fig = go.Figure()
    color_col = color_metric if color_metric in data.columns else None
    symbol_col = symbol_metric if symbol_metric in data.columns else None
    color_numeric = pd.to_numeric(data[color_col], errors="coerce") if color_col else None
    numeric_color = bool(color_col and color_numeric is not None and color_numeric.notna().any())
    color_values = (
        sorted(data[color_col].dropna().unique(), key=lambda value: str(value))
        if color_col and not numeric_color
        else [None]
    )
    symbol_values = (
        sorted(data[symbol_col].dropna().unique(), key=lambda value: str(value))
        if symbol_col
        else [None]
    )
    color_map = {
        _categorical_label(value): EDA_COLOR_SEQUENCE[idx % len(EDA_COLOR_SEQUENCE)]
        for idx, value in enumerate(color_values)
    }
    symbol_map = {
        _categorical_label(value): EDA_SYMBOL_SEQUENCE[idx % len(EDA_SYMBOL_SEQUENCE)]
        for idx, value in enumerate(symbol_values)
    }
    group_cols = [col for col in (color_col if not numeric_color else None, symbol_col) if col]
    scatter_cls = go.Scattergl if use_webgl else go.Scatter
    groups = [((), data)] if not group_cols else list(data.groupby(group_cols, dropna=False, sort=True, observed=True))
    for raw_key, group in groups:
        if group.empty:
            continue
        keys = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        key_by_col = dict(zip(group_cols, keys))
        color_label = _categorical_label(key_by_col.get(color_col)) if color_col and not numeric_color else None
        symbol_label = _categorical_label(key_by_col.get(symbol_col)) if symbol_col else None
        name_parts = []
        if color_label is not None:
            name_parts.append(str(color_label))
        if symbol_label is not None:
            name_parts.append(str(symbol_label))
        trace_name = " | ".join(name_parts) if name_parts else "queue"
        marker: dict[str, Any] = {
            "size": 7,
            "symbol": symbol_map.get(symbol_label or "missing", "circle"),
            "line": {"width": 0.6, "color": colors["marker_line"]},
        }
        if numeric_color and color_col:
            marker_colors: list[float | None] = []
            for value in pd.to_numeric(group[color_col], errors="coerce").tolist():
                try:
                    marker_colors.append(float(value) if np.isfinite(float(value)) else None)
                except Exception:
                    marker_colors.append(None)
            marker.update(
                {
                    "color": marker_colors,
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": {"text": str(color_col)}},
                }
            )
        else:
            marker["color"] = color_map.get(color_label or "missing", colors["accent"])
        fig.add_trace(
            scatter_cls(
                x=[float(v) for v in group[x_metric].tolist()],
                y=[float(v) for v in group[y_metric].tolist()],
                mode="markers",
                marker=marker,
                opacity=0.78,
                name=trace_name,
                showlegend=bool(name_parts),
                customdata=[[str(v)] for v in group["candidate_id"].astype(str).tolist()],
                text=[_hover_text(row, hover_cols) for _idx, row in group.iterrows()],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    _style_eda_figure(fig, theme=theme, height=height)
    fig.update_layout(title=None)
    fig.update_xaxes(title_text=str(x_metric), title_standoff=8, type="log" if log_x else "linear")
    fig.update_yaxes(title_text=str(y_metric), title_standoff=10, type="log" if log_y else "linear")

    selected_id = str(selected_candidate_id or "").strip()
    selected = (
        data[data["candidate_id"].astype(str) == selected_id]
        if selected_id
        else data.iloc[0:0]
    )
    fig.add_trace(
        _current_marker_trace(
            x=[float(v) for v in selected[x_metric].tolist()],
            y=[float(v) for v in selected[y_metric].tolist()],
            customdata=[[str(v)] for v in selected["candidate_id"].astype(str).tolist()],
            colors=colors,
            use_webgl=use_webgl,
        )
    )
    return fig


def eda_publication_figure(
    frame: pd.DataFrame,
    *,
    x_metric: str | None,
    y_metric: str | None,
    color_metric: str | None = None,
    symbol_metric: str | None = None,
    selected_candidate_id: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    width: int = 1200,
    height: int = 820,
) -> go.Figure:
    """Build a white, vector-safe EDA scatter for PDF export."""
    base = eda_scatter_figure(
        frame,
        x_metric=x_metric,
        y_metric=y_metric,
        color_metric=color_metric,
        symbol_metric=symbol_metric,
        selected_candidate_id=selected_candidate_id,
        log_x=log_x,
        log_y=log_y,
        theme="white",
        height=height,
        use_webgl=False,
    )
    for trace in base.data:
        if str(trace.name or "") == "current":
            trace.opacity = 1.0
            trace.marker.size = 17
            trace.marker.symbol = "diamond-open"
            trace.marker.color = "#ffffff"
            trace.marker.line = {"width": 2.8, "color": "#111827"}
        elif hasattr(trace, "marker"):
            trace.opacity = 0.82
            trace.marker.size = 8
            trace.marker.line = {"width": 0.45, "color": "rgba(17, 24, 39, 0.45)"}

    title = f"{y_metric} vs {x_metric}" if x_metric and y_metric else "Review EDA"
    return publication_figure(
        base,
        title=title,
        width=width,
        height=height,
        legend_outside=True,
        right_margin=275,
        xaxis_title=x_metric,
        yaxis_title=y_metric,
    )


def _jsonable(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def eda_table_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cols = [col["id"] for col in EDA_TABLE_COLUMNS if col["id"] in frame.columns]
    if "candidate_key" in frame.columns:
        cols.append("candidate_key")
    rows: list[dict[str, Any]] = []
    for record in frame.loc[:, cols].to_dict("records"):
        row = {key: _jsonable(value) for key, value in record.items()}
        row_id = str(row.get("candidate_id") or row.get("candidate_key") or "").strip()
        if row_id:
            row["id"] = row_id
        rows.append(row)
    return rows


def candidate_ids_from_eda_table_context(
    active_cell: object,
    viewport_rows: object,
    virtual_rows: object,
    table_rows: object,
    page_current: object = None,
    page_size: object = None,
) -> list[str]:
    """Resolve candidate IDs from a Dash DataTable active cell across paging modes."""
    if not isinstance(active_cell, dict):
        return []

    candidate_ids: list[str] = []

    def add(value: object) -> None:
        text = str(value or "").strip()
        if text and text not in candidate_ids:
            candidate_ids.append(text)

    def add_row(row: object) -> None:
        if not isinstance(row, dict):
            return
        for value in (row.get("candidate_key"), row.get("candidate_id"), row.get("id")):
            add(value)

    add(active_cell.get("row_id"))
    if candidate_ids:
        return candidate_ids
    try:
        row_idx = int(active_cell.get("row"))
    except (TypeError, ValueError):
        return candidate_ids

    for rows, idx in (
        (viewport_rows, row_idx),
        (virtual_rows, _absolute_table_row_index(row_idx, page_current, page_size)),
        (virtual_rows, row_idx),
        (table_rows, row_idx),
    ):
        if idx is None or not isinstance(rows, list):
            continue
        try:
            before = len(candidate_ids)
            add_row(rows[int(idx)])
        except Exception:
            continue
        if len(candidate_ids) > before:
            return candidate_ids

    return candidate_ids


def _trace_customdata_value(figure: object, curve_number: object, point_number: object) -> object:
    try:
        curve_idx = int(curve_number)
        point_idx = int(point_number)
    except (TypeError, ValueError):
        return None
    if curve_idx < 0 or point_idx < 0:
        return None

    traces = []
    if isinstance(figure, go.Figure):
        traces = list(figure.data)
    elif isinstance(figure, dict):
        raw_traces = figure.get("data")
        traces = raw_traces if isinstance(raw_traces, list) else []
    if curve_idx >= len(traces):
        return None

    trace = traces[curve_idx]
    customdata = trace.get("customdata") if isinstance(trace, dict) else getattr(trace, "customdata", None)
    try:
        value = customdata[point_idx]
    except Exception:
        return None
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def candidate_ids_from_plotly_selection(selection_data: object, figure: object | None = None) -> list[str]:
    """Resolve candidate IDs from Plotly selectedData point payloads."""
    if not isinstance(selection_data, dict):
        return []

    candidate_ids: list[str] = []

    def add(value: object) -> None:
        text = str(value or "").strip()
        if text and text not in candidate_ids:
            candidate_ids.append(text)

    def add_from_text(value: object) -> None:
        text = str(value or "")
        for part in text.replace("<br />", "<br>").split("<br>"):
            label, sep, raw_candidate_id = part.partition(":")
            if sep and label.strip() == "candidate_id":
                add(raw_candidate_id)
                return

    for point in selection_data.get("points") or []:
        if not isinstance(point, dict):
            continue
        custom = point.get("customdata")
        if isinstance(custom, (list, tuple)) and custom:
            add(custom[0])
        elif custom is not None:
            add(custom)
        else:
            point_number = point.get("pointNumber", point.get("pointIndex"))
            trace_value = _trace_customdata_value(figure, point.get("curveNumber"), point_number)
            if trace_value is not None:
                add(trace_value)
            else:
                add_from_text(point.get("text") or point.get("hovertext"))

    return candidate_ids


def _absolute_table_row_index(row_idx: int, page_current: object, page_size: object) -> int | None:
    try:
        page = int(page_current or 0)
        size = int(page_size or 0)
    except (TypeError, ValueError):
        return None
    if page < 0 or size <= 0:
        return None
    return page * size + int(row_idx)


def candidate_index_in_queue(queue_data: object, candidate_id: object) -> int | None:
    if not isinstance(queue_data, dict):
        return None
    target = str(candidate_id or "").strip()
    if not target:
        return None
    candidate_ids = [str(value) for value in (queue_data.get("candidate_ids") or [])]
    try:
        return candidate_ids.index(target)
    except ValueError:
        return None


def selected_candidate_from_queue(queue_data: object, index: object) -> str:
    if not isinstance(queue_data, dict):
        return ""
    candidate_ids = [str(value) for value in (queue_data.get("candidate_ids") or [])]
    try:
        idx = int(index or 0)
    except (TypeError, ValueError):
        idx = 0
    if idx < 0 or idx >= len(candidate_ids):
        return ""
    return candidate_ids[idx]


def selected_candidate_row_style(selected_candidate_id: object, *, theme: str | None = None) -> list[dict[str, Any]]:
    selected_id = str(selected_candidate_id or "").strip()
    if not selected_id:
        return []
    colors = _theme_palette(theme)
    escaped_id = selected_id.replace("\\", "\\\\").replace('"', '\\"')
    return [
        {
            "if": {"filter_query": f'{{candidate_id}} = "{escaped_id}"'},
            "backgroundColor": "rgba(10, 167, 255, 0.22)" if str(theme or "").lower() != "white" else "#dbeeff",
            "color": colors["text"],
            "fontWeight": "700",
            "boxShadow": "inset 3px 0 0 #0aa7ff",
        }
    ]


def selected_row_style(rows: list[dict[str, Any]], selected_candidate_id: object, *, theme: str | None = None) -> list[dict[str, Any]]:
    selected_id = str(selected_candidate_id or "").strip()
    if not selected_id:
        return []
    for row in rows:
        if str(row.get("candidate_id") or "") == selected_id:
            return selected_candidate_row_style(selected_id, theme=theme)
    return []
