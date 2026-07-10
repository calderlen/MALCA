"""Plotly backend for unified review light-curve plot specifications."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from malca.plotting.lightcurve_publication import PUBLICATION_PLOTLY_FONT
from malca.review.interactive_plot import (
    PHASE_TIME_COLORSCALE,
    _event_annotation_y,
    _status_figure,
    _subplot_axis_ref,
    _subplot_domain_ref,
    _theme_palette,
)
from malca.review.lightcurve_assembly import (
    PlotAnnotation,
    PlotEventOverlay,
    PlotTrace,
    ReviewLightCurvePlotSpec,
)


def _panel_row_map(panels: tuple) -> dict[str, int]:
    return {panel.panel_id: idx + 1 for idx, panel in enumerate(panels)}


def _clip_domain_value(value: float) -> float:
    """Keep Plotly domain endpoints valid after harmless float roundoff."""
    return float(np.clip(value, 0.0, 1.0))


def _panel_vertical_domains(panels: tuple, *, default_gap: float = 0.05) -> list[tuple[float, float]]:
    gaps = [
        0.0 if upper.panel_id == "raw" and lower.panel_id == "resid" else default_gap
        for upper, lower in zip(panels, panels[1:])
    ]
    total_weight = float(sum(panel.height_ratio for panel in panels)) or 1.0
    available = max(0.1, 1.0 - float(sum(gaps)))
    heights = [available * (panel.height_ratio / total_weight) for panel in panels]
    domains: list[tuple[float, float]] = []
    top = 1.0
    for idx, height in enumerate(heights):
        bottom = top - height
        domains.append((_clip_domain_value(bottom), _clip_domain_value(top)))
        if idx < len(gaps):
            top = bottom - gaps[idx]
    return domains


def _resolve_axis_ref(panel_id: str, row_map: dict[str, int], axis: str, xref: str) -> str:
    row = row_map.get(panel_id, 1)
    if xref == "domain":
        return _subplot_domain_ref(row, axis)  # type: ignore[arg-type]
    if xref == "axis":
        return _subplot_axis_ref(row, axis)  # type: ignore[arg-type]
    return xref


def _add_trace(fig: go.Figure, trace: PlotTrace, row: int, colors: dict[str, str], *, colorbar_shown: bool) -> bool:
    if trace.kind == "line":
        fig.add_trace(
            go.Scatter(
                x=trace.x,
                y=trace.y,
                mode="lines",
                showlegend=trace.showlegend,
                name=trace.label or "",
                line={"width": trace.line_width, "color": trace.color or colors["guide_line"]},
                opacity=trace.alpha,
                customdata=trace.customdata,
                hovertemplate=trace.hovertemplate,
            ),
            row=row,
            col=1,
        )
        return colorbar_shown

    if trace.kind == "phase_scatter_cmap":
        marker: dict = {
            "size": trace.marker_size,
            "symbol": trace.marker or "circle",
            "color": trace.cmap_values,
            "colorscale": PHASE_TIME_COLORSCALE,
            "line": {"width": 0.45, "color": colors["marker_line"]},
            "showscale": not colorbar_shown,
            "colorbar": {"title": "Δm", "len": 0.34},
        }
        if trace.cmap_vmin is not None and trace.cmap_vmax is not None:
            marker["cmin"] = trace.cmap_vmin
            marker["cmax"] = trace.cmap_vmax
        fig.add_trace(
            go.Scatter(
                x=trace.x,
                y=trace.y,
                mode="markers",
                name=trace.label or "",
                showlegend=trace.showlegend,
                marker=marker,
                customdata=trace.customdata,
                hovertemplate=trace.hovertemplate,
            ),
            row=row,
            col=1,
        )
        return True

    marker_style = {
        "size": trace.marker_size,
        "symbol": trace.marker or "circle",
        "color": trace.color,
        "line": {"width": 0.8 if trace.panel_id == "raw" else 0.7, "color": colors["marker_line"]},
    }
    error_y = None
    if trace.yerr is not None and np.isfinite(trace.yerr).any():
        error_y = {
            "type": "data",
            "array": trace.yerr,
            "visible": True,
            "thickness": 1,
            "width": 0,
            "color": trace.color,
        }
    fig.add_trace(
        go.Scatter(
            x=trace.x,
            y=trace.y,
            mode="markers",
            name=trace.label or "",
            showlegend=trace.showlegend,
            marker=marker_style,
            error_y=error_y,
            legendgroup=trace.legendgroup,
            customdata=trace.customdata,
            hovertemplate=trace.hovertemplate,
        ),
        row=row,
        col=1,
    )
    return colorbar_shown


def _raw_event_geometry(spec: ReviewLightCurvePlotSpec) -> tuple[np.ndarray, np.ndarray, float]:
    raw_traces = [t for t in spec.traces if t.panel_id == "raw" and t.kind == "scatter" and t.showlegend]
    if not raw_traces:
        raw_traces = [t for t in spec.traces if t.panel_id == "raw" and t.kind == "scatter"]
    x_values = np.concatenate([np.asarray(t.x, dtype=float) for t in raw_traces]) if raw_traces else np.array([])
    y_values = np.concatenate([np.asarray(t.y, dtype=float) for t in raw_traces]) if raw_traces else np.array([])
    finite = np.isfinite(y_values)
    if finite.any():
        span = float(np.nanmax(y_values[finite]) - np.nanmin(y_values[finite]))
    else:
        span = 1.0
    pad = span * 0.10
    if not np.isfinite(pad) or pad <= 0:
        pad = 0.05 * max(1.0, abs(float(y_values[finite][0])) if finite.any() else 1.0)
    return x_values, y_values, pad


def _render_event(
    fig: go.Figure,
    event: PlotEventOverlay,
    row: int,
    colors: dict[str, str],
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    pad: float,
    is_flux: bool,
    annotations: tuple[PlotAnnotation, ...],
) -> None:
    raw_xref = _subplot_axis_ref(row, "x")
    raw_yref = _subplot_axis_ref(row, "y")
    raw_y_domain_ref = _subplot_domain_ref(row, "y")
    color = event.color or colors["guide_line"]
    marker_y, label_y = _event_annotation_y(
        x_values,
        y_values,
        float(event.x0),
        float(event.half_width),
        kind=event.kind,
        is_flux=is_flux,
        pad=pad,
    )
    label_yanchor = "top" if event.kind == "dip" else "bottom"
    fig.add_shape(
        type="line",
        x0=event.x0,
        x1=event.x0,
        y0=0.0,
        y1=1.0,
        xref=raw_xref,
        yref=raw_y_domain_ref,
        line={"color": color, "dash": "dash", "width": 1.8},
    )
    if event.show_span and float(event.half_width) > 0:
        half_width = float(event.half_width)
        fig.add_shape(
            type="rect",
            x0=event.x0 - half_width,
            x1=event.x0 + half_width,
            y0=0.0,
            y1=1.0,
            xref=raw_xref,
            yref=raw_y_domain_ref,
            fillcolor=color,
            opacity=0.11,
            line={"width": 0},
            layer="below",
        )

    for ann in annotations:
        if ann.panel_id != event.panel_id:
            continue
        if abs(float(ann.x) - float(event.x0)) > 1e-6:
            continue
        if ann.text == "◆":
            fig.add_annotation(
                x=event.x0,
                y=marker_y,
                xref=raw_xref,
                yref=raw_yref,
                text=ann.text,
                showarrow=False,
                font={"size": int(ann.font_size or 18), "color": ann.color or color},
                hovertext=ann.hovertext,
            )
        elif ann.text.startswith(str(event.kind).title()):
            fig.add_annotation(
                x=event.x0,
                y=label_y,
                xref=raw_xref,
                yref=raw_yref,
                text=ann.text,
                showarrow=False,
                font={"size": int(ann.font_size or 9), "color": colors["annotation"]},
                yanchor=label_yanchor,
                bgcolor=colors["paper_bg"],
                opacity=ann.opacity if ann.opacity is not None else 0.92,
            )


def _render_annotation(fig: go.Figure, ann: PlotAnnotation, row_map: dict[str, int], colors: dict[str, str]) -> None:
    if ann.text == "◆" or ann.text.startswith("Dip thr") or ann.text.startswith("Jump thr"):
        return
    row = row_map.get(ann.panel_id or "", 1)
    xref = _resolve_axis_ref(ann.panel_id or "", row_map, "x", ann.xref)
    yref = _resolve_axis_ref(ann.panel_id or "", row_map, "y", ann.yref)
    font: dict = {"size": int(ann.font_size or 11), "color": ann.color or colors["text"]}
    fig.add_annotation(
        x=ann.x,
        y=ann.y,
        xref=xref,
        yref=yref,
        text=ann.text,
        showarrow=ann.showarrow,
        font=font,
        xanchor=ann.xanchor,
        yanchor=ann.yanchor,
        bgcolor=ann.bgcolor or colors.get("paper_bg"),
        bordercolor=ann.bordercolor or colors.get("grid"),
        borderwidth=ann.borderwidth,
        opacity=ann.opacity,
        hovertext=ann.hovertext,
    )


def render_review_lightcurve_plotly(
    spec: ReviewLightCurvePlotSpec,
    *,
    theme: str = "black",
    uirevision_key: str = "",
) -> dict:
    """Render a review light-curve spec as a Plotly figure bundle."""
    if spec.status != "ok":
        message = spec.status_message or spec.status
        return {
            "figure": _status_figure(message, theme=theme),
            "camera_options": list(spec.camera_options),
            "camera_values": list(spec.camera_values),
            "stat_rows": list(spec.stat_rows),
            "status": spec.status,
            "status_message": spec.status_message,
            "camera_diagnostics": spec.camera_diagnostics,
            "warnings": list(spec.warnings),
        }

    if not spec.panels:
        return {
            "figure": _status_figure("No panels selected. Enable an LC source, Residuals, or Phase-fold.", theme=theme),
            "camera_options": list(spec.camera_options),
            "camera_values": list(spec.camera_values),
            "stat_rows": list(spec.stat_rows),
            "status": spec.status,
            "status_message": spec.status_message,
            "camera_diagnostics": spec.camera_diagnostics,
            "warnings": list(spec.warnings),
        }

    colors = _theme_palette(theme)
    row_map = _panel_row_map(spec.panels)
    total_weight = float(sum(p.height_ratio for p in spec.panels)) or 1.0
    row_heights = [p.height_ratio / total_weight for p in spec.panels]
    fig = make_subplots(
        rows=len(spec.panels),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.0,
        row_heights=row_heights,
    )
    for row, domain in enumerate(_panel_vertical_domains(spec.panels), start=1):
        fig.update_yaxes(domain=list(domain), row=row, col=1)

    colorbar_shown = False
    for trace in spec.traces:
        row = row_map[trace.panel_id]
        colorbar_shown = _add_trace(fig, trace, row, colors, colorbar_shown=colorbar_shown)
    for trace in spec.baselines:
        row = row_map[trace.panel_id]
        colorbar_shown = _add_trace(fig, trace, row, colors, colorbar_shown=colorbar_shown)

    if spec.events:
        x_values, y_values, pad = _raw_event_geometry(spec)
        for event in spec.events:
            _render_event(
                fig,
                event,
                row_map[event.panel_id],
                colors,
                x_values=x_values,
                y_values=y_values,
                pad=pad,
                is_flux=spec.is_flux,
                annotations=spec.annotations,
            )

    for hline in spec.hlines:
        fig.add_hline(
            y=hline.y,
            line_color=hline.color or colors["guide_line"],
            line_dash=hline.dash,
            line_width=hline.line_width,
            row=row_map[hline.panel_id],
            col=1,
        )
    for vline in spec.vlines:
        fig.add_vline(
            x=vline.x,
            line_color=vline.color or colors["guide_line"],
            line_dash=vline.dash,
            line_width=vline.line_width,
            row=row_map[vline.panel_id],
            col=1,
        )

    for ann in spec.annotations:
        _render_annotation(fig, ann, row_map, colors)

    for panel in spec.panels:
        row = row_map[panel.panel_id]
        yaxis_kwargs: dict = {}
        if panel.y_label:
            yaxis_kwargs["title_text"] = panel.y_label
        if panel.y_range is not None:
            yaxis_kwargs["range"] = list(panel.y_range)
        elif panel.y_autorange == "reversed":
            yaxis_kwargs["autorange"] = "reversed"
        elif panel.y_autorange is True:
            yaxis_kwargs["autorange"] = True
        elif panel.y_autorange is False:
            yaxis_kwargs["autorange"] = False
        if yaxis_kwargs:
            fig.update_yaxes(row=row, col=1, **yaxis_kwargs)
        xaxis_kwargs: dict = {}
        if panel.x_label:
            xaxis_kwargs["title_text"] = panel.x_label
        elif panel.kind in {"raw", "resid", "external"}:
            xaxis_kwargs["showticklabels"] = False
            xaxis_kwargs["ticks"] = ""
        if panel.x_range is not None:
            xaxis_kwargs["range"] = list(panel.x_range)
        if xaxis_kwargs:
            fig.update_xaxes(row=row, col=1, **xaxis_kwargs)

    if spec.jd_match_panel_id and spec.jd_match_panel_id in row_map:
        anchor_axis = _subplot_axis_ref(row_map[spec.jd_match_panel_id], "x")
        for panel in spec.panels:
            if panel.kind in {"raw", "resid", "external"} and panel.panel_id != spec.jd_match_panel_id:
                fig.update_xaxes(matches=anchor_axis, row=row_map[panel.panel_id], col=1)

    fig.update_layout(
        title=spec.title,
        title_font={"size": 14, "color": colors["title"]},
        paper_bgcolor=colors["paper_bg"],
        plot_bgcolor=colors["plot_bg"],
        margin={"l": 55, "r": 20, "t": 68, "b": 44},
        font={"color": colors["text"], "family": PUBLICATION_PLOTLY_FONT, "size": 11},
        hovermode="closest",
        legend={
            "bgcolor": colors["legend_bg"],
            "bordercolor": colors["legend_border"],
            "borderwidth": 1,
            "font": {"size": 10},
        },
        height=None,
        uirevision=uirevision_key,
    )
    fig.update_xaxes(showgrid=True, gridcolor=colors["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=colors["grid"], zeroline=False)

    return {
        "figure": fig,
        "camera_options": list(spec.camera_options),
        "camera_values": list(spec.camera_values),
        "stat_rows": list(spec.stat_rows),
        "status": spec.status,
        "status_message": spec.status_message,
        "camera_diagnostics": spec.camera_diagnostics,
        "warnings": list(spec.warnings),
    }
