"""Matplotlib PDF backend for unified review light-curve plot specifications."""

from __future__ import annotations

from io import BytesIO

import numpy as np

from malca.plotting.lightcurve_publication import FIG_TWO_COL_WIDTH, PUBLICATION_STYLE, _load_matplotlib, finalize_publication_figure, style_publication_axis
from malca.review.interactive_plot import DIP_EVENT_COLOR, JUMP_EVENT_COLOR, PHASE_TIME_COLORSCALE
from malca.review.lightcurve_assembly import MARKER_MAP, PlotTrace, ReviewLightCurvePlotSpec

_HEADER_BOX = {
    "boxstyle": "square,pad=0.28",
    "facecolor": "white",
    "edgecolor": "0.15",
    "linewidth": 0.85,
    "alpha": 1.0,
}

_RAW_MARKER_CAP_PT = 3.0
_SECONDARY_MARKER_CAP_PT = 2.4
_MARKER_EDGE_WIDTH = 0.15
_MARKER_ERRORBAR_WIDTH = 0.3


def _mpl_marker(plotly_marker: str | None) -> str:
    if not plotly_marker:
        return "o"
    return MARKER_MAP.get(plotly_marker, {}).get("mpl", "o")


def _adaptive_pdf_marker_size(panel_id: str, requested_size: float, n_points: int) -> float:
    """Return a publication PDF marker size that stays readable for dense traces."""
    cap = _RAW_MARKER_CAP_PT if str(panel_id) == "raw" else _SECONDARY_MARKER_CAP_PT
    size = min(float(requested_size), cap)
    if n_points >= 800:
        size = min(size, 1.6)
    elif n_points >= 400:
        size = min(size, 1.9)
    elif n_points >= 200:
        size = min(size, 2.2)
    return max(size, 0.8)


def _phase_time_colormap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "malca_phase_time_delta_m",
        [item[1] for item in PHASE_TIME_COLORSCALE],
    )


def _axis_label_for_offset(jd_offset: float) -> str:
    if abs(float(jd_offset) - round(float(jd_offset))) < 1e-6:
        return rf"$\mathrm{{JD}} - {int(round(float(jd_offset)))}\ [\mathrm{{d}}]$"
    return rf"$\mathrm{{JD}} - {float(jd_offset):.1f}\ [\mathrm{{d}}]$"


def _style_lightcurve_axis(ax) -> None:
    style_publication_axis(ax, top=True, right=True)
    ax.grid(True, which="major", linewidth=0.42, alpha=0.24)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def _draw_header_boxes(ax, *, left: str | None, right: str | None) -> None:
    """Draw left/right coordinate boxes inside the top plot corners."""
    if left:
        ax.text(
            0.012,
            0.965,
            left,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            color="0.12",
            bbox=_HEADER_BOX,
            clip_on=False,
            zorder=30,
        )
    if right:
        ax.text(
            0.988,
            0.965,
            right,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            color="0.12",
            bbox=_HEADER_BOX,
            clip_on=False,
            zorder=30,
        )


def _set_robust_limits(ax, values: list[np.ndarray], *, inverted: bool, pad_fraction: float = 0.05) -> None:
    finite = np.concatenate([np.asarray(v, dtype=float).reshape(-1) for v in values if np.asarray(v).size])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        pad = max(0.1, abs(lo) * pad_fraction)
        lo -= pad
        hi += pad
    else:
        pad = max(0.03, (hi - lo) * pad_fraction)
        lo -= pad
        hi += pad
    ax.set_ylim((hi, lo) if inverted else (lo, hi))


def _attach_raw_residual_axes(ax_by_panel: dict[str, object], panels: tuple) -> None:
    """Remove vertical whitespace between adjacent raw and residual axes."""
    for upper, lower in zip(panels, panels[1:]):
        if upper.panel_id != "raw" or lower.panel_id != "resid":
            continue
        upper_ax = ax_by_panel.get(upper.panel_id)
        lower_ax = ax_by_panel.get(lower.panel_id)
        if upper_ax is None or lower_ax is None:
            continue
        upper_pos = upper_ax.get_position()
        lower_pos = lower_ax.get_position()
        if upper_pos.y0 <= lower_pos.y1:
            continue
        lower_ax.set_position(
            [upper_pos.x0, lower_pos.y0, upper_pos.width, lower_pos.height]
        )
        upper_ax.set_position(
            [upper_pos.x0, lower_pos.y1, upper_pos.width, upper_pos.y1 - lower_pos.y1]
        )


def _plot_trace(ax, trace: PlotTrace, *, label: str | None = None, marker_size: float | None = None) -> None:
    x = np.asarray(trace.x, dtype=float)
    y = np.asarray(trace.y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        return
    x = x[mask]
    y = y[mask]
    marker = _mpl_marker(trace.marker)
    size = _adaptive_pdf_marker_size(
        trace.panel_id,
        float(marker_size if marker_size is not None else trace.marker_size),
        int(x.size),
    )
    yerr = None
    if trace.yerr is not None:
        err = np.asarray(trace.yerr, dtype=float)[mask]
        if np.isfinite(err).any():
            yerr = err
    color = trace.color or "0.2"
    if yerr is not None:
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=marker,
            linestyle="none",
            markersize=size,
            markerfacecolor=color,
            markeredgecolor="0.12",
            markeredgewidth=_MARKER_EDGE_WIDTH,
            ecolor=color,
            elinewidth=_MARKER_ERRORBAR_WIDTH,
            capsize=0.0,
            alpha=trace.alpha,
            label=label if label is not None else trace.label,
            zorder=4,
        )
    else:
        ax.scatter(
            x,
            y,
            s=size**2,
            marker=marker,
            facecolors=color,
            edgecolors="0.12",
            linewidths=_MARKER_EDGE_WIDTH,
            alpha=trace.alpha,
            label=label if label is not None else trace.label,
            zorder=4,
        )


def render_review_lightcurve_pdf(spec: ReviewLightCurvePlotSpec) -> bytes:
    """Render a review light-curve spec as publication PDF bytes."""
    if spec.status != "ok":
        raise ValueError(spec.status_message or spec.status)

    if not spec.panels:
        raise ValueError("No light-curve panels selected for export.")

    n_rows = len(spec.panels)
    if n_rows == 1:
        height_ratios = [1.0]
    elif n_rows == 2:
        height_ratios = [1.45, 1.0]
    else:
        lower = 0.25
        height_ratios = [1.0 - 2.0 * lower, lower, lower]

    plt, _auto_minor = _load_matplotlib()
    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(FIG_TWO_COL_WIDTH, 1.70 + 1.25 * n_rows),
            sharex=False,
            gridspec_kw={"height_ratios": height_ratios[:n_rows]},
        )
        axes_list = list(np.atleast_1d(axes))
        ax_by_panel = {panel.panel_id: axes_list[idx] for idx, panel in enumerate(spec.panels)}

        legend_handles: list = []
        legend_labels: list[str] = []
        value_buckets: dict[str, list[np.ndarray]] = {p.panel_id: [] for p in spec.panels}

        for trace in spec.traces:
            ax = ax_by_panel.get(trace.panel_id)
            if ax is None:
                continue
            if trace.kind == "phase_scatter_cmap":
                cmap = _phase_time_colormap()
                x = np.asarray(trace.x, dtype=float)
                y = np.asarray(trace.y, dtype=float)
                c = np.asarray(trace.cmap_values, dtype=float)
                valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
                if not valid.any():
                    continue
                size = _adaptive_pdf_marker_size(trace.panel_id, trace.marker_size, int(np.count_nonzero(valid)))
                scatter_kwargs = {
                    "s": size**2,
                    "marker": _mpl_marker(trace.marker),
                    "c": c[valid],
                    "cmap": cmap,
                    "edgecolors": "0.12",
                    "linewidths": _MARKER_EDGE_WIDTH,
                    "alpha": 0.84,
                    "zorder": 4,
                }
                if trace.cmap_vmin is not None and trace.cmap_vmax is not None:
                    scatter_kwargs["vmin"] = trace.cmap_vmin
                    scatter_kwargs["vmax"] = trace.cmap_vmax
                sc = ax.scatter(x[valid], y[valid], **scatter_kwargs)
                value_buckets[trace.panel_id].append(c[valid])
                if trace.panel_id == "phase" and not any(isinstance(h, type(sc)) for h in legend_handles):
                    cbar = fig.colorbar(sc, ax=ax, pad=0.012, aspect=14)
                    cbar.set_label(r"$\Delta m$", fontsize=7.0)
                    cbar.ax.tick_params(labelsize=6.5)
            else:
                label = trace.label if trace.showlegend else None
                _plot_trace(ax, trace, label=label)
                value_buckets[trace.panel_id].append(np.asarray(trace.y, dtype=float))

        for trace in spec.baselines:
            ax = ax_by_panel.get(trace.panel_id)
            if ax is None:
                continue
            x = np.asarray(trace.x, dtype=float)
            y = np.asarray(trace.y, dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if not valid.any():
                continue
            ax.plot(
                x[valid],
                y[valid],
                color=trace.color or "0.3",
                linewidth=float(trace.line_width) * 0.6,
                alpha=float(trace.alpha),
                zorder=3,
            )
            value_buckets[trace.panel_id].append(y[valid])

        for event in spec.events:
            ax = ax_by_panel.get(event.panel_id)
            if ax is None:
                continue
            color = event.color or (DIP_EVENT_COLOR if event.kind == "dip" else JUMP_EVENT_COLOR)
            if str(color).startswith("rgba("):
                color = DIP_EVENT_COLOR if event.kind == "dip" else JUMP_EVENT_COLOR
            alpha = 0.22 + 0.28 * float(event.confidence)
            x0 = float(event.x0)
            half_width = float(event.half_width or 0.0)
            if event.show_span and half_width > 0:
                ax.axvspan(x0 - half_width, x0 + half_width, color=color, alpha=0.08, linewidth=0, zorder=1)
            ax.axvline(x0, color=color, linestyle="--", linewidth=0.9, alpha=alpha + 0.25, zorder=2)

        for hline in spec.hlines:
            ax = ax_by_panel.get(hline.panel_id)
            if ax is None:
                continue
            ax.axhline(hline.y, color=hline.color or "0.45", linestyle=":", linewidth=0.65, alpha=0.75, zorder=1)

        for vline in spec.vlines:
            ax = ax_by_panel.get(vline.panel_id)
            if ax is None:
                continue
            ax.axvline(vline.x, color=vline.color or "0.55", linestyle=":", linewidth=0.65, alpha=0.7, zorder=1)

        for ann in spec.annotations:
            if ann.text == "◆" or ann.text.startswith("Dip thr") or ann.text.startswith("Jump thr"):
                continue
            ax = ax_by_panel.get(ann.panel_id or "")
            if ax is None:
                continue
            if ann.xref == "paper" and ann.yref == "paper":
                ax.text(
                    ann.x,
                    ann.y,
                    ann.text,
                    transform=ax.transAxes,
                    ha=ann.xanchor or "left",
                    va=ann.yanchor or "top",
                    fontsize=float(ann.font_size or 7.0),
                    color=ann.color or "0.20",
                )
            elif ann.xref == "domain" and ann.yref == "domain":
                ax.text(
                    ann.x,
                    ann.y,
                    ann.text,
                    transform=ax.transAxes,
                    ha=ann.xanchor or "left",
                    va=ann.yanchor or "top",
                    fontsize=float(ann.font_size or 11.0),
                    color=ann.color or "0.20",
                    bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9, "linewidth": 0.6},
                )

        time_panel = None
        for panel in spec.panels:
            ax = ax_by_panel[panel.panel_id]
            _style_lightcurve_axis(ax)
            if panel.y_label:
                ax.set_ylabel(panel.y_label)
            if panel.x_label:
                ax.set_xlabel(panel.x_label)
            if panel.x_range is not None:
                ax.set_xlim(panel.x_range)
            if panel.y_range is not None:
                ax.set_ylim(panel.y_range)
            elif panel.kind in {"raw", "resid", "phase"} and value_buckets.get(panel.panel_id):
                _set_robust_limits(
                    ax,
                    value_buckets[panel.panel_id],
                    inverted=bool(panel.invert_y),
                )
            if panel.kind in {"raw", "resid", "external"}:
                time_panel = panel.panel_id

        if time_panel is not None:
            ax_by_panel[time_panel].set_xlabel(_axis_label_for_offset(spec.jd_offset))
            for panel in spec.panels:
                if panel.kind in {"raw", "resid", "external"} and panel.panel_id != time_panel:
                    ax_by_panel[panel.panel_id].tick_params(axis="x", labelbottom=False, bottom=False)

        raw_ax = ax_by_panel.get("raw")
        if raw_ax is not None:
            handles, labels = raw_ax.get_legend_handles_labels()
            seen: set[str] = set()
            for handle, label in zip(handles, labels):
                if not label or label in seen:
                    continue
                seen.add(label)
                legend_handles.append(handle)
                legend_labels.append(label)
            if legend_handles:
                ncol = 1 if len(legend_labels) <= 9 else 2
                raw_ax.legend(
                    legend_handles,
                    legend_labels,
                    loc="upper left",
                    bbox_to_anchor=(1.01, 1.0),
                    frameon=False,
                    borderaxespad=0.0,
                    fontsize=6.7,
                    ncol=ncol,
                    columnspacing=0.7,
                    handletextpad=0.35,
                )

        top_panel_id = spec.panels[0].panel_id if spec.panels else None
        top_ax = ax_by_panel.get(top_panel_id) if top_panel_id else None
        if top_ax is not None and (spec.header_left or spec.header_right):
            _draw_header_boxes(top_ax, left=spec.header_left, right=spec.header_right)

        if spec.title and not (spec.header_left or spec.header_right):
            fig.text(0.09, 0.985, spec.title, ha="left", va="top", fontsize=8.0, color="0.15")
        if legend_handles:
            right = 0.86 if len(legend_labels) <= 6 else 0.82
        else:
            right = 0.985
        top = 0.97 if (spec.header_left or spec.header_right) else 0.98
        finalize_publication_figure(fig, h_pad=0.18, rect=(0.075, 0.055, right, top))
        _attach_raw_residual_axes(ax_by_panel, spec.panels)

        buf = BytesIO()
        try:
            fig.savefig(buf, format="pdf", dpi=300, metadata={"Creator": "MALCA"}, bbox_inches=None)
            return buf.getvalue()
        finally:
            plt.close(fig)
