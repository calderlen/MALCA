"""Publication-quality Plotly export helpers for review figures."""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import plotly.graph_objects as go


PUBLICATION_FONT = "Helvetica, Arial, DejaVu Sans, sans-serif"
PLOTLY_IMAGE_EXPORT_BUTTON = "toImage"
MATPLOTLIB_PUBLICATION_STYLE = {
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.8,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 7.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def slugify_token(value: object, *, fallback: str = "figure") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    return text[:96] or fallback


def graph_config_without_image_export(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a Plotly graph config with the built-in image export disabled."""
    out = dict(config or {})
    raw_remove = out.get("modeBarButtonsToRemove") or []
    if isinstance(raw_remove, str):
        remove = [raw_remove]
    else:
        try:
            remove = list(raw_remove)
        except TypeError:
            remove = []
    if PLOTLY_IMAGE_EXPORT_BUTTON not in remove:
        remove.append(PLOTLY_IMAGE_EXPORT_BUTTON)
    out["modeBarButtonsToRemove"] = remove
    return out


def latex_axis_label(label: object) -> object:
    """Return a conservative MathJax label for common review axis labels."""
    text = str(label or "").strip()
    if not text:
        return label
    if text.startswith("$") and text.endswith("$"):
        return text
    lower = text.lower()
    exact = {
        "jd": r"$t\ [\mathrm{JD}]$",
        "time": r"$t$",
        "relative flux": r"$F/F_{\mathrm{GP}}$",
        "flux": r"$F$",
        "m [mag]": r"$m\ [\mathrm{mag}]$",
        "g [mag]": r"$G\ [\mathrm{mag}]$",
        "jd - 2458000": r"$\mathrm{JD}-2458000$",
        "lambda [a]": r"$\lambda\ [\mathring{\mathrm{A}}]$",
    }
    if lower in exact:
        return exact[lower]
    replacements = {
        "[d]": r"\ [\mathrm{d}]",
        "[mag]": r"\ [\mathrm{mag}]",
        "[arcsec]": r"\ [\mathrm{arcsec}]",
        "[mas/yr]": r"\ [\mathrm{mas\ yr^{-1}}]",
        "[km/s]": r"\ [\mathrm{km\ s^{-1}}]",
    }
    out = text
    for raw, repl in replacements.items():
        if raw in out:
            out = out.replace(raw, repl)
    if out != text:
        out = out.replace("_", r"\_")
        return f"${out}$"
    return label


def _vectorized_figure_dict(figure: go.Figure | dict[str, Any]) -> dict[str, Any]:
    raw = go.Figure(figure).to_dict()
    for trace in raw.get("data", []):
        if trace.get("type") == "scattergl":
            trace["type"] = "scatter"
    return raw


def publication_figure(
    figure: go.Figure | dict[str, Any],
    *,
    title: str | None = None,
    width: int = 1200,
    height: int = 820,
    legend_outside: bool = True,
    right_margin: int | None = None,
    top_margin: int = 92,
    bottom_margin: int = 78,
    left_margin: int = 90,
    xaxis_title: object | None = None,
    yaxis_title: object | None = None,
) -> go.Figure:
    """Return a white-background export figure with stable margins and labels."""
    fig = go.Figure(_vectorized_figure_dict(figure))
    if title is None:
        existing = fig.layout.title.text if fig.layout.title else None
        title = str(existing or "").strip() or None

    margin_r = right_margin if right_margin is not None else (275 if legend_outside else 55)
    legend = dict(
        bgcolor="rgba(255,255,255,0.96)",
        bordercolor="rgba(40,40,40,0.22)",
        borderwidth=1,
        font=dict(color="#111111", family=PUBLICATION_FONT, size=10),
    )
    if legend_outside:
        legend.update(
            orientation="v",
            x=1.02,
            xanchor="left",
            y=1.0,
            yanchor="top",
        )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111111", family=PUBLICATION_FONT, size=12),
        title=dict(
            text=title or "",
            x=0.5,
            xanchor="center",
            y=0.975,
            yanchor="top",
            font=dict(color="#111111", family=PUBLICATION_FONT, size=16),
        ),
        margin=dict(t=top_margin, l=left_margin, r=margin_r, b=bottom_margin),
        legend=legend,
        autosize=False,
        width=width,
        height=height,
    )

    fig.update_xaxes(
        color="#111111",
        title_font_color="#111111",
        tickfont_color="#111111",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.14)",
        linecolor="rgba(0,0,0,0.45)",
        ticks="outside",
        zeroline=False,
    )
    fig.update_yaxes(
        color="#111111",
        title_font_color="#111111",
        tickfont_color="#111111",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.14)",
        linecolor="rgba(0,0,0,0.45)",
        ticks="outside",
        zeroline=False,
    )
    if xaxis_title is not None:
        fig.update_xaxes(title=latex_axis_label(xaxis_title))
    else:
        for axis in fig.select_xaxes():
            if axis.title and axis.title.text:
                axis.title.text = latex_axis_label(axis.title.text)
    if yaxis_title is not None:
        fig.update_yaxes(title=latex_axis_label(yaxis_title))
    else:
        for axis in fig.select_yaxes():
            if axis.title and axis.title.text:
                axis.title.text = latex_axis_label(axis.title.text)
    return fig


def _matplotlib_imports():
    cache_dir = Path(tempfile.gettempdir()) / "malca-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if "MPLCONFIGDIR" not in os.environ:
        default_config = Path.home() / ".config" / "matplotlib"
        if not os.access(default_config, os.W_OK):
            os.environ["MPLCONFIGDIR"] = str(cache_dir)
    if "XDG_CACHE_HOME" not in os.environ:
        default_cache = Path.home() / ".cache"
        if not os.access(default_cache, os.W_OK):
            os.environ["XDG_CACHE_HOME"] = str(cache_dir)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Circle, Rectangle

    return plt, to_rgba, Circle, Rectangle


def _clean_math_text(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<sub>(.*?)</sub>", r"_{\1}", text, flags=re.IGNORECASE)
    text = re.sub(r"<sup>(.*?)</sup>", r"^{\1}", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return text


def _layout_text(layout: dict[str, Any], *path: str) -> str:
    value: Any = layout
    for key in path:
        if not isinstance(value, dict):
            return ""
        value = value.get(key)
    if isinstance(value, dict):
        value = value.get("text")
    return _clean_math_text(value)


def _axis_layout(layout: dict[str, Any], axis_ref: str) -> dict[str, Any]:
    suffix = axis_ref[1:] if len(axis_ref) > 1 else ""
    key = f"{'xaxis' if axis_ref.startswith('x') else 'yaxis'}{suffix}"
    value = layout.get(key)
    return value if isinstance(value, dict) else {}


def _axis_domain(layout: dict[str, Any], axis_ref: str, fallback: tuple[float, float]) -> tuple[float, float]:
    raw = _axis_layout(layout, axis_ref).get("domain")
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            lo = float(raw[0])
            hi = float(raw[1])
            if 0.0 <= lo < hi <= 1.0:
                return lo, hi
        except Exception:
            pass
    return fallback


def _plotly_color(value: object, fallback: str = "#1f77b4") -> tuple[float, float, float, float]:
    _plt, to_rgba, _Circle, _Rectangle = _matplotlib_imports()
    text = str(value or "").strip()
    if not text:
        text = fallback
    match = re.match(r"rgba?\(([^)]+)\)", text, flags=re.IGNORECASE)
    if match:
        parts = [p.strip() for p in match.group(1).split(",")]
        try:
            r = float(parts[0]) / 255.0
            g = float(parts[1]) / 255.0
            b = float(parts[2]) / 255.0
            a = float(parts[3]) if len(parts) > 3 else 1.0
            return (r, g, b, max(0.0, min(1.0, a)))
        except Exception:
            pass
    try:
        return to_rgba(text)
    except Exception:
        return to_rgba(fallback)


def _trace_array(trace: dict[str, Any], key: str) -> list[float]:
    raw = trace.get(key)
    if raw is None:
        return []
    if isinstance(raw, dict) and "bdata" in raw:
        return []
    try:
        import numpy as np

        arr = np.asarray(raw, dtype=float).reshape(-1)
        return [float(v) for v in arr]
    except Exception:
        out: list[float] = []
        try:
            for value in raw:
                out.append(float(value))
        except Exception:
            return []
        return out


def _numeric_sequence(value: object) -> list[float]:
    if value is None or isinstance(value, str):
        return []
    if isinstance(value, dict) and "bdata" in value:
        return []
    try:
        import numpy as np

        arr = np.asarray(value, dtype=float).reshape(-1)
        return [float(v) for v in arr if np.isfinite(v)]
    except Exception:
        return []


def _same_length_xy(trace: dict[str, Any]) -> tuple[list[float], list[float]]:
    y = _trace_array(trace, "y")
    x = _trace_array(trace, "x")
    if not y:
        return [], []
    if not x:
        x = list(range(len(y)))
    n = min(len(x), len(y))
    return x[:n], y[:n]


def _line_style(line: dict[str, Any] | None) -> str:
    dash = str((line or {}).get("dash") or "solid").lower()
    return {
        "solid": "-",
        "dash": "--",
        "dot": ":",
        "dashdot": "-.",
        "longdash": "--",
    }.get(dash, "-")


def _marker_style(symbol: object) -> str:
    text = str(symbol or "circle").lower()
    if "star" in text:
        return "*"
    if "diamond" in text:
        return "D"
    if "square" in text:
        return "s"
    if "triangle-up" in text:
        return "^"
    if "triangle-down" in text:
        return "v"
    if "cross" in text:
        return "x"
    return "o"


def _axis_ref(trace: dict[str, Any], key: str) -> str:
    value = str(trace.get(key) or key[0]).strip()
    return value if value else key[0]


def _subplot_refs(data: list[dict[str, Any]], layout: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for trace in data:
        if trace.get("visible") in (False, "legendonly"):
            continue
        pair = (_axis_ref(trace, "xaxis"), _axis_ref(trace, "yaxis"))
        if pair not in refs:
            refs.append(pair)
    if not refs:
        return [("x", "y")]
    if not isinstance(layout, dict):
        return refs

    return sorted(
        refs,
        key=lambda ref: (
            -_axis_domain(layout, ref[1], (0.0, 1.0))[0],
            _axis_domain(layout, ref[0], (0.0, 1.0))[0],
        ),
    )


def _apply_axis_layout(ax, layout: dict[str, Any], xref: str, yref: str, *, show_xlabel: bool = True) -> None:
    xaxis = _axis_layout(layout, xref)
    yaxis = _axis_layout(layout, yref)
    xlabel = _layout_text(xaxis, "title") or _layout_text(xaxis, "title", "text")
    ylabel = _layout_text(yaxis, "title") or _layout_text(yaxis, "title", "text")
    if xlabel and show_xlabel:
        ax.set_xlabel(xlabel)
    elif not show_xlabel:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    if ylabel:
        ax.set_ylabel(ylabel)
    if str(xaxis.get("type") or "").lower() == "log":
        ax.set_xscale("log")
    if str(yaxis.get("type") or "").lower() == "log":
        ax.set_yscale("log")
    _apply_axis_range(ax, xaxis, axis="x")
    _apply_axis_range(ax, yaxis, axis="y")
    if yaxis.get("autorange") == "reversed":
        ax.invert_yaxis()


def _apply_axis_range(ax, axis_layout: dict[str, Any], *, axis: str) -> None:
    raw_range = axis_layout.get("range")
    if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
        return
    try:
        lo = float(raw_range[0])
        hi = float(raw_range[1])
    except Exception:
        return
    if str(axis_layout.get("type") or "").lower() == "log":
        lo = 10 ** lo
        hi = 10 ** hi
    if axis == "x":
        ax.set_xlim(lo, hi)
    else:
        ax.set_ylim(lo, hi)


def _draw_shapes(ax, layout: dict[str, Any], xref: str, yref: str) -> None:
    _plt, _to_rgba, Circle, Rectangle = _matplotlib_imports()
    shapes = layout.get("shapes") or []
    if not isinstance(shapes, list):
        return
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        sxref = str(shape.get("xref") or "x")
        syref = str(shape.get("yref") or "y")
        if sxref != xref or syref != yref:
            continue
        line = shape.get("line") if isinstance(shape.get("line"), dict) else {}
        color = _plotly_color(line.get("color"), "#555555")
        width = float(line.get("width") or 1.0)
        ls = _line_style(line)
        try:
            x0 = float(shape.get("x0"))
            x1 = float(shape.get("x1"))
            y0 = float(shape.get("y0"))
            y1 = float(shape.get("y1"))
        except Exception:
            continue
        shape_type = str(shape.get("type") or "line").lower()
        if shape_type == "line":
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=width, linestyle=ls, zorder=3)
        elif shape_type == "circle":
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            radius = max(abs(x1 - x0), abs(y1 - y0)) / 2.0
            ax.add_patch(Circle((cx, cy), radius=radius, fill=False, edgecolor=color, linewidth=width, linestyle=ls, zorder=3))
        elif shape_type == "rect":
            ax.add_patch(Rectangle((min(x0, x1), min(y0, y1)), abs(x1 - x0), abs(y1 - y0), fill=False, edgecolor=color, linewidth=width, linestyle=ls, zorder=3))


def _draw_trace(ax, trace: dict[str, Any], previous_xy: tuple[list[float], list[float]] | None) -> tuple[list[float], list[float]] | None:
    trace_type = str(trace.get("type") or "scatter").lower()
    name = str(trace.get("name") or "").strip()
    showlegend = trace.get("showlegend")
    label = name if name and showlegend is not False else None
    if trace_type in {"scatter", "scattergl"}:
        x, y = _same_length_xy(trace)
        if not x or not y:
            return previous_xy
        line = trace.get("line") if isinstance(trace.get("line"), dict) else {}
        marker = trace.get("marker") if isinstance(trace.get("marker"), dict) else {}
        line_color = _plotly_color(line.get("color") or marker.get("color"), "#1f77b4")
        marker_color = _plotly_color(marker.get("color") or line.get("color"), "#1f77b4")
        marker_color_values = _numeric_sequence(marker.get("color"))
        if len(marker_color_values) != len(x):
            marker_color_values = []
        mode = str(trace.get("mode") or "lines").lower()
        fill = str(trace.get("fill") or "").lower()
        if fill and previous_xy is not None and fill in {"tonexty", "tonextx"}:
            px, py = previous_xy
            n = min(len(x), len(y), len(px), len(py))
            if n > 1:
                fill_color = _plotly_color(trace.get("fillcolor") or line.get("color") or marker.get("color"), "#9ca3af")
                ax.fill_between(x[:n], py[:n], y[:n], color=fill_color, alpha=fill_color[3], linewidth=0.0, zorder=1)
                return (x, y)
        if "lines" in mode:
            ax.plot(
                x,
                y,
                color=line_color,
                linewidth=max(0.6, float(line.get("width") or 1.2)),
                linestyle=_line_style(line),
                label=label if "markers" not in mode else None,
                zorder=2,
            )
        if "markers" in mode:
            size = marker.get("size")
            try:
                marker_size = float(size[0] if isinstance(size, (list, tuple)) else size or 6.0)
            except Exception:
                marker_size = 6.0
            error_y = trace.get("error_y") if isinstance(trace.get("error_y"), dict) else {}
            yerr = None
            if error_y and error_y.get("visible", True):
                arr = _trace_array(error_y, "array")
                arrminus = _trace_array(error_y, "arrayminus")
                if arr and arrminus:
                    nerr = min(len(arr), len(arrminus), len(y))
                    yerr = [arrminus[:nerr], arr[:nerr]]
                    x = x[:nerr]
                    y = y[:nerr]
                elif arr:
                    nerr = min(len(arr), len(y))
                    yerr = arr[:nerr]
                    x = x[:nerr]
                    y = y[:nerr]
            if yerr is not None:
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt=_marker_style(marker.get("symbol")),
                    markersize=max(2.0, marker_size * 0.55),
                    markerfacecolor=marker_color,
                    markeredgecolor=_plotly_color((marker.get("line") or {}).get("color") if isinstance(marker.get("line"), dict) else None, "#111111"),
                    markeredgewidth=0.45,
                    ecolor=_plotly_color((error_y or {}).get("color"), "#555555"),
                    elinewidth=0.65,
                    capsize=1.8,
                    linestyle="none",
                    label=label,
                    zorder=4,
                )
            else:
                scatter_kwargs: dict[str, Any] = (
                    {"c": marker_color_values, "cmap": "viridis"}
                    if marker_color_values
                    else {"color": marker_color}
                )
                scatter = ax.scatter(
                    x,
                    y,
                    s=max(8.0, marker_size ** 2 * 0.7),
                    marker=_marker_style(marker.get("symbol")),
                    linewidths=0.35,
                    edgecolors=_plotly_color((marker.get("line") or {}).get("color") if isinstance(marker.get("line"), dict) else None, "#111111"),
                    label=label,
                    zorder=4,
                    rasterized=len(x) > 3000,
                    **scatter_kwargs,
                )
                if marker_color_values:
                    colorbar = marker.get("colorbar") if isinstance(marker.get("colorbar"), dict) else {}
                    colorbar_title = _layout_text(colorbar, "title") or _layout_text(colorbar, "title", "text")
                    cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.035)
                    if colorbar_title:
                        cbar.set_label(colorbar_title)
        return (x, y)
    if trace_type == "bar":
        x, y = _same_length_xy(trace)
        color = _plotly_color((trace.get("marker") or {}).get("color") if isinstance(trace.get("marker"), dict) else None, "#4c78a8")
        if x and y:
            ax.bar(x, y, color=color, label=label, zorder=2)
            return (x, y)
    if trace_type == "heatmap":
        x = _trace_array(trace, "x")
        y = _trace_array(trace, "y")
        z_raw = trace.get("z")
        try:
            import numpy as np

            z = np.asarray(z_raw, dtype=float)
        except Exception:
            z = None
        if z is not None and z.size > 0:
            extent = None
            if x and y:
                extent = [min(x), max(x), min(y), max(y)]
            image = ax.imshow(
                z,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="viridis",
                rasterized=True,
            )
            ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    return previous_xy


def _axes_from_plotly_domains(fig, refs: list[tuple[str, str]], layout: dict[str, Any], *, legend_outside: bool) -> dict[tuple[str, str], Any]:
    left = 0.085
    right = 0.82 if legend_outside else 0.965
    bottom = 0.085
    top = 0.965
    width = right - left
    height = top - bottom
    axes: dict[tuple[str, str], Any] = {}
    for ref in refs:
        x0, x1 = _axis_domain(layout, ref[0], (0.0, 1.0))
        y0, y1 = _axis_domain(layout, ref[1], (0.0, 1.0))
        bounds = [
            left + x0 * width,
            bottom + y0 * height,
            max(0.05, (x1 - x0) * width),
            max(0.05, (y1 - y0) * height),
        ]
        axes[ref] = fig.add_axes(bounds)
    return axes


def _bottom_axis_refs(refs: list[tuple[str, str]], layout: dict[str, Any]) -> set[tuple[str, str]]:
    if not refs:
        return set()
    y_starts = [_axis_domain(layout, ref[1], (0.0, 1.0))[0] for ref in refs]
    bottom = min(y_starts)
    return {ref for ref, y0 in zip(refs, y_starts) if abs(y0 - bottom) < 1e-6}


def _dedupe_legend_items(axes: list[Any]) -> tuple[list[Any], list[str]]:
    seen: set[str] = set()
    handles_out: list[Any] = []
    labels_out: list[str] = []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            label_text = str(label or "").strip()
            if not label_text or label_text.startswith("_") or label_text in seen:
                continue
            seen.add(label_text)
            handles_out.append(handle)
            labels_out.append(label_text)
    return handles_out, labels_out


def _add_publication_legend(fig, axes: list[Any], *, legend_outside: bool) -> None:
    handles, labels = _dedupe_legend_items(axes)
    if not handles:
        return
    if legend_outside:
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.84, 0.965),
            frameon=False,
            fontsize=6.8,
            markerscale=0.75,
            handlelength=1.1,
            handletextpad=0.45,
            labelspacing=0.34,
            borderaxespad=0.0,
        )
    else:
        axes[0].legend(handles, labels, loc="best", frameon=False, fontsize=7.2)


def matplotlib_pdf_from_plotly(
    figure: go.Figure | dict[str, Any],
    *,
    title: str | None = None,
    width: int = 1200,
    height: int = 820,
    legend_outside: bool = True,
    include_title: bool = False,
) -> bytes:
    """Render a Plotly figure state as a publication-quality Matplotlib PDF."""
    plt, _to_rgba, _Circle, _Rectangle = _matplotlib_imports()
    import numpy as np

    raw = go.Figure(figure).to_dict()
    data = [trace for trace in raw.get("data", []) if isinstance(trace, dict) and trace.get("visible") not in (False, "legendonly")]
    layout = raw.get("layout") if isinstance(raw.get("layout"), dict) else {}
    refs = _subplot_refs(data, layout)
    figsize = (max(4.6, width / 240.0), max(3.2, height / 240.0))
    with plt.rc_context(MATPLOTLIB_PUBLICATION_STYLE):
        fig = plt.figure(figsize=figsize, constrained_layout=False)
        ax_by_ref = _axes_from_plotly_domains(fig, refs, layout, legend_outside=legend_outside)
        axes_list = [ax_by_ref[ref] for ref in refs]
        previous_by_ref: dict[tuple[str, str], tuple[list[float], list[float]] | None] = {ref: None for ref in refs}
        for trace in data:
            ref = (_axis_ref(trace, "xaxis"), _axis_ref(trace, "yaxis"))
            ax = ax_by_ref.get(ref, axes_list[0])
            previous_by_ref[ref] = _draw_trace(ax, trace, previous_by_ref.get(ref))
        bottom_refs = _bottom_axis_refs(refs, layout)
        for ref, ax in ax_by_ref.items():
            _draw_shapes(ax, layout, ref[0], ref[1])
            _apply_axis_layout(ax, layout, ref[0], ref[1], show_xlabel=ref in bottom_refs)
            ax.grid(True, color="#9ca3af", alpha=0.25, linewidth=0.55)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#333333")
                ax.spines[side].set_linewidth(0.8)
        _add_publication_legend(fig, axes_list, legend_outside=legend_outside)
        title_text = _clean_math_text(title or _layout_text(layout, "title") or "")
        if include_title and title_text:
            fig.suptitle(title_text, fontsize=12.5, fontweight="semibold")
        if not data:
            axes_list[0].text(0.5, 0.5, "No data", transform=axes_list[0].transAxes, ha="center", va="center")
        buf = BytesIO()
        try:
            fig.savefig(buf, format="pdf", metadata={"Creator": "MALCA"}, dpi=300, bbox_inches="tight", pad_inches=0.04)
            return buf.getvalue()
        finally:
            plt.close(fig)


def render_publication_pdf(
    figure: go.Figure | dict[str, Any],
    *,
    title: str | None = None,
    width: int = 1200,
    height: int = 820,
    legend_outside: bool = True,
    right_margin: int | None = None,
    top_margin: int = 92,
    bottom_margin: int = 78,
    left_margin: int = 90,
    xaxis_title: object | None = None,
    yaxis_title: object | None = None,
    style: bool = True,
) -> bytes:
    """Render a review figure as a Matplotlib-backed publication PDF.

    Plotly remains the interactive display format in Dash. Export callbacks
    should use this helper so PDF output has one non-Kaleido path.
    """
    export_fig = (
        publication_figure(
            figure,
            title=title,
            width=width,
            height=height,
            legend_outside=legend_outside,
            right_margin=right_margin,
            top_margin=top_margin,
            bottom_margin=bottom_margin,
            left_margin=left_margin,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
        )
        if style
        else go.Figure(figure)
    )
    return matplotlib_pdf_from_plotly(
        export_fig,
        title=title,
        width=width,
        height=height,
        legend_outside=legend_outside,
    )
