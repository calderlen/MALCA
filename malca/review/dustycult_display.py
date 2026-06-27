"""Pure DustyCult display helpers shared by the review app and notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from malca.plotting.lightcurve_publication import PUBLICATION_PLOTLY_FONT
from malca.review.dustycult import parse_json_cell
from malca.review.dustycult_visualization import (
    occulter_parameters_from_fit,
    star_parameters_from_fit,
)


def format_dustycult_float(value: object, digits: int = 4) -> str:
    """Format DustyCult numeric values compactly for UI tables."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    if number == 0:
        return "0"
    if abs(number) >= 10000 or abs(number) < 0.001:
        return f"{number:.{digits}g}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def dustycult_theme(theme: str | None) -> dict[str, object]:
    """Theme tokens for DustyCult review plots and tables."""
    mode = str(theme or "black").strip().lower()
    if mode == "white":
        return {
            "muted": "#5a6b7b",
            "error": "#a53a3a",
            "paper_bg": "#ffffff",
            "plot_bg": "#ffffff",
            "font": "#1c2733",
            "grid": "rgba(104, 128, 149, 0.18)",
            "legend_bg": "rgba(255, 255, 255, 0.92)",
            "legend_border": "rgba(120, 140, 158, 0.35)",
        }
    if mode == "gray":
        return {
            "muted": "#aab6c7",
            "error": "#f29f9f",
            "paper_bg": "#2e3440",
            "plot_bg": "#2e3440",
            "font": "#d8dee9",
            "grid": "rgba(129, 161, 193, 0.15)",
            "legend_bg": "rgba(59, 66, 82, 0.9)",
            "legend_border": "rgba(129, 161, 193, 0.3)",
        }
    return {
        "muted": "#9fb6cb",
        "error": "#dd8080",
        "paper_bg": "#0d0d0d",
        "plot_bg": "#0d0d0d",
        "font": "#dce8f2",
        "grid": "rgba(96, 116, 130, 0.22)",
        "legend_bg": "rgba(13, 13, 13, 0.88)",
        "legend_border": "rgba(113, 140, 160, 0.3)",
    }


def _series_get(row: pd.Series | Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return row.get(key, default)


def _latest_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    work = frame.copy()
    sort_cols = []
    for col in ("updated_at", "created_at"):
        if col in work.columns:
            parsed = pd.to_datetime(work[col], errors="coerce", utc=True)
            helper = f"__{col}_sort"
            work[helper] = parsed
            sort_cols.append(helper)
    if sort_cols:
        work = work.sort_values(sort_cols, na_position="first")
    return work.iloc[-1]


def select_dustycult_display_row(fits: pd.DataFrame | None, mode: str | None = None) -> pd.Series | None:
    """Select the DustyCult row to display: requested mode, full ok, quick ok, latest."""
    if fits is None or fits.empty:
        return None
    frame = fits.copy()
    if "mode" not in frame.columns:
        return _latest_row(frame)
    mode_series = frame["mode"].astype(str).str.lower()
    status_series = (
        frame["status"].astype(str).str.lower()
        if "status" in frame.columns
        else pd.Series([""] * len(frame), index=frame.index)
    )
    if mode:
        requested = str(mode).strip().lower()
        sub = frame[mode_series == requested]
        if sub.empty:
            return None
        ok = sub[status_series.loc[sub.index] == "ok"]
        return _latest_row(ok if not ok.empty else sub)
    for preferred in ("full", "quick"):
        matches = frame[(mode_series == preferred) & (status_series == "ok")]
        if not matches.empty:
            return _latest_row(matches)
    return _latest_row(frame)


def dustycult_status_card_rows(fits: pd.DataFrame | None) -> list[dict[str, str]]:
    """Return quick/full status card data for the review display."""
    by_mode: dict[str, pd.Series] = {}
    if fits is not None and not fits.empty and "mode" in fits.columns:
        for mode in ("quick", "full"):
            row = select_dustycult_display_row(fits, mode=mode)
            if row is not None:
                by_mode[mode] = row
    cards: list[dict[str, str]] = []
    for mode in ("quick", "full"):
        row = by_mode.get(mode)
        status = "not run"
        detail = ""
        if row is not None:
            status = str(row.get("status") or "unknown")
            detail_parts = []
            runtime = row.get("runtime_sec")
            n_points = row.get("n_input_points")
            if runtime is not None and not pd.isna(runtime):
                detail_parts.append(f"{format_dustycult_float(runtime, 3)} s")
            if n_points is not None and not pd.isna(n_points):
                detail_parts.append(f"{int(float(n_points))} pts")
            error = str(row.get("error") or "").strip()
            if status != "ok" and error:
                detail_parts.append(error[:120])
            detail = " | ".join(detail_parts)
        cards.append({"mode": mode, "label": mode.capitalize(), "status": status, "detail": detail})
    return cards


def build_dustycult_fit_figure(curves: pd.DataFrame, fit_row: pd.Series | Mapping[str, Any], theme: str | None) -> go.Figure:
    """Build the DustyCult posterior predictive figure used by the review app."""
    spec = dustycult_theme(theme)
    fig = go.Figure()
    mode = str(_series_get(fit_row, "mode", "quick") or "quick")
    palette = {"g": "#69c779", "V": "#f2c86b"}
    if curves is not None and not curves.empty:
        work = curves.copy()
        for col in ("time", "observed", "error", "lower95", "lower68", "median", "upper68", "upper95"):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
        band_order = {"g": 0, "v": 1}
        bands = (
            sorted(
                (str(b) for b in work["band"].dropna().unique()),
                key=lambda value: (band_order.get(str(value).lower(), 99), str(value)),
            )
            if "band" in work.columns
            else [""]
        )
        for band in bands:
            part = work[work["band"].astype(str) == band].sort_values("time")
            color = palette.get(band, "#7da8c4")
            name_prefix = f"{band} " if band else ""
            for lower, upper, fill, opacity in (
                ("lower95", "upper95", "95%", 0.08),
                ("lower68", "upper68", "68%", 0.16),
            ):
                if lower in part.columns and upper in part.columns:
                    interval = part[np.isfinite(part["time"]) & np.isfinite(part[lower]) & np.isfinite(part[upper])]
                    if not interval.empty:
                        fig.add_trace(go.Scatter(
                            x=interval["time"],
                            y=interval[lower],
                            mode="lines",
                            line=dict(width=0, color=color),
                            hoverinfo="skip",
                            showlegend=False,
                            legendgroup=band,
                        ))
                        fig.add_trace(go.Scatter(
                            x=interval["time"],
                            y=interval[upper],
                            mode="lines",
                            line=dict(width=0, color=color),
                            fill="tonexty",
                            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {opacity})",
                            name=f"{name_prefix}{fill}",
                            hoverinfo="skip",
                            showlegend=False,
                            legendgroup=band,
                        ))
            if "median" in part.columns:
                med = part[np.isfinite(part["time"]) & np.isfinite(part["median"])]
                if not med.empty:
                    fig.add_trace(go.Scatter(
                        x=med["time"],
                        y=med["median"],
                        mode="lines",
                        name=f"{name_prefix}median",
                        line=dict(color=color, width=2),
                        legendgroup=band,
                    ))
            if "observed" in part.columns:
                obs = part[np.isfinite(part["time"]) & np.isfinite(part["observed"])]
                if not obs.empty:
                    error_y = None
                    if "error" in obs.columns and np.isfinite(obs["error"]).any():
                        error_y = dict(type="data", array=obs["error"], visible=True, thickness=0.8)
                    fig.add_trace(go.Scatter(
                        x=obs["time"],
                        y=obs["observed"],
                        mode="markers",
                        name=f"{name_prefix}observed",
                        marker=dict(color=color, size=6, line=dict(color="#111827", width=0.5)),
                        error_y=error_y,
                        legendgroup=band,
                    ))
    else:
        fig.add_annotation(text="No predictive curve rows stored for this fit.", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")

    start = _series_get(fit_row, "start_jd")
    end = _series_get(fit_row, "end_jd")
    t0 = _series_get(fit_row, "t0_jd")
    try:
        if np.isfinite(float(start)) and np.isfinite(float(end)):
            fig.add_vrect(x0=float(start), x1=float(end), fillcolor="rgba(125,145,166,0.10)", line_width=0)
    except Exception:
        pass
    try:
        if np.isfinite(float(t0)):
            fig.add_vline(x=float(t0), line=dict(color="#d66b6b", width=1.4, dash="dash"))
    except Exception:
        pass
    fig.update_layout(
        template=None,
        title=dict(
            text=f"DustyCult {mode.capitalize()} Fit",
            x=0.02,
            xanchor="left",
            y=0.985,
            yanchor="top",
            font=dict(size=14, family=PUBLICATION_PLOTLY_FONT),
        ),
        paper_bgcolor=spec["paper_bg"],
        plot_bgcolor=spec["plot_bg"],
        font=dict(color=spec["font"], family=PUBLICATION_PLOTLY_FONT, size=11),
        margin=dict(l=58, r=24, t=86, b=56),
        height=390,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="left",
            x=0,
            bgcolor=spec["legend_bg"],
            bordercolor=spec["legend_border"],
            borderwidth=1,
            font=dict(size=10, family=PUBLICATION_PLOTLY_FONT),
            itemwidth=30,
        ),
    )
    fig.update_xaxes(title=dict(text=r"$t\ [\mathrm{JD}]$", standoff=8), gridcolor=spec["grid"], zeroline=False, ticks="outside")
    fig.update_yaxes(title=dict(text=r"$F/F_{\mathrm{GP}}$", standoff=8), gridcolor=spec["grid"], zeroline=False, ticks="outside")
    return fig


def dustycult_fit_metadata_rows(fit_row: pd.Series | Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return app/notebook metadata rows for a selected fit."""
    rows = [
        ("mode", str(_series_get(fit_row, "mode", "quick") or "quick")),
        ("status", str(_series_get(fit_row, "status", "unknown") or "unknown")),
        ("runtime", f"{format_dustycult_float(_series_get(fit_row, 'runtime_sec'), 3)} s"),
        (
            "window",
            f"{format_dustycult_float(_series_get(fit_row, 'start_jd'), 2)} to "
            f"{format_dustycult_float(_series_get(fit_row, 'end_jd'), 2)}",
        ),
    ]
    artifact = str(_series_get(fit_row, "artifact_dir", "") or "").strip()
    if artifact:
        rows.append(("artifact", artifact))
    return rows


def dustycult_fit_metadata_text(fit_row: pd.Series | Mapping[str, Any]) -> str:
    """Return the compact metadata line used by the review panel."""
    return " | ".join(f"{label}={value}" for label, value in dustycult_fit_metadata_rows(fit_row))


def dustycult_geometry_rows(fit_row: pd.Series | Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return physical occulter and stellar geometry rows from a fit posterior."""
    dust = occulter_parameters_from_fit(fit_row)
    star = star_parameters_from_fit(fit_row)
    return [
        ("t0", format_dustycult_float(dust.t0, 5)),
        ("v", format_dustycult_float(dust.v, 5)),
        ("b", format_dustycult_float(dust.b, 5)),
        ("b / R_star", format_dustycult_float(dust.b / star.R, 5)),
        ("tau0", format_dustycult_float(dust.tau0, 5)),
        ("lambda0 [nm]", format_dustycult_float(dust.lambda0, 5)),
        ("alpha", format_dustycult_float(dust.alpha, 5)),
        ("sigma_y", format_dustycult_float(dust.sigma_y, 5)),
        ("sigma_x_plus", format_dustycult_float(dust.sigma_x_plus, 5)),
        ("sigma_x_minus", format_dustycult_float(dust.sigma_x_minus, 5)),
        ("R_star", format_dustycult_float(star.R, 5)),
        ("u1", format_dustycult_float(star.u1, 5)),
        ("u2", format_dustycult_float(star.u2, 5)),
    ]


def dustycult_posterior_rows(fit_row: pd.Series | Mapping[str, Any], *, limit: int | None = 18) -> list[tuple[str, str, str, str]]:
    """Return posterior summary rows as Parameter, Median, p16, p84."""
    posterior = parse_json_cell(_series_get(fit_row, "posterior_json"), {})
    if not isinstance(posterior, dict):
        return []
    rows: list[tuple[str, str, str, str]] = []
    names = sorted(posterior.keys())
    if limit is not None:
        names = names[: int(limit)]
    for name in names:
        stats = posterior.get(name)
        if not isinstance(stats, Mapping):
            continue
        rows.append(
            (
                str(name),
                format_dustycult_float(stats.get("median")),
                format_dustycult_float(stats.get("p16")),
                format_dustycult_float(stats.get("p84")),
            )
        )
    return rows
