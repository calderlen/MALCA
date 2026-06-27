"""DustyCult occulter visualization helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from malca.plotting.lightcurve_publication import PUBLICATION_PLOTLY_FONT
from malca.review.dustycult import DUSTYCULT_BANDPASS_NM, parse_json_cell


@dataclass(frozen=True)
class DustOcculterParameters:
    t0: float
    v: float
    b: float
    tau0: float
    lambda0: float
    alpha: float
    sigma_y: float
    sigma_x_plus: float
    sigma_x_minus: float


@dataclass(frozen=True)
class DustStarParameters:
    R: float = 1.0
    I0: float = 1.0
    u1: float = 0.0
    u2: float = 0.0


def _row_get(row: Mapping[str, Any] | pd.Series, key: str, default: Any = None) -> Any:
    if isinstance(row, pd.Series):
        return row.get(key, default)
    return row.get(key, default)


def _json_dict(value: object) -> dict[str, Any]:
    parsed = parse_json_cell(value, {})
    return parsed if isinstance(parsed, dict) else {}


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _posterior_median(posterior: Mapping[str, Any], key: str) -> float | None:
    value = posterior.get(key)
    if isinstance(value, Mapping):
        for stat_key in ("median", "p50", "mean"):
            number = _finite_float(value.get(stat_key))
            if number is not None:
                return number
    return _finite_float(value)


def _positive_from_posterior(
    posterior: Mapping[str, Any],
    physical_key: str,
    log_key: str,
) -> float | None:
    value = _posterior_median(posterior, physical_key)
    if value is not None and value > 0:
        return value
    log_value = _posterior_median(posterior, log_key)
    if log_value is None:
        return None
    try:
        value = math.exp(log_value)
    except OverflowError:
        return None
    return value if np.isfinite(value) and value > 0 else None


def occulter_parameters_from_fit(row: Mapping[str, Any] | pd.Series) -> DustOcculterParameters:
    """Extract physical DustyCult occulter parameters from a fit row."""
    posterior = _json_dict(_row_get(row, "posterior_json"))
    config = _json_dict(_row_get(row, "config_json"))
    bandpass = config.get("bandpass") if isinstance(config.get("bandpass"), dict) else {}
    wavelengths = [
        _finite_float(entry.get("wavelength"))
        for entry in bandpass.values()
        if isinstance(entry, Mapping)
    ]
    wavelengths = [w for w in wavelengths if w is not None and w > 0]

    t0 = _posterior_median(posterior, "t0") or _finite_float(_row_get(row, "t0_jd"))
    v = _positive_from_posterior(posterior, "v", "log_v")
    tau0 = _positive_from_posterior(posterior, "tau0", "log_tau0")
    lambda0 = _positive_from_posterior(posterior, "lambda0", "log_lambda0")
    if lambda0 is None:
        lambda0 = float(np.median(wavelengths)) if wavelengths else float(np.median(list(DUSTYCULT_BANDPASS_NM.values())))
    sigma_y = _positive_from_posterior(posterior, "sigma_y", "log_sigma_y")
    sigma_x_plus = _positive_from_posterior(posterior, "sigma_x_plus", "log_sigma_x_plus")
    sigma_x_minus = _positive_from_posterior(posterior, "sigma_x_minus", "log_sigma_x_minus")
    b = _posterior_median(posterior, "b")
    alpha = _posterior_median(posterior, "alpha")

    missing = [
        name
        for name, value in (
            ("t0", t0),
            ("v", v),
            ("tau0", tau0),
            ("sigma_y", sigma_y),
            ("sigma_x_plus", sigma_x_plus),
            ("sigma_x_minus", sigma_x_minus),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"Missing DustyCult posterior parameters: {', '.join(missing)}")

    return DustOcculterParameters(
        t0=float(t0),
        v=float(v),
        b=float(0.0 if b is None else b),
        tau0=float(tau0),
        lambda0=float(lambda0),
        alpha=float(0.0 if alpha is None else alpha),
        sigma_y=float(sigma_y),
        sigma_x_plus=float(sigma_x_plus),
        sigma_x_minus=float(sigma_x_minus),
    )


def star_parameters_from_fit(row: Mapping[str, Any] | pd.Series) -> DustStarParameters:
    stellar = _json_dict(_row_get(row, "stellar_json"))
    if not stellar:
        config = _json_dict(_row_get(row, "config_json"))
        value = config.get("star")
        stellar = value if isinstance(value, dict) else {}
    radius = _finite_float(stellar.get("R")) or 1.0
    if radius <= 0:
        radius = 1.0
    return DustStarParameters(
        R=float(radius),
        I0=float(_finite_float(stellar.get("I0")) or 1.0),
        u1=float(_finite_float(stellar.get("u1")) or 0.0),
        u2=float(_finite_float(stellar.get("u2")) or 0.0),
    )


def occulter_absorption_grid(
    dust: DustOcculterParameters,
    star: DustStarParameters,
    wavelength_nm: float,
    *,
    grid_n: int = 251,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return x/Rstar, y/Rstar, absorption, and displayed extent."""
    scale = max(dust.sigma_y, dust.sigma_x_plus, dust.sigma_x_minus) / star.R
    center_y = abs(dust.b) / star.R
    extent = float(np.clip(max(1.35, center_y + 3.2 * scale, 3.2 * scale), 1.35, 5.0))
    coords = np.linspace(-extent, extent, int(grid_n))
    xx_norm, yy_norm = np.meshgrid(coords, coords)
    xx = xx_norm * star.R
    yy = yy_norm * star.R
    x_occ = xx
    y_occ = yy - dust.b
    shape_y = np.exp(-0.5 * (y_occ / dust.sigma_y) ** 2)
    shape = 0.5 * (
        np.exp(-0.5 * (x_occ / dust.sigma_x_plus) ** 2) * shape_y
        + np.exp(-0.5 * (x_occ / dust.sigma_x_minus) ** 2) * shape_y
    )
    tau = shape * dust.tau0 * (float(wavelength_nm) / dust.lambda0) ** (-dust.alpha)
    absorption = 1.0 - np.exp(-tau)
    return coords, coords, absorption, extent


def build_dustycult_occulter_figure(
    fit_row: Mapping[str, Any] | pd.Series,
    *,
    theme: str | None = None,
    grid_n: int = 251,
    title: str = "DustyCult Occulter Model",
) -> go.Figure:
    """Build a static g/V occulter absorption map from the fit posterior median."""
    dust = occulter_parameters_from_fit(fit_row)
    star = star_parameters_from_fit(fit_row)
    mode = str(_row_get(fit_row, "mode", "") or "").strip()
    mode_title = f" ({mode})" if mode else ""
    white = str(theme or "").strip().lower() == "white"
    paper = "#ffffff" if white else "#0d0d0d"
    plot = "#ffffff" if white else "#0d0d0d"
    font = "#111111" if white else "#dce8f2"
    grid = "rgba(0,0,0,0.16)" if white else "rgba(96,116,130,0.22)"
    bands = [("g", DUSTYCULT_BANDPASS_NM["g"]), ("V", DUSTYCULT_BANDPASS_NM["V"])]
    grids = [occulter_absorption_grid(dust, star, wavelength, grid_n=grid_n) for _band, wavelength in bands]
    zmax = max(float(np.nanmax(values[2])) for values in grids)
    zmax = max(zmax, 1e-6)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"{band} ({wavelength:.0f} nm)" for band, wavelength in bands],
        horizontal_spacing=0.08,
    )
    for idx, ((band, _wavelength), (x, y, absorption, extent)) in enumerate(zip(bands, grids), start=1):
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=absorption,
                zmin=0.0,
                zmax=zmax,
                coloraxis="coloraxis",
                zsmooth="best",
                hoverongaps=False,
                hovertemplate=(
                    f"<b>{band}</b><br>"
                    "$x/R_\\star$: %{x:.3f}<br>"
                    "$y/R_\\star$: %{y:.3f}<br>"
                    "absorption: %{z:.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=idx,
        )
        xref = "x" if idx == 1 else f"x{idx}"
        yref = "y" if idx == 1 else f"y{idx}"
        center_y = dust.b / star.R
        fig.add_shape(
            type="circle",
            x0=-1,
            x1=1,
            y0=-1,
            y1=1,
            xref=xref,
            yref=yref,
            line=dict(color=font, width=1.8),
        )
        fig.add_shape(
            type="line",
            x0=-extent,
            x1=extent,
            y0=center_y,
            y1=center_y,
            xref=xref,
            yref=yref,
            line=dict(color="#d66b6b", width=1.2, dash="dash"),
        )
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[center_y],
                mode="markers",
                marker=dict(size=7, color="#d66b6b", line=dict(color=font, width=0.7)),
                name="occulter center" if idx == 1 else "occulter center",
                showlegend=False,
                hovertemplate="center<br>$x/R_\\star$: %{x:.3f}<br>$y/R_\\star$: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=idx,
        )
        fig.add_annotation(
            x=0.78 * extent,
            y=center_y,
            ax=-0.78 * extent,
            ay=center_y,
            xref=xref,
            yref=yref,
            axref=xref,
            ayref=yref,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.2,
            arrowcolor=font,
            text="",
        )
        fig.update_xaxes(range=[-extent, extent], scaleanchor=yref, scaleratio=1, constrain="domain", row=1, col=idx)
        fig.update_yaxes(range=[-extent, extent], constrain="domain", row=1, col=idx)

    fig.update_layout(
        template=None,
        title=dict(
            text=f"{title}{mode_title}",
            x=0.02,
            xanchor="left",
            y=0.985,
            yanchor="top",
            font=dict(size=14, family=PUBLICATION_PLOTLY_FONT),
        ),
        paper_bgcolor=paper,
        plot_bgcolor=plot,
        font=dict(color=font, family=PUBLICATION_PLOTLY_FONT, size=11),
        margin=dict(l=58, r=94, t=78, b=62),
        height=430,
        showlegend=False,
        coloraxis=dict(
            colorscale="Inferno",
            cmin=0.0,
            cmax=zmax,
            colorbar=dict(
                title=dict(text="absorption", side="right", font=dict(size=11)),
                len=0.76,
                y=0.46,
                yanchor="middle",
                thickness=18,
                x=1.02,
                xanchor="left",
                tickformat=".2f",
                tickfont=dict(size=10),
                ticks="outside",
            ),
        ),
    )
    for annotation in fig.layout.annotations:
        if annotation.xref == "paper" and annotation.yref == "paper" and not annotation.showarrow:
            annotation.font = dict(size=14, color=font)
            annotation.y = 1.02
    fig.update_xaxes(
        title=dict(text=r"$x/R_\star$", standoff=8),
        gridcolor=grid,
        zeroline=False,
        ticks="outside",
        tickfont=dict(size=10),
        constrain="domain",
    )
    fig.update_yaxes(
        title=dict(text=r"$y/R_\star$", standoff=8),
        gridcolor=grid,
        zeroline=False,
        ticks="outside",
        tickfont=dict(size=10),
        constrain="domain",
    )
    return fig
