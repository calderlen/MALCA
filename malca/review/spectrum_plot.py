from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def build_spectrum_figure(
    wavelength: np.ndarray,
    flux: np.ndarray,
    *,
    flux_err: np.ndarray | None = None,
    survey: str = "",
    candidate_id: str = "",
    redshift: float | None = None,
    theme: str = "dark",
    height: int = 400,
) -> go.Figure:
    """Build a λ vs flux plot for a single spectrum."""
    tokens = _theme_tokens(theme)
    fig = go.Figure()

    if flux_err is not None and len(flux_err) == len(wavelength):
        upper = flux + flux_err
        lower = flux - flux_err
        fig.add_trace(go.Scatter(
            x=np.concatenate([wavelength, wavelength[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=tokens["error_fill"],
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=wavelength,
        y=flux,
        mode="lines",
        line=dict(color=tokens["line_color"], width=1.2),
        name=survey or "spectrum",
    ))

    title_parts = []
    if survey:
        title_parts.append(survey)
    if candidate_id:
        title_parts.append(candidate_id)
    if redshift is not None and np.isfinite(redshift):
        title_parts.append(f"z={redshift:.4f}")
    title = " | ".join(title_parts) if title_parts else "Spectrum"

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=tokens["font"])),
        xaxis=dict(
            title="Wavelength (Å)",
            color=tokens["font"],
            gridcolor=tokens["grid"],
            zeroline=False,
        ),
        yaxis=dict(
            title="Flux",
            color=tokens["font"],
            gridcolor=tokens["grid"],
            zeroline=False,
        ),
        paper_bgcolor=tokens["paper_bg"],
        plot_bgcolor=tokens["plot_bg"],
        font=dict(color=tokens["font"]),
        margin=dict(l=60, r=20, t=40, b=50),
        height=height,
        legend=dict(
            bgcolor=tokens["legend_bg"],
            bordercolor=tokens["legend_border"],
        ),
    )
    return fig


def _theme_tokens(theme: str) -> dict[str, str]:
    mode = str(theme or "dark").strip().lower()
    if mode == "white":
        return {
            "paper_bg": "#ffffff",
            "plot_bg": "#ffffff",
            "font": "#1c2733",
            "grid": "rgba(104, 128, 149, 0.18)",
            "line_color": "#2563eb",
            "error_fill": "rgba(37, 99, 235, 0.12)",
            "legend_bg": "rgba(255, 255, 255, 0.92)",
            "legend_border": "rgba(120, 140, 158, 0.35)",
        }
    if mode == "gray":
        return {
            "paper_bg": "#2e3440",
            "plot_bg": "#2e3440",
            "font": "#d8dee9",
            "grid": "rgba(216, 222, 233, 0.12)",
            "line_color": "#88c0d0",
            "error_fill": "rgba(136, 192, 208, 0.15)",
            "legend_bg": "rgba(46, 52, 64, 0.92)",
            "legend_border": "rgba(216, 222, 233, 0.2)",
        }
    return {
        "paper_bg": "#0a1628",
        "plot_bg": "#0a1628",
        "font": "#c8d8e6",
        "grid": "rgba(104, 128, 149, 0.18)",
        "line_color": "#5eead4",
        "error_fill": "rgba(94, 234, 212, 0.12)",
        "legend_bg": "rgba(8, 16, 24, 0.92)",
        "legend_border": "rgba(120, 140, 158, 0.25)",
    }
