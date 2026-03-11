"""Diagnostic plot builders for the review GUI.

Each function takes the candidate payload dict and a theme string,
returning a plotly Figure or None if required data is missing.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from malca.config.config_ltv import CMD_A_G_PER_AV, CMD_E_BP_RP_PER_AV
from malca.config.config_classify import (
    YSO_CLASS_I_W1W2,
    YSO_CLASS_II_W1W2_MIN,
    YSO_CLASS_II_HK,
    YSO_DUST_CORRECTION_HK,
    YSO_DUST_CORRECTION_W1W2,
)


_BACKGROUND_POINT_LIMIT = 4000


def _safe_float(payload: dict, key: str) -> float | None:
    """Extract a finite float from the payload, or None."""
    v = payload.get(key)
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _theme_spec(theme: str) -> dict:
    """Minimal theme tokens matching _external_followup_theme in app.py."""
    mode = str(theme or "black").strip().lower()
    if mode == "white":
        return {
            "paper_bg": "#ffffff",
            "plot_bg": "#ffffff",
            "font": "#1c2733",
            "grid": "rgba(104, 128, 149, 0.18)",
            "muted": "#5a6b7b",
            "marker": "#e03e2d",
            "region_alpha": 0.12,
        }
    if mode == "gray":
        return {
            "paper_bg": "#2e3440",
            "plot_bg": "#2e3440",
            "font": "#d8dee9",
            "grid": "rgba(129, 161, 193, 0.15)",
            "muted": "#aab6c7",
            "marker": "#5bc0de",
            "region_alpha": 0.15,
        }
    return {
        "paper_bg": "#0d0d0d",
        "plot_bg": "#0d0d0d",
        "font": "#dce8f2",
        "grid": "rgba(96, 116, 130, 0.22)",
        "muted": "#9fb6cb",
        "marker": "#5bc0de",
        "region_alpha": 0.15,
    }


def _apply_layout(fig: go.Figure, *, title: str, spec: dict, height: int = 280) -> go.Figure:
    """Apply consistent themed layout to a diagnostic figure."""
    fig.update_layout(
        height=height,
        margin=dict(l=48, r=12, t=30, b=36),
        title=dict(text=title, font=dict(size=12)),
        paper_bgcolor=spec["paper_bg"],
        plot_bgcolor=spec["plot_bg"],
        font=dict(color=spec["font"], size=10),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=spec["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=spec["grid"], zeroline=False)
    return fig


def _axis_range_with_padding(values: list[float], *, pad_fraction: float, min_pad: float) -> list[float] | None:
    """Return a padded axis range covering finite values, or None."""
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return None
    lo = min(finite)
    hi = max(finite)
    span = hi - lo
    pad = max(min_pad, span * pad_fraction)
    if span <= 0:
        pad = max(min_pad, abs(lo) * pad_fraction, 0.25)
    return [float(lo - pad), float(hi + pad)]


def _finite_xy(background: dict | None, x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return finite paired background arrays for a diagnostic plane."""
    bg = background or {}
    raw_x = bg.get(x_key)
    raw_y = bg.get(y_key)
    if raw_x is None or raw_y is None:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    try:
        arr_x = np.asarray(raw_x, dtype=np.float64).reshape(-1)
        arr_y = np.asarray(raw_y, dtype=np.float64).reshape(-1)
    except Exception:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    if arr_x.size == 0 or arr_y.size == 0 or arr_x.size != arr_y.size:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    if not mask.any():
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    arr_x = arr_x[mask]
    arr_y = arr_y[mask]
    if arr_x.size > _BACKGROUND_POINT_LIMIT:
        idx = np.linspace(0, arr_x.size - 1, _BACKGROUND_POINT_LIMIT, dtype=int)
        arr_x = arr_x[idx]
        arr_y = arr_y[idx]
    return arr_x, arr_y


def _safe_float_from_keys(payload: dict, keys: tuple[str, ...]) -> float | None:
    """Return the first present finite float from *keys*."""
    for key in keys:
        value = _safe_float(payload, key)
        if value is not None:
            return value
    return None


def _log_axis_range(values: list[float], *, pad_fraction: float = 0.08, min_pad: float = 0.12) -> list[float] | None:
    """Return plotly log10 axis bounds for positive values."""
    positive = [float(v) for v in values if v is not None and math.isfinite(float(v)) and float(v) > 0]
    if not positive:
        return None
    lo = math.log10(min(positive))
    hi = math.log10(max(positive))
    span = hi - lo
    pad = max(min_pad, span * pad_fraction)
    if span <= 0:
        pad = max(min_pad, 0.25)
    return [float(lo - pad), float(hi + pad)]


def _metric_plane_figure(
    payload: dict,
    theme: str,
    *,
    background: dict | None,
    title: str,
    x_keys: tuple[str, ...],
    y_keys: tuple[str, ...],
    background_x_key: str,
    background_y_key: str,
    x_title: str,
    y_title: str,
    x_log: bool = False,
    y_log: bool = False,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
    x_pad_fraction: float = 0.08,
    y_pad_fraction: float = 0.08,
    x_min_pad: float = 0.2,
    y_min_pad: float = 0.2,
    vlines: tuple[float, ...] = (),
    hlines: tuple[float, ...] = (),
    diagonal: bool = False,
    annotations: tuple[tuple[float, float, str], ...] = (),
) -> go.Figure | None:
    """Build a generic candidate-vs-population metric plane."""
    x_val = _safe_float_from_keys(payload, x_keys)
    y_val = _safe_float_from_keys(payload, y_keys)
    if x_val is None or y_val is None:
        return None
    if (x_log and x_val <= 0) or (y_log and y_val <= 0):
        return None

    spec = _theme_spec(theme)
    fig = go.Figure()
    bg_x, bg_y = _finite_xy(background, background_x_key, background_y_key)
    if bg_x.size > 0:
        mask = np.ones(bg_x.size, dtype=bool)
        if x_log:
            mask &= bg_x > 0
        if y_log:
            mask &= bg_y > 0
        bg_x = bg_x[mask]
        bg_y = bg_y[mask]
    if bg_x.size > 0:
        fig.add_trace(go.Scattergl(
            x=bg_x,
            y=bg_y,
            mode="markers",
            marker=dict(size=2, color=spec["muted"], opacity=0.22),
            hoverinfo="skip",
            name="Sample",
        ))

    for value in vlines:
        fig.add_vline(x=float(value), line_color=spec["muted"], line_dash="dot", line_width=1.0, opacity=0.6)
    for value in hlines:
        fig.add_hline(y=float(value), line_color=spec["muted"], line_dash="dot", line_width=1.0, opacity=0.6)

    if diagonal:
        diag_vals = [x_val, y_val]
        if bg_x.size > 0:
            diag_vals.extend(bg_x.tolist())
            diag_vals.extend(bg_y.tolist())
        diag_vals = [float(v) for v in diag_vals if math.isfinite(float(v)) and ((not x_log and not y_log) or float(v) > 0)]
        if diag_vals:
            d0 = min(diag_vals)
            d1 = max(diag_vals)
            fig.add_shape(
                type="line",
                x0=d0,
                y0=d0,
                x1=d1,
                y1=d1,
                line=dict(color=spec["muted"], dash="dot", width=1.0),
                opacity=0.7,
            )

    for ann_x, ann_y, text in annotations:
        fig.add_annotation(
            x=float(ann_x),
            y=float(ann_y),
            text=text,
            showarrow=False,
            font=dict(size=8, color=spec["muted"]),
            opacity=0.8,
        )

    fig.add_trace(go.Scattergl(
        x=[x_val],
        y=[y_val],
        mode="markers",
        marker=dict(size=12, color=spec["marker"], symbol="star", line=dict(width=2, color=spec["font"])),
        name="Candidate",
        hovertemplate=f"{x_title} = {x_val:.3g}<br>{y_title} = {y_val:.3g}<extra></extra>",
    ))

    _apply_layout(fig, title=title, spec=spec)

    x_values = [x_val]
    y_values = [y_val]
    if bg_x.size > 0:
        x_values.extend(bg_x.tolist())
    if bg_y.size > 0:
        y_values.extend(bg_y.tolist())

    if x_log:
        fig.update_xaxes(title=x_title, type="log", range=_log_axis_range(x_values) if x_range is None else [math.log10(x_range[0]), math.log10(x_range[1])])
    else:
        fig.update_xaxes(title=x_title, range=x_range or _axis_range_with_padding(x_values, pad_fraction=x_pad_fraction, min_pad=x_min_pad))

    if y_log:
        fig.update_yaxes(title=y_title, type="log", range=_log_axis_range(y_values) if y_range is None else [math.log10(y_range[0]), math.log10(y_range[1])])
    else:
        fig.update_yaxes(title=y_title, range=y_range or _axis_range_with_padding(y_values, pad_fraction=y_pad_fraction, min_pad=y_min_pad))

    return fig


# ---------------------------------------------------------------------------
# 1. Gaia CMD
# ---------------------------------------------------------------------------

_CMD_REGIONS = [
    # (label, x, y)  — approximate positions for text annotations
    ("Main Sequence", 2.0, 8.0),
    ("Red Giant Branch", 2.5, -1.0),
    ("White Dwarfs", -0.2, 13.0),
    ("Pre-MS", 3.5, 5.5),
]


def build_cmd_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Gaia extinction-corrected color-magnitude diagram."""
    g_mag = _safe_float(payload, "phot_g_mean_mag")
    bp_rp = _safe_float(payload, "bp_rp")
    av = _safe_float(payload, "A_v_3d")
    bprp0_payload = _safe_float(payload, "bprp0")
    mg0_payload = _safe_float(payload, "mg0")

    # Distance: prefer distance_gspphot, fall back to parallax
    dist = _safe_float(payload, "distance_gspphot")
    if dist is None or dist <= 0:
        plx = _safe_float(payload, "parallax")
        if plx is not None and plx > 0:
            dist = 1000.0 / plx
        else:
            dist = None

    if g_mag is None or dist is None or dist <= 0:
        return None
    if bp_rp is None:
        # Try computing from BP and RP
        bp = _safe_float(payload, "phot_bp_mean_mag")
        rp = _safe_float(payload, "phot_rp_mean_mag")
        if bp is not None and rp is not None:
            bp_rp = bp - rp
        else:
            return None

    m_g = g_mag - 5.0 * math.log10(dist) + 5.0

    if mg0_payload is not None and bprp0_payload is not None:
        m_g0 = mg0_payload
        bp_rp0 = bprp0_payload
    elif av is not None and av >= 0:
        m_g0 = m_g - CMD_A_G_PER_AV * av
        bp_rp0 = bp_rp - CMD_E_BP_RP_PER_AV * av
    else:
        m_g0 = m_g
        bp_rp0 = bp_rp

    spec = _theme_spec(theme)
    fig = go.Figure()

    # Background sample
    bg = background or {}
    bg_x = bg.get("cmd_bprp0")
    bg_y = bg.get("cmd_mg0")
    if bg_x is not None and bg_y is not None and len(bg_x) > 0:
        fig.add_trace(go.Scattergl(
            x=bg_x, y=bg_y,
            mode="markers",
            marker=dict(size=2, color=spec["muted"], opacity=0.25),
            hoverinfo="skip",
            name="Sample",
        ))

    # If dereddened differs from observed, show both with reddening vector
    show_vector = (abs(bp_rp0 - bp_rp) > 0.01 or abs(m_g0 - m_g) > 0.01)

    if show_vector:
        # Observed (hollow)
        fig.add_trace(go.Scattergl(
            x=[bp_rp], y=[m_g],
            mode="markers",
            marker=dict(size=10, color=spec["plot_bg"], symbol="star",
                        line=dict(width=1.5, color=spec["marker"])),
            name="Observed",
            hovertemplate=f"BP-RP = {bp_rp:.2f}<br>M_G = {m_g:.2f}<extra>observed</extra>",
        ))
        # Reddening vector arrow
        fig.add_annotation(
            x=bp_rp0, y=m_g0, ax=bp_rp, ay=m_g,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1.2,
            arrowcolor=spec["muted"], opacity=0.6,
        )

    # Dereddened (filled)
    fig.add_trace(go.Scattergl(
        x=[bp_rp0], y=[m_g0],
        mode="markers",
        marker=dict(size=12, color=spec["marker"], symbol="star",
                    line=dict(width=2, color=spec["font"])),
        name="Dereddened",
        hovertemplate=f"BP-RP₀ = {bp_rp0:.2f}<br>M_G₀ = {m_g0:.2f}<extra>dereddened</extra>",
    ))

    # Region labels
    for label, rx, ry in _CMD_REGIONS:
        fig.add_annotation(
            x=rx, y=ry, text=label, showarrow=False,
            font=dict(size=8, color=spec["muted"]),
            opacity=0.7,
        )

    default_x_range = [-0.5, 5.0]
    default_y_range = [16, -8]
    x_range = _axis_range_with_padding(
        [default_x_range[0], default_x_range[1], bp_rp, bp_rp0],
        pad_fraction=0.08,
        min_pad=0.2,
    ) or default_x_range
    y_range = _axis_range_with_padding(
        [-8.0, 16.0, m_g, m_g0],
        pad_fraction=0.08,
        min_pad=0.6,
    ) or [-8.0, 16.0]

    _apply_layout(fig, title="Gaia CMD", spec=spec)
    fig.update_xaxes(title="BP - RP₀", range=x_range)
    fig.update_yaxes(title="M<sub>G,0</sub>", range=[y_range[1], y_range[0]])

    return fig


# ---------------------------------------------------------------------------
# 2. IR Color-Color Diagram
# ---------------------------------------------------------------------------

def build_ir_colorcolor_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """H-K vs W1-W2 infrared color-color diagram with YSO regions."""
    # Observed magnitudes
    h = _safe_float(payload, "tmass_h")
    k = _safe_float(payload, "tmass_k")
    w1 = _safe_float(payload, "unwise_w1")
    w2 = _safe_float(payload, "unwise_w2")

    if h is None or k is None or w1 is None or w2 is None:
        return None

    hk_obs = h - k
    w1w2_obs = w1 - w2

    # Pre-computed dereddened colors or compute from A_v
    hk_dered = _safe_float(payload, "H_K_dered")
    w1w2_dered = _safe_float(payload, "W1_W2_dered")

    if hk_dered is None or w1w2_dered is None:
        av = _safe_float(payload, "A_v_3d")
        if av is not None and av > 0:
            hk_dered = hk_obs - YSO_DUST_CORRECTION_HK * av
            w1w2_dered = w1w2_obs - YSO_DUST_CORRECTION_W1W2 * av
        else:
            hk_dered = hk_obs
            w1w2_dered = w1w2_obs

    spec = _theme_spec(theme)
    fig = go.Figure()

    # Background sample
    bg = background or {}
    bg_x = bg.get("ir_w1w2")
    bg_y = bg.get("ir_hk")
    if bg_x is not None and bg_y is not None and len(bg_x) > 0:
        fig.add_trace(go.Scattergl(
            x=bg_x, y=bg_y,
            mode="markers",
            marker=dict(size=2, color=spec["muted"], opacity=0.25),
            hoverinfo="skip",
            name="Sample",
        ))

    # YSO classification regions (x = W1-W2, y = H-K)
    # Axis ranges for shaded rectangles
    x_lo, x_hi = -0.5, 2.5
    y_lo, y_hi = -0.3, 2.0

    regions = [
        # (label, x0, x1, y0, y1, color)
        ("Main Sequence", x_lo, YSO_CLASS_II_W1W2_MIN, y_lo, y_hi, "gray"),
        ("Transition Disk", YSO_CLASS_II_W1W2_MIN, YSO_CLASS_I_W1W2, y_lo, YSO_CLASS_II_HK, "gold"),
        ("Class II", YSO_CLASS_II_W1W2_MIN, YSO_CLASS_I_W1W2, YSO_CLASS_II_HK, y_hi, "orange"),
        ("Class I", YSO_CLASS_I_W1W2, x_hi, y_lo, y_hi, "red"),
    ]

    for label, x0, x1, y0, y1, color in regions:
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
            fillcolor=color, opacity=spec["region_alpha"] * 0.5,
            line=dict(width=1, dash="dot", color=spec["muted"]),
            layer="below",
        )
        fig.add_annotation(
            x=(x0 + x1) / 2, y=y_hi - 0.1, text=label,
            showarrow=False, font=dict(size=8, color=spec["muted"]),
            opacity=0.7,
        )

    # If dereddened differs from observed, show both with reddening vector
    show_vector = (abs(hk_dered - hk_obs) > 0.01 or abs(w1w2_dered - w1w2_obs) > 0.01)

    if show_vector:
        # Observed (hollow)
        fig.add_trace(go.Scattergl(
            x=[w1w2_obs], y=[hk_obs],
            mode="markers",
            marker=dict(size=8, color=spec["plot_bg"], symbol="circle",
                        line=dict(width=1.5, color=spec["marker"])),
            name="Observed",
            hovertemplate=f"W1-W2 = {w1w2_obs:.2f}<br>H-K = {hk_obs:.2f}<extra>observed</extra>",
        ))
        # Reddening vector arrow
        fig.add_annotation(
            x=w1w2_dered, y=hk_dered, ax=w1w2_obs, ay=hk_obs,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1.2,
            arrowcolor=spec["muted"], opacity=0.6,
        )

    # Dereddened (filled)
    fig.add_trace(go.Scattergl(
        x=[w1w2_dered], y=[hk_dered],
        mode="markers",
        marker=dict(size=11, color=spec["marker"], symbol="circle",
                    line=dict(width=2, color=spec["font"])),
        name="Dereddened",
        hovertemplate=f"W1-W2₀ = {w1w2_dered:.2f}<br>H-K₀ = {hk_dered:.2f}<extra>dereddened</extra>",
    ))

    _apply_layout(fig, title="IR Color-Color", spec=spec)
    fig.update_xaxes(title="W1 - W2", range=[x_lo, x_hi])
    fig.update_yaxes(title="H - K", range=[y_lo, y_hi])

    return fig


# ---------------------------------------------------------------------------
# 3. Kiel Diagram
# ---------------------------------------------------------------------------

_KIEL_REGIONS = [
    ("Main Sequence", 7000, 4.5),
    ("Subgiant", 5800, 3.2),
    ("Red Giant Branch", 3800, 1.5),
    ("Pre-MS", 3500, 3.8),
]


def build_kiel_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Teff vs log g (Kiel diagram) with StarHorse error bars."""
    # Prefer StarHorse, fall back to Gaia GSP-Phot
    teff = _safe_float(payload, "teff50")
    logg = _safe_float(payload, "logg50")
    teff16 = _safe_float(payload, "teff16")
    teff84 = _safe_float(payload, "teff84")
    logg16 = _safe_float(payload, "logg16")
    logg84 = _safe_float(payload, "logg84")

    if teff is None or logg is None:
        teff = _safe_float(payload, "teff_gspphot")
        logg = _safe_float(payload, "logg_gspphot")
        teff16 = teff84 = logg16 = logg84 = None

    if teff is None or logg is None:
        return None

    # Compute asymmetric error bars if percentiles available
    error_x = None
    error_y = None
    if teff16 is not None and teff84 is not None:
        error_x = dict(
            type="data", symmetric=False,
            array=[teff84 - teff], arrayminus=[teff - teff16],
        )
    if logg16 is not None and logg84 is not None:
        error_y = dict(
            type="data", symmetric=False,
            array=[logg84 - logg], arrayminus=[logg - logg16],
        )

    spec = _theme_spec(theme)
    fig = go.Figure()

    # Background sample
    bg = background or {}
    bg_x = bg.get("kiel_teff")
    bg_y = bg.get("kiel_logg")
    if bg_x is not None and bg_y is not None and len(bg_x) > 0:
        fig.add_trace(go.Scattergl(
            x=bg_x, y=bg_y,
            mode="markers",
            marker=dict(size=2, color=spec["muted"], opacity=0.25),
            hoverinfo="skip",
            name="Sample",
        ))

    fig.add_trace(go.Scattergl(
        x=[teff], y=[logg],
        mode="markers",
        marker=dict(size=12, color=spec["marker"], symbol="star",
                    line=dict(width=2, color=spec["font"])),
        error_x=error_x,
        error_y=error_y,
        name="Candidate",
        hovertemplate=f"T_eff = {teff:.0f} K<br>log g = {logg:.2f}<extra></extra>",
    ))

    for label, rx, ry in _KIEL_REGIONS:
        fig.add_annotation(
            x=rx, y=ry, text=label, showarrow=False,
            font=dict(size=8, color=spec["muted"]),
            opacity=0.7,
        )

    _apply_layout(fig, title="Kiel Diagram", spec=spec)
    fig.update_xaxes(title="T<sub>eff</sub> (K)", range=[40000, 2500])
    fig.update_yaxes(title="log g (cgs)", range=[6, -1])

    return fig


# ---------------------------------------------------------------------------
# 4. Reduced Proper Motion Diagram
# ---------------------------------------------------------------------------

_RPM_REGIONS = [
    ("Main Sequence", 1.5, 12),
    ("White Dwarfs", 0.2, 16),
    ("Subdwarfs", 0.8, 14),
    ("Giants", 2.0, 5),
]


def build_rpm_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Reduced proper motion diagram (H_G vs BP-RP)."""
    g_mag = _safe_float(payload, "phot_g_mean_mag")
    pmra = _safe_float(payload, "pmra")
    pmdec = _safe_float(payload, "pmdec")

    if g_mag is None or pmra is None or pmdec is None:
        return None

    bp_rp = _safe_float(payload, "bp_rp")
    if bp_rp is None:
        bp = _safe_float(payload, "phot_bp_mean_mag")
        rp = _safe_float(payload, "phot_rp_mean_mag")
        if bp is not None and rp is not None:
            bp_rp = bp - rp
        else:
            return None

    pm_total = math.sqrt(pmra**2 + pmdec**2)  # mas/yr
    if pm_total <= 0:
        return None
    pm_arcsec = pm_total / 1000.0
    h_g = g_mag + 5.0 * math.log10(pm_arcsec) + 5.0

    spec = _theme_spec(theme)
    fig = go.Figure()

    # Background sample
    bg = background or {}
    bg_x = bg.get("rpm_bprp")
    bg_y = bg.get("rpm_hg")
    if bg_x is not None and bg_y is not None and len(bg_x) > 0:
        fig.add_trace(go.Scattergl(
            x=bg_x, y=bg_y,
            mode="markers",
            marker=dict(size=2, color=spec["muted"], opacity=0.25),
            hoverinfo="skip",
            name="Sample",
        ))

    # Candidate marker
    fig.add_trace(go.Scattergl(
        x=[bp_rp], y=[h_g],
        mode="markers",
        marker=dict(size=12, color=spec["marker"], symbol="star",
                    line=dict(width=2, color=spec["font"])),
        name="Candidate",
        hovertemplate=f"BP-RP = {bp_rp:.2f}<br>H_G = {h_g:.2f}<extra></extra>",
    ))

    for label, rx, ry in _RPM_REGIONS:
        fig.add_annotation(
            x=rx, y=ry, text=label, showarrow=False,
            font=dict(size=8, color=spec["muted"]),
            opacity=0.7,
        )

    _apply_layout(fig, title="Reduced Proper Motion", spec=spec)
    fig.update_xaxes(title="BP - RP")
    fig.update_yaxes(title="H<sub>G</sub>", autorange="reversed")

    return fig


# ---------------------------------------------------------------------------
# 5. UV-Optical Color Diagram
# ---------------------------------------------------------------------------

_UV_OPTICAL_REGIONS = [
    ("UV Excess", 0.5, 2),
    ("Normal MS", 1.5, 8),
]


def build_uv_optical_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """GALEX NUV - G vs BP-RP UV-optical color diagram."""
    nuv = _safe_float(payload, "galex_nuv")
    g_mag = _safe_float(payload, "phot_g_mean_mag")
    bp_rp = _safe_float(payload, "bp_rp")

    if nuv is None or g_mag is None or bp_rp is None:
        return None

    nuv_g = nuv - g_mag

    spec = _theme_spec(theme)
    fig = go.Figure()

    # Background sample
    bg = background or {}
    bg_x = bg.get("uv_bprp")
    bg_y = bg.get("uv_nuv_g")
    if bg_x is not None and bg_y is not None and len(bg_x) > 0:
        fig.add_trace(go.Scattergl(
            x=bg_x, y=bg_y,
            mode="markers",
            marker=dict(size=2, color=spec["muted"], opacity=0.25),
            hoverinfo="skip",
            name="Sample",
        ))

    # Candidate marker
    fig.add_trace(go.Scattergl(
        x=[bp_rp], y=[nuv_g],
        mode="markers",
        marker=dict(size=12, color=spec["marker"], symbol="star",
                    line=dict(width=2, color=spec["font"])),
        name="Candidate",
        hovertemplate=f"BP-RP = {bp_rp:.2f}<br>NUV-G = {nuv_g:.2f}<extra></extra>",
    ))

    for label, rx, ry in _UV_OPTICAL_REGIONS:
        fig.add_annotation(
            x=rx, y=ry, text=label, showarrow=False,
            font=dict(size=8, color=spec["muted"]),
            opacity=0.7,
        )

    _apply_layout(fig, title="UV-Optical Color", spec=spec)
    fig.update_xaxes(title="BP - RP")
    fig.update_yaxes(title="NUV - G")

    return fig


# ---------------------------------------------------------------------------
# 6. Periodicity Plane
# ---------------------------------------------------------------------------

def build_periodicity_plane_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Phase quality vs periodicity score against the review population."""
    periodicity = _safe_float(payload, "periodicity_score")
    phase_quality = _safe_float(payload, "phase_quality_score")
    if periodicity is None or phase_quality is None:
        return None

    spec = _theme_spec(theme)
    fig = go.Figure()
    bg_x, bg_y = _finite_xy(background, "metric_periodicity_score", "metric_phase_quality_score")
    if bg_x.size > 0:
        fig.add_trace(go.Scattergl(
            x=bg_x,
            y=bg_y,
            mode="markers",
            marker=dict(size=2, color=spec["muted"], opacity=0.22),
            hoverinfo="skip",
            name="Sample",
        ))

    fig.add_vline(x=0.5, line_color=spec["muted"], line_dash="dot", line_width=1.0, opacity=0.6)
    fig.add_hline(y=0.5, line_color=spec["muted"], line_dash="dot", line_width=1.0, opacity=0.6)
    fig.add_trace(go.Scattergl(
        x=[periodicity],
        y=[phase_quality],
        mode="markers",
        marker=dict(size=12, color=spec["marker"], symbol="star", line=dict(width=2, color=spec["font"])),
        name="Candidate",
        hovertemplate=(
            f"Periodicity score = {periodicity:.3f}<br>"
            f"Phase quality = {phase_quality:.3f}<extra></extra>"
        ),
    ))

    max_x = max(1.05, periodicity * 1.12)
    max_y = max(1.05, phase_quality * 1.12)
    if bg_x.size > 0:
        max_x = max(max_x, float(np.nanpercentile(bg_x, 99.0)) * 1.05)
    if bg_y.size > 0:
        max_y = max(max_y, float(np.nanpercentile(bg_y, 99.0)) * 1.05)

    _apply_layout(fig, title="Periodicity Plane", spec=spec)
    fig.update_xaxes(title="Periodicity score", range=[-0.05, max_x])
    fig.update_yaxes(title="Phase quality score", range=[-0.05, max_y])
    return fig


# ---------------------------------------------------------------------------
# 7. Morphology Score Plane
# ---------------------------------------------------------------------------

def build_score_balance_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Dipper score vs jumper score against the review population."""
    dipper_score = _safe_float(payload, "dipper_score")
    jumper_score = _safe_float(payload, "jumper_score")
    if dipper_score is None or jumper_score is None:
        return None

    spec = _theme_spec(theme)
    fig = go.Figure()
    bg_x, bg_y = _finite_xy(background, "metric_dipper_score", "metric_jumper_score")
    if bg_x.size > 0:
        fig.add_trace(go.Scattergl(
            x=bg_x,
            y=bg_y,
            mode="markers",
            marker=dict(size=2, color=spec["muted"], opacity=0.22),
            hoverinfo="skip",
            name="Sample",
        ))

    upper = max(6.0, dipper_score, jumper_score)
    if bg_x.size > 0:
        upper = max(upper, float(np.nanpercentile(bg_x, 99.0)))
    if bg_y.size > 0:
        upper = max(upper, float(np.nanpercentile(bg_y, 99.0)))
    upper *= 1.08

    fig.add_shape(
        type="line",
        x0=0.0,
        y0=0.0,
        x1=upper,
        y1=upper,
        line=dict(color=spec["muted"], dash="dot", width=1.0),
        opacity=0.7,
    )
    fig.add_annotation(x=upper * 0.78, y=upper * 0.18, text="Dip-like", showarrow=False, font=dict(size=8, color=spec["muted"]), opacity=0.8)
    fig.add_annotation(x=upper * 0.18, y=upper * 0.78, text="Jump-like", showarrow=False, font=dict(size=8, color=spec["muted"]), opacity=0.8)
    fig.add_trace(go.Scattergl(
        x=[dipper_score],
        y=[jumper_score],
        mode="markers",
        marker=dict(size=12, color=spec["marker"], symbol="star", line=dict(width=2, color=spec["font"])),
        name="Candidate",
        hovertemplate=(
            f"Dipper score = {dipper_score:.3f}<br>"
            f"Jumper score = {jumper_score:.3f}<extra></extra>"
        ),
    ))

    _apply_layout(fig, title="Morphology Scores", spec=spec)
    fig.update_xaxes(title="Dipper score", range=[-0.25, upper])
    fig.update_yaxes(title="Jumper score", range=[-0.25, upper])
    return fig


def build_catalog_support_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Period-support count vs dip recurrence."""
    return _metric_plane_figure(
        payload,
        theme,
        background=background,
        title="Catalog Support vs Dip Recurrence",
        x_keys=("period_n_sources",),
        y_keys=("dip_run_count",),
        background_x_key="plane_catalog_support_x",
        background_y_key="plane_catalog_support_y",
        x_title="Period N sources",
        y_title="Dip run count",
        x_range=[-0.25, 5.25],
        y_range=[-0.25, 8.25],
        vlines=(2.0,),
        hlines=(2.0,),
    )


def build_recurrence_regularity_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Dip spacing median vs spacing scatter."""
    return _metric_plane_figure(
        payload,
        theme,
        background=background,
        title="Recurrence Regularity",
        x_keys=("dip_inter_event_spacing_median",),
        y_keys=("dip_inter_event_spacing_std",),
        background_x_key="plane_recurrence_regularity_x",
        background_y_key="plane_recurrence_regularity_y",
        x_title="Dip spacing median (d)",
        y_title="Dip spacing std (d)",
        x_log=True,
        y_log=True,
        diagonal=True,
    )


def build_dip_repeatability_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Amplitude-consistency vs duration-consistency plane."""
    return _metric_plane_figure(
        payload,
        theme,
        background=background,
        title="Dip Repeatability",
        x_keys=("dip_amplitude_consistency",),
        y_keys=("dip_duration_consistency",),
        background_x_key="plane_dip_repeatability_x",
        background_y_key="plane_dip_repeatability_y",
        x_title="Amplitude consistency",
        y_title="Duration consistency",
        x_range=[-0.05, 1.05],
        y_range=[-0.05, 1.05],
        vlines=(0.5,),
        hlines=(0.5,),
    )


def build_variability_strength_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Scatter amplitude vs dipper score."""
    return _metric_plane_figure(
        payload,
        theme,
        background=background,
        title="Dipper Score vs Scatter",
        x_keys=("stats_photometry_robust_sigma_mag",),
        y_keys=("dipper_score",),
        background_x_key="plane_var_strength_x",
        background_y_key="plane_var_strength_y",
        x_title="Robust sigma (mag)",
        y_title="Dipper score",
        x_log=True,
        hlines=(5.0,),
    )


def build_stetson_scatter_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Scatter vs Stetson-J variability plane."""
    return _metric_plane_figure(
        payload,
        theme,
        background=background,
        title="Scatter vs Stetson J",
        x_keys=("stats_photometry_robust_sigma_mag",),
        y_keys=("stats_variability_stetson_J",),
        background_x_key="plane_stetson_x",
        background_y_key="plane_stetson_y",
        x_title="Robust sigma (mag)",
        y_title="Stetson J",
        x_log=True,
    )


def build_shape_impulsiveness_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Shape asymmetry vs impulsive slope."""
    return _metric_plane_figure(
        payload,
        theme,
        background=background,
        title="Shape and Impulsiveness",
        x_keys=("stats_skew",),
        y_keys=("stats_max_slope",),
        background_x_key="plane_shape_x",
        background_y_key="plane_shape_y",
        x_title="Skew",
        y_title="Max slope",
        y_log=True,
    )


def build_harmonic_quality_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Harmonic model amplitude vs reduced chi^2."""
    return _metric_plane_figure(
        payload,
        theme,
        background=background,
        title="Harmonic Fit Quality",
        x_keys=("stats_harmonics_model_amplitude",),
        y_keys=("stats_harmonics_reduced_chi2",),
        background_x_key="plane_harmonic_x",
        background_y_key="plane_harmonic_y",
        x_title="Harmonic amplitude",
        y_title="Harmonic reduced chi^2",
        x_log=True,
        y_log=True,
    )


def build_autocorr_memory_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Lag-1 autocorrelation vs autocorrelation length."""
    return _metric_plane_figure(
        payload,
        theme,
        background=background,
        title="Autocorrelation Memory",
        x_keys=("stats_variability_lag1_autocorr",),
        y_keys=("stats_autocor_length",),
        background_x_key="plane_autocorr_x",
        background_y_key="plane_autocorr_y",
        x_title="Lag-1 autocorr",
        y_title="Autocorr length",
        x_range=[-1.05, 1.05],
        y_log=True,
    )


def build_cluster_astrometry_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Cluster offset sigma vs RUWE."""
    return _metric_plane_figure(
        payload,
        theme,
        background=background,
        title="Cluster Astrometry",
        x_keys=("pm_cluster_offset_sigma",),
        y_keys=("ruwe",),
        background_x_key="plane_cluster_x",
        background_y_key="plane_cluster_y",
        x_title="PM cluster offset sigma",
        y_title="RUWE",
        x_log=True,
        y_log=True,
        vlines=(3.0,),
        hlines=(1.4,),
    )


def build_classifier_plane_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Disk-vs-EB classifier probability plane."""
    return _metric_plane_figure(
        payload,
        theme,
        background=background,
        title="Classifier Plane",
        x_keys=("P_disk",),
        y_keys=("P_eb",),
        background_x_key="plane_classifier_x",
        background_y_key="plane_classifier_y",
        x_title="P_disk",
        y_title="P_eb",
        x_range=[-0.05, 1.05],
        y_range=[-0.05, 1.05],
        diagonal=True,
        annotations=((0.78, 0.18, "disk-like"), (0.2, 0.8, "EB-like")),
    )


def build_atlas_range_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """ATLAS cyan vs orange range plane."""
    return _metric_plane_figure(
        payload,
        theme,
        background=background,
        title="ATLAS Range Plane",
        x_keys=("atlas_cyan_range",),
        y_keys=("atlas_orange_range",),
        background_x_key="plane_atlas_x",
        background_y_key="plane_atlas_y",
        x_title="ATLAS cyan range",
        y_title="ATLAS orange range",
        x_log=True,
        y_log=True,
        diagonal=True,
    )


def build_ztf_range_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """ZTF g vs r variability range plane."""
    return _metric_plane_figure(
        payload,
        theme,
        background=background,
        title="ZTF Range Plane",
        x_keys=("ztf_lc_g_range",),
        y_keys=("ztf_lc_r_range",),
        background_x_key="plane_ztf_x",
        background_y_key="plane_ztf_y",
        x_title="ZTF g range",
        y_title="ZTF r range",
        x_log=True,
        y_log=True,
        diagonal=True,
    )


def build_neowise_range_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """NEOWISE W1 vs W2 range plane."""
    return _metric_plane_figure(
        payload,
        theme,
        background=background,
        title="NEOWISE Range Plane",
        x_keys=("neowise_w1_range",),
        y_keys=("neowise_w2_range",),
        background_x_key="plane_neowise_range_x",
        background_y_key="plane_neowise_range_y",
        x_title="NEOWISE W1 range",
        y_title="NEOWISE W2 range",
        x_log=True,
        y_log=True,
        diagonal=True,
    )


def build_gaia_epoch_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Gaia-epoch coverage vs amplitude plane."""
    return _metric_plane_figure(
        payload,
        theme,
        background=background,
        title="Gaia Epoch Coverage",
        x_keys=("gaia_epoch_n_obs",),
        y_keys=("gaia_epoch_g_range",),
        background_x_key="plane_gaia_epoch_x",
        background_y_key="plane_gaia_epoch_y",
        x_title="Gaia epoch N obs",
        y_title="Gaia epoch G range",
        x_log=True,
        y_log=True,
    )


def build_ltv_trend_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Long-term trend slope vs dispersion plane."""
    return _metric_plane_figure(
        payload,
        theme,
        background=background,
        title="LTV Trend vs Dispersion",
        x_keys=("ltv_slope",),
        y_keys=("ltv_dispersion",),
        background_x_key="plane_ltv_x",
        background_y_key="plane_ltv_y",
        x_title="LTV slope (mag/yr)",
        y_title="LTV dispersion (mag)",
        y_log=True,
    )


def build_neowise_trend_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """NEOWISE W1 slope vs W1-W2 color slope."""
    return _metric_plane_figure(
        payload,
        theme,
        background=background,
        title="NEOWISE Trend Plane",
        x_keys=("ltv_neowise_w1_slope",),
        y_keys=("ltv_neowise_w1_w2_slope",),
        background_x_key="plane_neowise_trend_x",
        background_y_key="plane_neowise_trend_y",
        x_title="W1 slope (mag/yr)",
        y_title="W1-W2 slope (mag/yr)",
    )
