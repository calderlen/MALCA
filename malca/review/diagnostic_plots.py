"""Diagnostic plot builders for the review GUI.

Each function takes the candidate payload dict and a theme string,
returning a plotly Figure or None if required data is missing.
"""

from __future__ import annotations

from io import BytesIO
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import plotly.graph_objects as go

from malca.plotting.color_color_labels import (
    LABEL_H_KS,
    LABEL_W1_W2,
    TITLE_H_K,
    TITLE_W1_W2,
)
from malca.plotting.lightcurve_publication import (
    FIG_SINGLE_COL_SQUARE,
    PUBLICATION_PLOTLY_FONT,
    apply_publication_rcparams,
    finalize_publication_figure,
)
from malca.ltv.cmd import dustmaps_cmd_from_fields
from malca.config import (
    YSO_CLASS_I_W1W2,
    YSO_CLASS_II_W1W2_MIN,
    YSO_CLASS_II_HK,
    YSO_DUST_CORRECTION_HK,
    YSO_DUST_CORRECTION_W1W2,
)


_BACKGROUND_POINT_LIMIT = 4000

_PUBLICATION_DPI = 300
_PUBLICATION_DENSITY_BINS = 240
_CANDIDATE_COLOR = "#dc2626"
_DEREDDENED_COLOR = "#2563eb"
_CANDIDATE_EDGE_COLOR = "#111827"
_PLOTLY_CANDIDATE_SIZE = 9
_PUBLICATION_CANDIDATE_SIZE = 42


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
        font=dict(color=spec["font"], family=PUBLICATION_PLOTLY_FONT, size=10),
        showlegend=False,
    )
    fig.update_traces(showlegend=False)
    fig.update_xaxes(gridcolor=spec["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=spec["grid"], zeroline=False)
    return fig


def _plotly_candidate_marker(*, dereddened: bool = False, size: int = _PLOTLY_CANDIDATE_SIZE) -> dict:
    """Consistent candidate marker for diagnostic Plotly figures."""
    return dict(
        size=size,
        color=_DEREDDENED_COLOR if dereddened else _CANDIDATE_COLOR,
        symbol="circle",
        line=dict(width=1.5, color=_CANDIDATE_EDGE_COLOR),
    )


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
        marker=_plotly_candidate_marker(),
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


def _dustmaps_cmd_from_payload(payload: dict) -> dict[str, object] | None:
    """Compute CMD coordinates from payload using dustmaps3d extinction."""
    dist = _safe_float(payload, "distance_gspphot")
    plx = _safe_float(payload, "parallax")
    coords = dustmaps_cmd_from_fields(
        g_mag=_safe_float(payload, "phot_g_mean_mag"),
        bp_rp=_safe_float(payload, "bp_rp"),
        dist_pc=dist,
        a_v_3d=_safe_float(payload, "A_v_3d"),
        bp_mag=_safe_float(payload, "phot_bp_mean_mag"),
        rp_mag=_safe_float(payload, "phot_rp_mean_mag"),
        parallax_mas=plx,
    )
    if coords["cmd_coordinate_source"] == "missing":
        return None
    return coords


def build_cmd_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Gaia extinction-corrected color-magnitude diagram."""
    coords = _dustmaps_cmd_from_payload(payload)
    if coords is None:
        return None

    bp_rp = float(coords["bp_rp"])
    m_g = float(coords["mg"])
    bp_rp0 = float(coords["cmd_color"])
    m_g0 = float(coords["cmd_mag"])

    spec = _theme_spec(theme)
    fig = go.Figure()

    bg_x, bg_y = _finite_xy(background, "cmd_bprp0", "cmd_mg0")
    if bg_x.size > 0:
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
        fig.add_trace(go.Scattergl(
            x=[bp_rp], y=[m_g],
            mode="markers",
            marker=_plotly_candidate_marker(),
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

    fig.add_trace(go.Scattergl(
        x=[bp_rp0], y=[m_g0],
        mode="markers",
        marker=_plotly_candidate_marker(dereddened=True),
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
    w1 = _safe_float(payload, "w1")
    w2 = _safe_float(payload, "w2")

    if h is None or k is None or w1 is None or w2 is None:
        return None

    hk_obs = h - k
    w1w2_obs = w1 - w2

    # Pre-computed dereddened colors or compute from A_v
    hk_dered = _safe_float(payload, "H_K_dered")
    w1w2_dered = _safe_float(payload, "w1_w2_dered")

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

    bg_x, bg_y = _finite_xy(background, "ir_w1w2", "ir_hk")
    if bg_x.size > 0:
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
        fig.add_trace(go.Scattergl(
            x=[w1w2_obs], y=[hk_obs],
            mode="markers",
            marker=_plotly_candidate_marker(),
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

    fig.add_trace(go.Scattergl(
        x=[w1w2_dered], y=[hk_dered],
        mode="markers",
        marker=_plotly_candidate_marker(dereddened=True),
        name="Dereddened",
        hovertemplate=f"W1-W2₀ = {w1w2_dered:.2f}<br>H-K₀ = {hk_dered:.2f}<extra>dereddened</extra>",
    ))

    _apply_layout(fig, title="IR Color-Color", spec=spec)
    fig.update_xaxes(title=TITLE_W1_W2, range=[x_lo, x_hi])
    fig.update_yaxes(title=TITLE_H_K, range=[y_lo, y_hi])

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

    bg_x, bg_y = _finite_xy(background, "kiel_teff", "kiel_logg")
    if bg_x.size > 0:
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
        marker=_plotly_candidate_marker(),
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
    fig.update_xaxes(title="T<sub>eff</sub> [K]", range=[40000, 2500])
    fig.update_yaxes(title="log g [cgs]", range=[6, -1])

    return fig


def build_teff_sed_alpha_figure(
    payload: dict, theme: str, *, background: dict | None = None,
) -> go.Figure | None:
    """Catalog Teff vs 2-24 micron SED spectral index."""
    return _metric_plane_figure(
        payload,
        theme,
        background=background,
        title="Teff vs SED Alpha",
        x_keys=("teff50", "teff_gspphot"),
        y_keys=("sed_alpha",),
        background_x_key="plane_teff_alpha_x",
        background_y_key="plane_teff_alpha_y",
        x_title="T_eff [K]",
        y_title="SED alpha",
        hlines=(-1.6, -0.3, 0.3),
        y_range=[-3.2, 2.2],
    )


def _publication_imports():
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
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    apply_publication_rcparams(plt)
    return plt, LinearSegmentedColormap, LogNorm


def _finite_publication_xy(
    background: dict | None,
    x_key: str,
    y_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    bg = background or {}
    try:
        x = np.asarray(bg.get(x_key), dtype=float).reshape(-1)
        y = np.asarray(bg.get(y_key), dtype=float).reshape(-1)
    except Exception:
        return np.empty(0), np.empty(0)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return np.empty(0), np.empty(0)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _quantile_limits(
    values: np.ndarray,
    extras: tuple[float | None, ...] = (),
    *,
    qlo: float = 0.005,
    qhi: float = 0.995,
    default: tuple[float, float],
    pad_fraction: float = 0.06,
    min_pad: float = 0.15,
) -> tuple[float, float]:
    finite = [float(v) for v in np.asarray(values, dtype=float).reshape(-1) if np.isfinite(v)]
    if finite:
        lo = float(np.quantile(finite, qlo))
        hi = float(np.quantile(finite, qhi))
    else:
        lo, hi = default
    for extra in extras:
        if extra is None:
            continue
        try:
            value = float(extra)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            lo = min(lo, value)
            hi = max(hi, value)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = default
    pad = max(min_pad, (hi - lo) * pad_fraction)
    return lo - pad, hi + pad


def _publication_axes(ax, *, title: str | None = None, xlabel: str, ylabel: str) -> None:
    if title:
        ax.set_title(title, fontsize=10.5, pad=10, fontweight="semibold")
    ax.set_xlabel(xlabel, fontsize=9.5, labelpad=7)
    ax.set_ylabel(ylabel, fontsize=9.5, labelpad=7)
    ax.tick_params(axis="both", labelsize=8.5, length=3.5, width=0.7, colors="#222222")
    ax.grid(True, color="#9ca3af", alpha=0.24, linewidth=0.55)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#333333")
        ax.spines[side].set_linewidth(0.8)


def _background_density(ax, x: np.ndarray, y: np.ndarray, *, extent: tuple[float, float, float, float]) -> None:
    if x.size == 0 or y.size == 0:
        return
    _plt, LinearSegmentedColormap, LogNorm = _publication_imports()
    cmap = LinearSegmentedColormap.from_list(
        "malca_density_blue",
        ["#ffffff", "#d9e7f3", "#96b7d4", "#4f7fa7", "#1f4f75"],
    )
    xmin, xmax, ymin, ymax = extent
    mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
    if not mask.any():
        return
    visible_x = x[mask]
    visible_y = y[mask]

    hist, xedges, yedges = np.histogram2d(
        visible_x,
        visible_y,
        bins=_PUBLICATION_DENSITY_BINS,
        range=((xmin, xmax), (ymin, ymax)),
    )
    density = hist.astype(float, copy=False)
    positive = density[np.isfinite(density) & (density > 0)]
    if positive.size == 0:
        return
    cutoff = max(float(np.nanmax(positive)) * 0.012, float(np.nanpercentile(positive, 8)))
    vmin = max(cutoff, float(np.nanpercentile(positive, 12)))
    vmax = float(np.nanmax(positive))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        _publication_scatter_background(ax, visible_x, visible_y, size=4.8, alpha=0.18)
        return

    occupied = np.isfinite(density) & (density > cutoff)
    if not occupied.any():
        _publication_scatter_background(ax, visible_x, visible_y, size=4.8, alpha=0.18)
        return
    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    x_idx, y_idx = np.nonzero(occupied)
    counts = density[x_idx, y_idx]
    ax.scatter(
        x_centers[x_idx],
        y_centers[y_idx],
        c=counts,
        s=5.5,
        marker="o",
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        alpha=0.95,
        linewidths=0.0,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )


def _save_publication_pdf(fig) -> bytes:
    finalize_publication_figure(fig)
    buf = BytesIO()
    try:
        fig.savefig(
            buf,
            format="pdf",
            dpi=_PUBLICATION_DPI,
            bbox_inches=None,
            metadata={"Creator": "MALCA"},
        )
        return buf.getvalue()
    finally:
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass


def _cmd_publication_pdf(payload: dict, background: dict | None) -> bytes | None:
    coords = _dustmaps_cmd_from_payload(payload)
    if coords is None:
        return None

    bp_rp = float(coords["bp_rp"])
    m_g = float(coords["mg"])
    bp_rp0 = float(coords["cmd_color"])
    m_g0 = float(coords["cmd_mag"])

    plt, _LinearSegmentedColormap, _LogNorm = _publication_imports()
    bg_x, bg_y = _finite_publication_xy(background, "cmd_bprp0", "cmd_mg0")
    xlim = _quantile_limits(bg_x, (bp_rp0, bp_rp), default=(-0.5, 4.5), min_pad=0.18)
    ylim = _quantile_limits(bg_y, (m_g0, m_g), default=(-6.0, 14.0), min_pad=0.45)
    xlim = (max(-0.8, xlim[0]), min(5.0, xlim[1]))
    ylim = (max(-8.0, ylim[0]), min(16.0, ylim[1]))

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_SQUARE)
    _background_density(ax, bg_x, bg_y, extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
    if abs(bp_rp0 - bp_rp) > 0.01 or abs(m_g0 - m_g) > 0.01:
        ax.scatter(
            [bp_rp],
            [m_g],
            s=_PUBLICATION_CANDIDATE_SIZE,
            marker="o",
            facecolors=_CANDIDATE_COLOR,
            edgecolors=_CANDIDATE_EDGE_COLOR,
            linewidths=0.95,
            zorder=5,
        )
        ax.annotate(
            "",
            xy=(bp_rp0, m_g0),
            xytext=(bp_rp, m_g),
            arrowprops=dict(arrowstyle="->", color="#6b7280", linewidth=0.9, shrinkA=4, shrinkB=5),
            zorder=4,
        )
    ax.scatter(
        [bp_rp0],
        [m_g0],
        s=_PUBLICATION_CANDIDATE_SIZE,
        marker="o",
        facecolors=_DEREDDENED_COLOR,
        edgecolors=_CANDIDATE_EDGE_COLOR,
        linewidths=0.95,
        zorder=6,
    )
    ax.text(0.70, 0.18, "main sequence", transform=ax.transAxes, color="#374151", fontsize=7.2)
    ax.text(0.67, 0.79, "giants", transform=ax.transAxes, color="#374151", fontsize=7.2)
    _publication_axes(ax, xlabel=r"$(G_{\rm BP}-G_{\rm RP})_0$", ylabel=r"$M_{G,0}$")
    ax.set_xlim(*xlim)
    ax.set_ylim(ylim[1], ylim[0])
    return _save_publication_pdf(fig)


def _ir_colorcolor_publication_pdf(payload: dict, background: dict | None) -> bytes | None:
    h = _safe_float(payload, "tmass_h")
    k = _safe_float(payload, "tmass_k")
    w1 = _safe_float(payload, "w1")
    w2 = _safe_float(payload, "w2")
    if h is None or k is None or w1 is None or w2 is None:
        return None

    hk_obs = h - k
    w1w2_obs = w1 - w2
    hk_dered = _safe_float(payload, "H_K_dered")
    w1w2_dered = _safe_float(payload, "w1_w2_dered")
    if hk_dered is None or w1w2_dered is None:
        av = _safe_float(payload, "A_v_3d")
        if av is not None and av > 0:
            hk_dered = hk_obs - YSO_DUST_CORRECTION_HK * av
            w1w2_dered = w1w2_obs - YSO_DUST_CORRECTION_W1W2 * av
        else:
            hk_dered = hk_obs
            w1w2_dered = w1w2_obs

    plt, _LinearSegmentedColormap, _LogNorm = _publication_imports()
    bg_x, bg_y = _finite_publication_xy(background, "ir_w1w2", "ir_hk")
    xlim = (-0.5, 2.5)
    ylim = (-0.3, 2.0)

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_SQUARE)
    regions = [
        ("Photospheres", xlim[0], YSO_CLASS_II_W1W2_MIN, ylim[0], ylim[1], "#f3f4f6"),
        ("Transition disk", YSO_CLASS_II_W1W2_MIN, YSO_CLASS_I_W1W2, ylim[0], YSO_CLASS_II_HK, "#fff7d6"),
        ("Class II", YSO_CLASS_II_W1W2_MIN, YSO_CLASS_I_W1W2, YSO_CLASS_II_HK, ylim[1], "#ffedd5"),
        ("Class I", YSO_CLASS_I_W1W2, xlim[1], ylim[0], ylim[1], "#fee2e2"),
    ]
    region_labels = {
        "Photospheres": ((xlim[0] + YSO_CLASS_II_W1W2_MIN) / 2.0, 1.86, "center", "top"),
        "Transition disk": ((YSO_CLASS_II_W1W2_MIN + YSO_CLASS_I_W1W2) / 2.0, 0.13, "center", "center"),
        "Class II": ((YSO_CLASS_II_W1W2_MIN + YSO_CLASS_I_W1W2) / 2.0, 1.86, "center", "top"),
        "Class I": ((YSO_CLASS_I_W1W2 + xlim[1]) / 2.0, 1.86, "center", "top"),
    }
    for label, x0, x1, y0, y1, color in regions:
        ax.axvspan(x0, x1, ymin=(y0 - ylim[0]) / (ylim[1] - ylim[0]), ymax=(y1 - ylim[0]) / (ylim[1] - ylim[0]), color=color, zorder=0)
    for label, (x, y, ha, va) in region_labels.items():
        ax.text(
            x,
            y,
            label,
            ha=ha,
            va=va,
            fontsize=7.2,
            color="#374151",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, boxstyle="round,pad=0.12"),
            zorder=2,
        )
    _background_density(ax, bg_x, bg_y, extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
    if abs(hk_dered - hk_obs) > 0.01 or abs(w1w2_dered - w1w2_obs) > 0.01:
        ax.scatter(
            [w1w2_obs],
            [hk_obs],
            s=_PUBLICATION_CANDIDATE_SIZE,
            marker="o",
            facecolors=_CANDIDATE_COLOR,
            edgecolors=_CANDIDATE_EDGE_COLOR,
            linewidths=0.95,
            zorder=5,
        )
        ax.annotate(
            "",
            xy=(w1w2_dered, hk_dered),
            xytext=(w1w2_obs, hk_obs),
            arrowprops=dict(arrowstyle="->", color="#6b7280", linewidth=0.9, shrinkA=4, shrinkB=5),
            zorder=4,
        )
    ax.scatter(
        [w1w2_dered],
        [hk_dered],
        s=_PUBLICATION_CANDIDATE_SIZE,
        marker="o",
        facecolors=_DEREDDENED_COLOR,
        edgecolors=_CANDIDATE_EDGE_COLOR,
        linewidths=0.95,
        zorder=6,
    )
    _publication_axes(ax, xlabel=LABEL_W1_W2, ylabel=LABEL_H_KS)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return _save_publication_pdf(fig)


def _kiel_publication_pdf(payload: dict, background: dict | None) -> bytes | None:
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

    plt, _LinearSegmentedColormap, _LogNorm = _publication_imports()
    bg_x, bg_y = _finite_publication_xy(background, "kiel_teff", "kiel_logg")
    xlim = _quantile_limits(bg_x, (teff,), default=(3200.0, 18000.0), qlo=0.005, qhi=0.995, min_pad=350.0)
    ylim = _quantile_limits(bg_y, (logg,), default=(0.0, 5.2), qlo=0.005, qhi=0.995, min_pad=0.25)
    xlim = (max(2500.0, xlim[0]), min(40000.0, max(xlim[1], min(18000.0, float(teff) * 1.08))))
    ylim = (max(-0.5, ylim[0]), min(6.0, ylim[1]))

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_SQUARE)
    _background_density(ax, bg_x, bg_y, extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
    xerr = None
    yerr = None
    if teff16 is not None and teff84 is not None:
        xerr = [[max(0.0, teff - teff16)], [max(0.0, teff84 - teff)]]
    if logg16 is not None and logg84 is not None:
        yerr = [[max(0.0, logg - logg16)], [max(0.0, logg84 - logg)]]
    ax.errorbar(
        [teff],
        [logg],
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        markersize=5.8,
        markerfacecolor=_CANDIDATE_COLOR,
        markeredgecolor=_CANDIDATE_EDGE_COLOR,
        markeredgewidth=0.95,
        ecolor="#374151",
        elinewidth=0.8,
        capsize=2.5,
        zorder=6,
    )
    ax.text(0.77, 0.14, "main sequence", transform=ax.transAxes, ha="center", color="#374151", fontsize=7.2)
    ax.text(0.18, 0.68, "giants", transform=ax.transAxes, ha="center", color="#374151", fontsize=7.2)
    ax.text(0.70, 0.45, "subgiants", transform=ax.transAxes, ha="center", color="#374151", fontsize=7.2)
    _publication_axes(ax, xlabel=r"$T_{\rm eff}\ {\rm [K]}$", ylabel=r"$\log g\ {\rm [cgs]}$")
    ax.set_xlim(xlim[1], xlim[0])
    ax.set_ylim(ylim[1], ylim[0])
    return _save_publication_pdf(fig)


def _teff_sed_alpha_publication_pdf(payload: dict, background: dict | None) -> bytes | None:
    teff = _safe_float(payload, "teff50")
    if teff is None:
        teff = _safe_float(payload, "teff_gspphot")
    alpha = _safe_float(payload, "sed_alpha")
    if teff is None or alpha is None:
        return None

    plt, _LinearSegmentedColormap, _LogNorm = _publication_imports()
    bg_x, bg_y = _finite_publication_xy(background, "plane_teff_alpha_x", "plane_teff_alpha_y")
    xlim = _quantile_limits(bg_x, (teff,), default=(2500.0, 12000.0), qlo=0.005, qhi=0.995, min_pad=350.0)
    ylim = _quantile_limits(bg_y, (alpha, -1.6, -0.3, 0.3), default=(-3.0, 2.0), qlo=0.005, qhi=0.995, min_pad=0.35)

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_SQUARE)
    _background_density(ax, bg_x, bg_y, extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
    for value in (-1.6, -0.3, 0.3):
        ax.axhline(value, color="#6b7280", linestyle=":", linewidth=0.8, zorder=2)
    ax.scatter(
        [teff],
        [alpha],
        s=_PUBLICATION_CANDIDATE_SIZE,
        marker="o",
        facecolors=_CANDIDATE_COLOR,
        edgecolors=_CANDIDATE_EDGE_COLOR,
        linewidths=0.95,
        zorder=6,
    )
    _publication_axes(ax, xlabel=r"$T_{\rm eff}\ {\rm [K]}$", ylabel=r"$\alpha_{2-24\mu{\rm m}}$")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return _save_publication_pdf(fig)


def _sample_publication_points(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= _BACKGROUND_POINT_LIMIT:
        return x, y
    idx = np.linspace(0, x.size - 1, _BACKGROUND_POINT_LIMIT, dtype=int)
    return x[idx], y[idx]


def _publication_scatter_background(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str = "#4f7fa7",
    size: float = 5.0,
    alpha: float = 0.18,
) -> None:
    if x.size == 0 or y.size == 0:
        return
    x, y = _sample_publication_points(x, y)
    ax.scatter(
        x,
        y,
        s=size,
        c=color,
        alpha=alpha,
        edgecolors="none",
        linewidths=0.0,
        rasterized=True,
        zorder=2,
    )


def _publication_candidate_marker(
    ax,
    x: float,
    y: float,
    *,
    color: str = _CANDIDATE_COLOR,
    size: float = _PUBLICATION_CANDIDATE_SIZE,
) -> None:
    ax.scatter(
        [x],
        [y],
        s=size,
        marker="o",
        facecolors=color,
        edgecolors=_CANDIDATE_EDGE_COLOR,
        linewidths=0.95,
        zorder=6,
    )


def _positive_log_limits(
    values: np.ndarray,
    extras: tuple[float | None, ...] = (),
    *,
    default: tuple[float, float],
    qlo: float = 0.01,
    qhi: float = 0.99,
    pad_fraction: float = 0.08,
) -> tuple[float, float]:
    positive = [float(v) for v in np.asarray(values, dtype=float).reshape(-1) if np.isfinite(v) and v > 0]
    if positive:
        logs = np.log10(np.asarray(positive, dtype=float))
        lo = float(np.quantile(logs, qlo))
        hi = float(np.quantile(logs, qhi))
    else:
        lo = math.log10(default[0])
        hi = math.log10(default[1])
    for extra in extras:
        if extra is None:
            continue
        try:
            value = float(extra)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            log_value = math.log10(value)
            lo = min(lo, log_value)
            hi = max(hi, log_value)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = math.log10(default[0])
        hi = math.log10(default[1])
    pad = max(0.08, (hi - lo) * pad_fraction)
    return 10 ** (lo - pad), 10 ** (hi + pad)


def _metric_publication_limits(
    x: np.ndarray,
    y: np.ndarray,
    candidate_x: float,
    candidate_y: float,
    *,
    default: tuple[float, float],
    fixed: tuple[float, float] | None = None,
    nonnegative: bool = False,
) -> tuple[float, float]:
    if fixed is not None:
        return fixed
    both = np.concatenate([np.asarray(x, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1)])
    lo, hi = _quantile_limits(
        both,
        (candidate_x, candidate_y),
        default=default,
        qlo=0.005,
        qhi=0.995,
        min_pad=0.2,
    )
    if nonnegative:
        lo = min(-0.08 * max(hi, 1.0), lo)
    return lo, hi


def _score_balance_publication_pdf(payload: dict, background: dict | None) -> bytes | None:
    dipper_score = _safe_float(payload, "dipper_score")
    jumper_score = _safe_float(payload, "jumper_score")
    if dipper_score is None or jumper_score is None:
        return None

    plt, _LinearSegmentedColormap, _LogNorm = _publication_imports()
    bg_x, bg_y = _finite_publication_xy(background, "metric_dipper_score", "metric_jumper_score")
    upper_lo, upper_hi = _metric_publication_limits(
        bg_x,
        bg_y,
        dipper_score,
        jumper_score,
        default=(0.0, 6.0),
        nonnegative=True,
    )
    upper = max(6.0, upper_hi)
    lower = min(-0.25, upper_lo)

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_SQUARE)
    _publication_scatter_background(ax, bg_x, bg_y)
    ax.plot([0.0, upper], [0.0, upper], color="#6b7280", linewidth=0.75, linestyle=":", zorder=3)
    ax.text(0.73, 0.20, "dip-like", transform=ax.transAxes, fontsize=7.2, color="#374151")
    ax.text(0.18, 0.78, "jump-like", transform=ax.transAxes, fontsize=7.2, color="#374151")
    _publication_candidate_marker(ax, dipper_score, jumper_score)
    _publication_axes(ax, xlabel=r"$S_{\rm dip}$", ylabel=r"$S_{\rm jump}$")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    return _save_publication_pdf(fig)


def _catalog_support_publication_pdf(payload: dict, background: dict | None) -> bytes | None:
    period_sources = _safe_float(payload, "period_n_sources")
    dip_runs = _safe_float(payload, "dip_run_count")
    if period_sources is None or dip_runs is None:
        return None

    plt, _LinearSegmentedColormap, _LogNorm = _publication_imports()
    bg_x, bg_y = _finite_publication_xy(background, "plane_catalog_support_x", "plane_catalog_support_y")

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_SQUARE)
    _publication_scatter_background(ax, bg_x, bg_y)
    ax.axvline(2.0, color="#6b7280", linewidth=0.75, linestyle=":", zorder=3)
    ax.axhline(2.0, color="#6b7280", linewidth=0.75, linestyle=":", zorder=3)
    ax.text(0.57, 0.82, "strong catalog support", transform=ax.transAxes, fontsize=7.1, color="#374151")
    ax.text(0.58, 0.18, "repeated dips", transform=ax.transAxes, fontsize=7.1, color="#374151")
    _publication_candidate_marker(ax, period_sources, dip_runs)
    _publication_axes(ax, xlabel=r"$N_{\rm period\ sources}$", ylabel=r"$N_{\rm dip\ runs}$")
    ax.set_xlim(-0.25, 5.25)
    ax.set_ylim(-0.25, 8.25)
    return _save_publication_pdf(fig)


def _recurrence_regularity_publication_pdf(payload: dict, background: dict | None) -> bytes | None:
    spacing_median = _safe_float(payload, "dip_inter_event_spacing_median")
    spacing_std = _safe_float(payload, "dip_inter_event_spacing_std")
    if spacing_median is None or spacing_std is None or spacing_median <= 0 or spacing_std <= 0:
        return None

    plt, _LinearSegmentedColormap, _LogNorm = _publication_imports()
    bg_x, bg_y = _finite_publication_xy(background, "plane_recurrence_regularity_x", "plane_recurrence_regularity_y")
    mask = (bg_x > 0) & (bg_y > 0)
    bg_x = bg_x[mask]
    bg_y = bg_y[mask]
    xlim = _positive_log_limits(bg_x, (spacing_median,), default=(1.0, 1000.0))
    ylim = _positive_log_limits(bg_y, (spacing_std,), default=(0.5, 1000.0))

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_SQUARE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _publication_scatter_background(ax, bg_x, bg_y)
    diag_lo = max(min(xlim[0], ylim[0]), 1e-6)
    diag_hi = max(xlim[1], ylim[1])
    ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color="#6b7280", linewidth=0.75, linestyle=":", zorder=3)
    ax.text(0.17, 0.82, "less regular", transform=ax.transAxes, fontsize=7.1, color="#374151")
    ax.text(0.62, 0.22, "more regular", transform=ax.transAxes, fontsize=7.1, color="#374151")
    _publication_candidate_marker(ax, spacing_median, spacing_std)
    _publication_axes(
        ax,
        xlabel=r"$\tilde{\Delta t}_{\rm dip}\ [{\rm d}]$",
        ylabel=r"$\sigma_{\Delta t}\ [{\rm d}]$",
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return _save_publication_pdf(fig)


def _dip_repeatability_publication_pdf(payload: dict, background: dict | None) -> bytes | None:
    amp_consistency = _safe_float(payload, "dip_amplitude_consistency")
    duration_consistency = _safe_float(payload, "dip_duration_consistency")
    if amp_consistency is None or duration_consistency is None:
        return None

    plt, _LinearSegmentedColormap, _LogNorm = _publication_imports()
    bg_x, bg_y = _finite_publication_xy(background, "plane_dip_repeatability_x", "plane_dip_repeatability_y")

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_SQUARE)
    _publication_scatter_background(ax, bg_x, bg_y)
    ax.axvline(0.5, color="#6b7280", linewidth=0.75, linestyle=":", zorder=3)
    ax.axhline(0.5, color="#6b7280", linewidth=0.75, linestyle=":", zorder=3)
    ax.text(0.58, 0.82, "repeatable dips", transform=ax.transAxes, fontsize=7.1, color="#374151")
    _publication_candidate_marker(ax, amp_consistency, duration_consistency)
    _publication_axes(ax, xlabel=r"$C_{\rm amp}$", ylabel=r"$C_{\rm duration}$")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    return _save_publication_pdf(fig)


def _variability_strength_publication_pdf(payload: dict, background: dict | None) -> bytes | None:
    robust_sigma = _safe_float(payload, "stats_photometry_robust_sigma_mag")
    dipper_score = _safe_float(payload, "dipper_score")
    if robust_sigma is None or robust_sigma <= 0 or dipper_score is None:
        return None

    plt, _LinearSegmentedColormap, _LogNorm = _publication_imports()
    bg_x, bg_y = _finite_publication_xy(background, "plane_var_strength_x", "plane_var_strength_y")
    mask = bg_x > 0
    bg_x = bg_x[mask]
    bg_y = bg_y[mask]
    xlim = _positive_log_limits(bg_x, (robust_sigma,), default=(0.003, 2.0))
    ylim = _quantile_limits(bg_y, (dipper_score, 5.0), default=(0.0, 20.0), min_pad=0.8)
    ylim = (min(-0.5, ylim[0]), ylim[1])

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_SQUARE)
    ax.set_xscale("log")
    _publication_scatter_background(ax, bg_x, bg_y)
    ax.axhline(5.0, color="#6b7280", linewidth=0.75, linestyle=":", zorder=3)
    _publication_candidate_marker(ax, robust_sigma, dipper_score)
    _publication_axes(ax, xlabel=r"$\sigma_{\rm robust}\ [{\rm mag}]$", ylabel=r"$S_{\rm dip}$")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return _save_publication_pdf(fig)


def _stetson_scatter_publication_pdf(payload: dict, background: dict | None) -> bytes | None:
    robust_sigma = _safe_float(payload, "stats_photometry_robust_sigma_mag")
    stetson_j = _safe_float(payload, "stats_variability_stetson_J")
    if robust_sigma is None or robust_sigma <= 0 or stetson_j is None:
        return None

    plt, _LinearSegmentedColormap, _LogNorm = _publication_imports()
    bg_x, bg_y = _finite_publication_xy(background, "plane_stetson_x", "plane_stetson_y")
    mask = bg_x > 0
    bg_x = bg_x[mask]
    bg_y = bg_y[mask]
    xlim = _positive_log_limits(bg_x, (robust_sigma,), default=(0.003, 2.0))
    ylim = _quantile_limits(bg_y, (stetson_j,), default=(-0.2, 5.0), min_pad=0.35)

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_SQUARE)
    ax.set_xscale("log")
    _publication_scatter_background(ax, bg_x, bg_y)
    _publication_candidate_marker(ax, robust_sigma, stetson_j)
    _publication_axes(ax, xlabel=r"$\sigma_{\rm robust}\ [{\rm mag}]$", ylabel=r"${\rm Stetson}\ J$")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return _save_publication_pdf(fig)


def _shape_impulsiveness_publication_pdf(payload: dict, background: dict | None) -> bytes | None:
    skew = _safe_float(payload, "stats_skew")
    max_slope = _safe_float(payload, "stats_max_slope")
    if skew is None or max_slope is None or max_slope <= 0:
        return None

    plt, _LinearSegmentedColormap, _LogNorm = _publication_imports()
    bg_x, bg_y = _finite_publication_xy(background, "plane_shape_x", "plane_shape_y")
    mask = bg_y > 0
    bg_x = bg_x[mask]
    bg_y = bg_y[mask]
    xlim = _quantile_limits(bg_x, (skew,), default=(-2.5, 3.0), min_pad=0.25)
    ylim = _positive_log_limits(bg_y, (max_slope,), default=(0.02, 1.0e5))

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_SQUARE)
    ax.set_yscale("log")
    _publication_scatter_background(ax, bg_x, bg_y)
    _publication_candidate_marker(ax, skew, max_slope)
    _publication_axes(ax, xlabel=r"${\rm Skew}$", ylabel=r"${\rm Max\ slope}$")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return _save_publication_pdf(fig)


def _rpm_publication_pdf(payload: dict, background: dict | None) -> bytes | None:
    g_mag = _safe_float(payload, "phot_g_mean_mag")
    pmra = _safe_float(payload, "pmra")
    pmdec = _safe_float(payload, "pmdec")
    if g_mag is None or pmra is None or pmdec is None:
        return None

    bp_rp = _safe_float(payload, "bp_rp")
    if bp_rp is None:
        bp = _safe_float(payload, "phot_bp_mean_mag")
        rp = _safe_float(payload, "phot_rp_mean_mag")
        if bp is None or rp is None:
            return None
        bp_rp = bp - rp

    pm_total = math.sqrt(pmra**2 + pmdec**2)
    if pm_total <= 0:
        return None
    h_g = g_mag + 5.0 * math.log10(pm_total / 1000.0) + 5.0

    plt, _LinearSegmentedColormap, _LogNorm = _publication_imports()
    bg_x, bg_y = _finite_publication_xy(background, "rpm_bprp", "rpm_hg")
    xlim = _quantile_limits(bg_x, (bp_rp,), default=(-0.4, 4.5), min_pad=0.2)
    ylim = _quantile_limits(bg_y, (h_g,), default=(0.0, 18.0), min_pad=0.7)
    xlim = (max(-1.0, xlim[0]), min(5.5, xlim[1]))
    ylim = (max(-5.0, ylim[0]), min(22.0, ylim[1]))

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_SQUARE)
    _publication_scatter_background(ax, bg_x, bg_y)
    _publication_candidate_marker(ax, bp_rp, h_g)
    ax.text(0.25, 0.24, "main sequence", transform=ax.transAxes, color="#374151", fontsize=7.2)
    ax.text(0.65, 0.76, "giants", transform=ax.transAxes, color="#374151", fontsize=7.2)
    _publication_axes(ax, xlabel=r"$G_{\rm BP}-G_{\rm RP}$", ylabel=r"$H_G$")
    ax.set_xlim(*xlim)
    ax.set_ylim(ylim[1], ylim[0])
    return _save_publication_pdf(fig)


def build_publication_diagnostic_pdf(
    plot_name: str,
    payload: dict,
    background: dict | None = None,
) -> bytes | None:
    """Render selected diagnostic plots as purpose-built Matplotlib PDFs."""
    key = str(plot_name or "").strip().lower().replace("-", "_")
    if key == "cmd":
        return _cmd_publication_pdf(payload, background)
    if key == "ir_colorcolor":
        return _ir_colorcolor_publication_pdf(payload, background)
    if key == "kiel":
        return _kiel_publication_pdf(payload, background)
    if key == "teff_sed_alpha":
        return _teff_sed_alpha_publication_pdf(payload, background)
    if key == "rpm":
        return _rpm_publication_pdf(payload, background)
    if key in {"score_balance", "morphology_scores"}:
        return _score_balance_publication_pdf(payload, background)
    if key in {"catalog_support", "catalog_support_vs_dip_recurrence"}:
        return _catalog_support_publication_pdf(payload, background)
    if key == "recurrence_regularity":
        return _recurrence_regularity_publication_pdf(payload, background)
    if key == "dip_repeatability":
        return _dip_repeatability_publication_pdf(payload, background)
    if key == "variability_strength":
        return _variability_strength_publication_pdf(payload, background)
    if key == "stetson_scatter":
        return _stetson_scatter_publication_pdf(payload, background)
    if key == "shape_impulsiveness":
        return _shape_impulsiveness_publication_pdf(payload, background)
    return None


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

    bg_x, bg_y = _finite_xy(background, "rpm_bprp", "rpm_hg")
    if bg_x.size > 0:
        fig.add_trace(go.Scattergl(
            x=bg_x, y=bg_y,
            mode="markers",
            marker=dict(size=2, color=spec["muted"], opacity=0.25),
            hoverinfo="skip",
            name="Sample",
        ))

    fig.add_trace(go.Scattergl(
        x=[bp_rp], y=[h_g],
        mode="markers",
        marker=_plotly_candidate_marker(),
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
    x_values = [bp_rp] + (bg_x.tolist() if bg_x.size > 0 else [])
    y_values = [h_g] + (bg_y.tolist() if bg_y.size > 0 else [])
    y_range = _axis_range_with_padding(y_values, pad_fraction=0.08, min_pad=0.6)
    fig.update_xaxes(title="BP - RP", range=_axis_range_with_padding(x_values, pad_fraction=0.08, min_pad=0.2))
    fig.update_yaxes(title="H<sub>G</sub>", range=[y_range[1], y_range[0]] if y_range else None)

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

    bg_x, bg_y = _finite_xy(background, "uv_bprp", "uv_nuv_g")
    if bg_x.size > 0:
        fig.add_trace(go.Scattergl(
            x=bg_x, y=bg_y,
            mode="markers",
            marker=dict(size=2, color=spec["muted"], opacity=0.25),
            hoverinfo="skip",
            name="Sample",
        ))

    fig.add_trace(go.Scattergl(
        x=[bp_rp], y=[nuv_g],
        mode="markers",
        marker=_plotly_candidate_marker(),
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
    x_values = [bp_rp] + (bg_x.tolist() if bg_x.size > 0 else [])
    y_values = [nuv_g] + (bg_y.tolist() if bg_y.size > 0 else [])
    fig.update_xaxes(title="BP - RP", range=_axis_range_with_padding(x_values, pad_fraction=0.08, min_pad=0.2))
    fig.update_yaxes(title="NUV - G", range=_axis_range_with_padding(y_values, pad_fraction=0.08, min_pad=0.6))

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
        marker=_plotly_candidate_marker(),
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
        marker=_plotly_candidate_marker(),
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
        x_title="Dip spacing median [d]",
        y_title="Dip spacing std [d]",
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
        x_title="Robust sigma [mag]",
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
        x_title="Robust sigma [mag]",
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
        x_title="LTV slope [mag/yr]",
        y_title="LTV dispersion [mag]",
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
        x_title="W1 slope [mag/yr]",
        y_title="W1-W2 slope [mag/yr]",
    )
