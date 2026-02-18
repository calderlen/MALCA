"""Interactive light-curve plotting for the review GUI."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from malca.baseline import global_median_baseline, per_camera_gp_baseline, per_camera_median_baseline
from malca.plot import _stable_camera_color, load_lightcurve_df
from malca.utils import (
    clean_lc,
    identify_bad_cameras,
    identify_catastrophic_outlier_cameras,
    identify_offset_cameras,
)


BASELINE_FUNCTIONS = {
    "global_median": global_median_baseline,
    "per_camera_median": per_camera_median_baseline,
    "per_camera_gp": per_camera_gp_baseline,
}

REQUIRED_COLUMNS = {"JD", "mag", "v_g_band"}

# Keep plotting caches bounded; large values can inflate long-running GUI memory.
_CACHE_LIMIT = 16
_CLEAN_CACHE: OrderedDict[tuple, tuple[pd.DataFrame, set[int], dict[str, list[str]]]] = OrderedDict()
_BASELINE_CACHE: OrderedDict[tuple, dict[int, pd.DataFrame]] = OrderedDict()
_EVENT_CACHE: OrderedDict[tuple, list[dict[str, object]]] = OrderedDict()


def _cache_get(cache: OrderedDict, key):
    value = cache.get(key)
    if value is None:
        return None
    cache.move_to_end(key)
    return value


def _cache_put(cache: OrderedDict, key, value) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_LIMIT:
        cache.popitem(last=False)


def resolve_lightcurve_path(payload: dict, plot_dir: Path | None) -> Path | None:
    """Resolve a candidate light-curve path for native plotting."""
    keys = ("path", "lc_path")
    for key in keys:
        raw_path = payload.get(key)
        if not raw_path:
            continue
        candidate = Path(str(raw_path)).expanduser()
        if candidate.exists():
            return candidate
        if plot_dir is None:
            continue
        bundle_candidate = plot_dir.parent / "bundle_assets" / "lightcurves" / candidate.name
        if bundle_candidate.exists():
            return bundle_candidate
    return None


def _camera_labels(df: pd.DataFrame) -> pd.Series:
    if "camera_name" in df.columns:
        return pd.Series(df["camera_name"].astype(str), index=df.index)
    if "camera#" in df.columns:
        return pd.Series(df["camera#"].astype(str), index=df.index)
    if "camera" in df.columns:
        return pd.Series(df["camera"].astype(str), index=df.index)
    return pd.Series(["unknown"] * len(df), index=df.index)


def _parse_num(payload: dict, key: str) -> float | None:
    val = payload.get(key)
    if val is None:
        return None
    try:
        f = float(val)
    except Exception:
        return None
    if not np.isfinite(f):
        return None
    return f


def _get_camera_reason_diagnostics(df: pd.DataFrame, scatter_ratio: float) -> dict[str, list[str]]:
    """Return camera reason tags used for explainable filtering."""
    if df.empty or "camera#" not in df.columns:
        return {}
    diagnostics: dict[str, set[str]] = {}
    try:
        scatter_bad = identify_bad_cameras(df, scatter_ratio_threshold=scatter_ratio)
    except Exception:
        scatter_bad = set()
    try:
        offset_bad, _ = identify_offset_cameras(df, offset_sigma_threshold=15.0, remove_full_camera=True)
    except Exception:
        offset_bad = set()
    try:
        catastrophic_bad = identify_catastrophic_outlier_cameras(df)
    except Exception:
        catastrophic_bad = set()

    for cam in scatter_bad:
        diagnostics.setdefault(str(cam), set()).add("scatter")
    for cam in offset_bad:
        diagnostics.setdefault(str(cam), set()).add("offset")
    for cam in catastrophic_bad:
        diagnostics.setdefault(str(cam), set()).add("catastrophic")

    return {cam: sorted(tags) for cam, tags in diagnostics.items()}


def _load_cleaned_df(
    lc_path: Path,
    *,
    filter_bad_cameras: bool,
    scatter_ratio: float,
    clean_max_error_absolute: float,
    clean_max_error_sigma: float,
) -> tuple[pd.DataFrame, set[int], dict[str, list[str]]]:
    key = (
        str(lc_path.resolve()),
        bool(filter_bad_cameras),
        float(scatter_ratio),
        float(clean_max_error_absolute),
        float(clean_max_error_sigma),
    )
    cached = _cache_get(_CLEAN_CACHE, key)
    if cached is not None:
        cdf, cams, diag = cached
        return cdf.copy(), set(cams), dict(diag)

    df, filtered_cameras = load_lightcurve_df(
        lc_path,
        filter_bad_cameras_enabled=filter_bad_cameras,
        bad_camera_scatter_ratio=scatter_ratio,
        return_filtered_info=True,
    )
    diagnostics = _get_camera_reason_diagnostics(df, scatter_ratio)
    df = clean_lc(df, max_error_absolute=clean_max_error_absolute, max_error_sigma=clean_max_error_sigma)
    _cache_put(_CLEAN_CACHE, key, (df.copy(), set(filtered_cameras), diagnostics))
    return df, set(filtered_cameras), diagnostics


def _compute_baseline_bands(df: pd.DataFrame, baseline_name: str, cache_key: tuple) -> dict[int, pd.DataFrame]:
    key = (cache_key, baseline_name)
    cached = _cache_get(_BASELINE_CACHE, key)
    if cached is not None:
        return {k: v.copy() for k, v in cached.items()}

    baseline_func = BASELINE_FUNCTIONS.get(baseline_name, per_camera_gp_baseline)
    baseline_kwargs = {}
    if baseline_func is per_camera_gp_baseline:
        baseline_kwargs["add_sigma_eff_col"] = True

    band_dfs: dict[int, pd.DataFrame] = {}
    for band in (0, 1):
        bdf = df[df["v_g_band"] == band].copy()
        if bdf.empty:
            continue
        try:
            out = baseline_func(bdf, **baseline_kwargs)
            if "baseline" in out.columns:
                bdf["baseline"] = out["baseline"].to_numpy()
                bdf["resid"] = bdf["mag"] - bdf["baseline"]
            else:
                bdf["baseline"] = np.nan
                bdf["resid"] = np.nan
        except Exception:
            bdf["baseline"] = np.nan
            bdf["resid"] = np.nan
        band_dfs[band] = bdf

    _cache_put(_BASELINE_CACHE, key, {k: v.copy() for k, v in band_dfs.items()})
    return band_dfs


def _build_title(payload: dict, df: pd.DataFrame) -> str:
    asas_sn_id = str(payload.get("asas_sn_id") or "").strip()
    source_name = str(payload.get("source") or "").strip()
    vsx_class = str(payload.get("vsx_class") or "").strip()
    category = str(payload.get("category") or "").strip()

    if source_name and asas_sn_id:
        label = f"{source_name} ({asas_sn_id})"
    elif asas_sn_id:
        label = asas_sn_id
    elif source_name:
        label = source_name
    else:
        label = "Source"

    parts = [label]
    if vsx_class:
        parts.append(f"VSX: {vsx_class}")
    if category:
        parts.append(category)
    if not df.empty and "JD" in df.columns:
        parts.append(f"JD {float(df['JD'].min()):.0f}-{float(df['JD'].max()):.0f}")

    dip_runs = _parse_num(payload, "dip_run_count")
    jump_runs = _parse_num(payload, "jump_run_count")
    if dip_runs is not None and dip_runs > 0:
        parts.append(f"Dips: {int(dip_runs)}")
    if jump_runs is not None and jump_runs > 0:
        parts.append(f"Jumps: {int(jump_runs)}")
    return " - ".join(parts)


def _build_stat_rows(payload: dict, df: pd.DataFrame, filtered_cameras: set[int]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("Points", f"{len(df):,}"),
        ("Cameras", str(int(df["camera_label"].nunique())) if "camera_label" in df.columns else "0"),
    ]
    for label, key, fmt in (
        ("Dipper score", "dipper_score", "{:.2f}"),
        ("Jumper score", "jumper_score", "{:.2f}"),
        ("Dip logBF", "dip_bayes_factor", "{:.1f}"),
        ("Jump logBF", "jump_bayes_factor", "{:.1f}"),
        ("RUWE", "ruwe", "{:.2f}"),
        ("Periodicity", "periodicity_score", "{:.3f}"),
        ("Phase P (d)", "phase_period_days", "{:.5f}"),
        ("Phase quality", "phase_quality_score", "{:.3f}"),
    ):
        value = _parse_num(payload, key)
        if value is not None:
            rows.append((label, fmt.format(value)))
    if filtered_cameras:
        rows.append(("Filtered cams", ",".join(str(c) for c in sorted(filtered_cameras))))
    return rows


def _event_thresholds(run_params: dict | None) -> dict[str, float | None]:
    """Extract event-related thresholds from run params."""
    if not run_params:
        return {"dip_logbf": None, "jump_logbf": None, "sig": None}
    dip_thr = run_params.get("logbf_threshold_dip")
    jump_thr = run_params.get("logbf_threshold_jump")
    sig_thr = run_params.get("significance_threshold")
    try:
        dip_thr = float(dip_thr) if dip_thr is not None else None
    except Exception:
        dip_thr = None
    try:
        jump_thr = float(jump_thr) if jump_thr is not None else None
    except Exception:
        jump_thr = None
    try:
        sig_thr = float(sig_thr) if sig_thr is not None else None
    except Exception:
        sig_thr = None
    return {"dip_logbf": dip_thr, "jump_logbf": jump_thr, "sig": sig_thr}


def _event_entries(payload: dict, jd_offset: float, run_params: dict | None) -> list[dict[str, object]]:
    thresholds = _event_thresholds(run_params)
    key = (
        _parse_num(payload, "dip_best_t0"),
        _parse_num(payload, "jump_best_t0"),
        _parse_num(payload, "dip_best_width_param"),
        _parse_num(payload, "jump_best_width_param"),
        _parse_num(payload, "dip_bayes_factor"),
        _parse_num(payload, "jump_bayes_factor"),
        str(payload.get("dip_best_morph") or ""),
        str(payload.get("jump_best_morph") or ""),
        thresholds["dip_logbf"],
        thresholds["jump_logbf"],
        thresholds["sig"],
        jd_offset,
    )
    cached = _cache_get(_EVENT_CACHE, key)
    if cached is not None:
        return [dict(x) for x in cached]

    entries: list[dict[str, object]] = []
    for prefix, color in (("dip", "#ff6b6b"), ("jump", "#55d66d")):
        t0 = _parse_num(payload, f"{prefix}_best_t0")
        if t0 is None:
            continue
        width = _parse_num(payload, f"{prefix}_best_width_param")
        bf = _parse_num(payload, f"{prefix}_bayes_factor")
        morph = str(payload.get(f"{prefix}_best_morph") or "")
        logbf_threshold = thresholds["dip_logbf"] if prefix == "dip" else thresholds["jump_logbf"]
        conf_base = logbf_threshold if logbf_threshold is not None else 3.0
        confidence = 0.0 if bf is None else float(np.clip((bf - conf_base) / max(conf_base, 8.0), 0.0, 1.0))
        approx_half_width = 0.0 if width is None else max(width * 2.0, 0.25)
        entries.append(
            {
                "kind": prefix,
                "t0": t0,
                "x0": t0 - jd_offset,
                "half_width": approx_half_width,
                "bf": bf,
                "morph": morph,
                "confidence": confidence,
                "base_color": color,
                "logbf_threshold": logbf_threshold,
                "sig_threshold": thresholds["sig"],
            }
        )

    _cache_put(_EVENT_CACHE, key, [dict(x) for x in entries])
    return entries


def _phase_period_days(payload: dict) -> float | None:
    """Return preferred phase-fold period from payload metadata."""
    phase_period = _parse_num(payload, "phase_period_days")
    if phase_period is None:
        phase_period = _parse_num(payload, "lsp_period")
    if phase_period is None or (not np.isfinite(phase_period)) or phase_period <= 0:
        return None
    return float(phase_period)


def _phase_fold_df(df: pd.DataFrame, period_days: float) -> pd.DataFrame:
    """Phase-fold dataframe to 0-2 cycles."""
    out = df.copy()
    out = out[np.isfinite(out["JD"]) & np.isfinite(out["mag"])].copy()
    if out.empty:
        return out
    jd0 = float(out["JD"].min())
    out["phase"] = ((out["JD"].to_numpy(dtype=float) - jd0) / float(period_days)) % 1.0
    wrap = out.copy()
    wrap["phase"] = wrap["phase"] + 1.0
    return pd.concat([out, wrap], ignore_index=True)


def _theme_palette(theme: str) -> dict[str, str]:
    mode = str(theme or "dark").lower()
    if mode == "solarized":
        return {
            "text": "#586e75",
            "title": "#073642",
            "paper_bg": "#fdf6e3",
            "plot_bg": "#fdf6e3",
            "grid": "rgba(88,110,117,0.22)",
            "legend_bg": "rgba(238,232,213,0.94)",
            "legend_border": "rgba(88,110,117,0.35)",
            "annotation": "#586e75",
            "marker_line": "rgba(88,110,117,0.85)",
            "guide_line": "rgba(88,110,117,0.40)",
        }
    return {
        "text": "#dce5ef",
        "title": "#dce5ef",
        "paper_bg": "rgba(0,0,0,0)",
        "plot_bg": "rgba(0,0,0,0)",
        "grid": "rgba(96,116,130,0.25)",
        "legend_bg": "rgba(0,0,0,0.22)",
        "legend_border": "rgba(113,140,160,0.35)",
        "annotation": "#bcd0e1",
        "marker_line": "rgba(10,10,10,0.95)",
        "guide_line": "rgba(210,210,210,0.35)",
    }


def _status_figure(message: str, theme: str = "dark") -> go.Figure:
    colors = _theme_palette(theme)
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        font={"size": 13, "color": colors["text"]},
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
    )
    fig.update_layout(
        paper_bgcolor=colors["paper_bg"],
        plot_bgcolor=colors["plot_bg"],
        margin={"l": 40, "r": 20, "t": 38, "b": 30},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def build_interactive_lightcurve_figure(
    payload: dict,
    *,
    plot_dir: Path | None,
    selected_cameras: list[str] | None,
    filter_bad_cameras: bool,
    show_baseline: bool,
    show_event_markers: bool,
    show_residuals: bool,
    show_phase_fold: bool = False,
    show_diagnostics: bool,
    confidence_colors: bool,
    run_params: dict | None,
    uirevision_key: str,
    theme: str = "dark",
    residual_fraction: float = 0.28,
    baseline_opacity: float = 0.5,
) -> dict:
    """Build a native Plotly light-curve figure for review mode."""
    colors = _theme_palette(theme)

    plot_dir = Path(plot_dir) if plot_dir else None
    lc_path = resolve_lightcurve_path(payload, plot_dir)
    if lc_path is None:
        return {
            "figure": _status_figure("No light-curve file found. Try PNG mode or check bundle_assets/lightcurves.", theme=theme),
            "camera_options": [],
            "camera_values": [],
            "stat_rows": [],
            "status": "missing-file",
            "status_message": "Missing light-curve file. Check candidate path or imported bundle assets.",
            "camera_diagnostics": {},
            "warnings": ["Missing LC file"],
        }

    scatter_ratio = float(run_params.get("bad_camera_scatter_ratio", 2.5)) if run_params else 2.5
    clean_abs = float(run_params.get("clean_max_error_absolute", 1.0)) if run_params else 1.0
    clean_sig = float(run_params.get("clean_max_error_sigma", 5.0)) if run_params else 5.0

    df, filtered_cameras, camera_diagnostics = _load_cleaned_df(
        lc_path,
        filter_bad_cameras=filter_bad_cameras,
        scatter_ratio=scatter_ratio,
        clean_max_error_absolute=clean_abs,
        clean_max_error_sigma=clean_sig,
    )

    missing_cols = sorted(list(REQUIRED_COLUMNS - set(df.columns)))
    if missing_cols:
        return {
            "figure": _status_figure(
                f"Missing required columns: {', '.join(missing_cols)}. Switch to PNG mode or verify light-curve schema.",
                theme=theme,
            ),
            "camera_options": [],
            "camera_values": [],
            "stat_rows": [],
            "status": "missing-columns",
            "status_message": f"Missing columns: {', '.join(missing_cols)}",
            "camera_diagnostics": camera_diagnostics,
            "warnings": ["Missing required columns"],
        }

    if df.empty:
        return {
            "figure": _status_figure("No points remain after cleaning/filtering. Try Select all cameras or disable bad-camera filtering.", theme=theme),
            "camera_options": [],
            "camera_values": [],
            "stat_rows": [],
            "status": "empty-after-filter",
            "status_message": "No points remain after filtering.",
            "camera_diagnostics": camera_diagnostics,
            "warnings": ["No data after filtering"],
        }

    median_jd = float(df["JD"].median())
    jd_offset = 2458000.0 if median_jd > 2000000 else 8000.0
    df = df[np.isfinite(df["JD"]) & np.isfinite(df["mag"])].copy()
    df["JD_plot"] = df["JD"] - jd_offset
    df["camera_label"] = _camera_labels(df)

    camera_ids = sorted(df["camera_label"].dropna().unique().tolist())
    selected = [str(c) for c in (selected_cameras or []) if str(c) in camera_ids]
    if not selected:
        selected = list(camera_ids)
    df = df[df["camera_label"].isin(selected)].copy()

    if df.empty:
        return {
            "figure": _status_figure("Camera selection removed all points. Use Select all cameras or Reset.", theme=theme),
            "camera_options": [{"label": f"{cam}", "value": str(cam)} for cam in camera_ids],
            "camera_values": selected,
            "stat_rows": [],
            "status": "empty-camera-selection",
            "status_message": "Current camera selection has no points.",
            "camera_diagnostics": camera_diagnostics,
            "warnings": ["No points for selected cameras"],
        }

    baseline_name = str(run_params.get("baseline_func", "per_camera_gp")) if run_params else "per_camera_gp"
    baseline_cache_key = (
        str(lc_path.resolve()),
        tuple(sorted(str(c) for c in selected)),
        bool(filter_bad_cameras),
        float(scatter_ratio),
        float(clean_abs),
        float(clean_sig),
    )
    band_dfs = _compute_baseline_bands(df, baseline_name, baseline_cache_key)

    warnings: list[str] = []
    phase_period = _phase_period_days(payload) if show_phase_fold else None
    phase_enabled = bool(show_phase_fold and phase_period is not None)
    if show_phase_fold and not phase_enabled:
        warnings.append("Phase panel requested, but no valid period was found.")

    try:
        residual_fraction = float(residual_fraction)
    except Exception:
        residual_fraction = 0.28
    if not np.isfinite(residual_fraction):
        residual_fraction = 0.28
    residual_fraction = float(np.clip(residual_fraction, 0.15, 0.45))

    n_rows = 1 + (1 if show_residuals else 0) + (1 if phase_enabled else 0)
    if n_rows == 1:
        row_heights = [1.0]
    elif n_rows == 2:
        row_heights = [1.0 - residual_fraction, residual_fraction] if show_residuals else [0.68, 0.32]
    else:
        phase_fraction = 0.22
        residual_fraction_3 = float(np.clip(residual_fraction, 0.15, 0.40))
        main_fraction = 1.0 - phase_fraction - residual_fraction_3
        row_heights = [main_fraction, residual_fraction_3, phase_fraction]

    residual_row = 2 if show_residuals else None
    phase_row = (3 if show_residuals else 2) if phase_enabled else None

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=(not phase_enabled),
        vertical_spacing=0.05,
        row_heights=row_heights,
    )

    band_labels = {0: "g", 1: "V"}
    band_markers = {0: "circle", 1: "square"}

    for band in (0, 1):
        bdf = band_dfs.get(band)
        if bdf is None or bdf.empty:
            continue
        for cam in selected:
            cdf = bdf[bdf["camera_label"] == cam]
            if cdf.empty:
                continue
            color = _stable_camera_color(cam)
            err = cdf["error"].to_numpy() if "error" in cdf.columns else np.full(len(cdf), np.nan)
            resid = cdf["resid"].to_numpy() if "resid" in cdf.columns else np.full(len(cdf), np.nan)
            baseline = cdf["baseline"].to_numpy() if "baseline" in cdf.columns else np.full(len(cdf), np.nan)
            hover = np.column_stack([cdf["JD"].to_numpy(), err, resid, baseline])

            fig.add_trace(
                go.Scatter(
                    x=cdf["JD_plot"],
                    y=cdf["mag"],
                    mode="markers",
                    name=f"{cam} ({band_labels[band]})",
                    marker={
                        "size": 7,
                        "symbol": band_markers[band],
                        "color": color,
                        "line": {"width": 0.8, "color": colors["marker_line"]},
                    },
                    error_y={"type": "data", "array": err, "visible": True, "thickness": 1, "width": 0, "color": color},
                    customdata=hover,
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "JD: %{customdata[0]:.5f}<br>"
                        "JD plot: %{x:.5f}<br>"
                        "Mag: %{y:.4f}<br>"
                        "Err: %{customdata[1]:.4f}<br>"
                        "Resid: %{customdata[2]:.4f}<br>"
                        "Baseline: %{customdata[3]:.4f}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

            if show_residuals:
                fig.add_trace(
                    go.Scatter(
                        x=cdf["JD_plot"],
                        y=resid,
                        mode="markers",
                        showlegend=False,
                        marker={
                            "size": 6,
                            "symbol": band_markers[band],
                            "color": color,
                            "line": {"width": 0.8, "color": colors["marker_line"]},
                        },
                        customdata=hover,
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            "JD: %{customdata[0]:.5f}<br>"
                            "Residual: %{y:.4f}<extra></extra>"
                        ),
                    ),
                    row=residual_row,
                    col=1,
                )

        if show_baseline and "baseline" in bdf.columns:
            for cam in selected:
                cbase = bdf[(bdf["camera_label"] == cam) & np.isfinite(bdf["baseline"])].sort_values("JD_plot")
                if cbase.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=cbase["JD_plot"],
                        y=cbase["baseline"],
                        mode="lines",
                        showlegend=False,
                        line={"width": 1.6, "color": _stable_camera_color(cam)},
                        opacity=baseline_opacity,
                        hovertemplate="Baseline: %{y:.4f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

    event_entries = _event_entries(payload, jd_offset, run_params)
    if show_event_markers:
        y_ref = float(df["mag"].min()) if not df.empty else 0.0
        for entry in event_entries:
            color = entry["base_color"]
            conf = float(entry["confidence"])
            bf_text = "n/a" if entry["bf"] is None else f"{float(entry['bf']):.2f}"
            logbf_thr = entry.get("logbf_threshold")
            sig_thr = entry.get("sig_threshold")
            logbf_thr_text = "n/a" if logbf_thr is None else f"{float(logbf_thr):.2f}"
            sig_thr_text = "n/a" if sig_thr is None else f"{float(sig_thr):.2f}"
            if confidence_colors:
                alpha = 0.35 + 0.55 * conf
                if entry["kind"] == "dip":
                    color = f"rgba(255,96,96,{alpha:.3f})"
                else:
                    color = f"rgba(92,214,110,{alpha:.3f})"

            fig.add_vline(x=float(entry["x0"]), line_color=color, line_dash="dash", line_width=1.8)

            if show_diagnostics and float(entry["half_width"]) > 0:
                fig.add_vrect(
                    x0=float(entry["x0"]) - float(entry["half_width"]),
                    x1=float(entry["x0"]) + float(entry["half_width"]),
                    fillcolor=color,
                    opacity=0.11,
                    line_width=0,
                    layer="below",
                )
                fig.add_annotation(
                    x=float(entry["x0"]),
                    y=1.0,
                    xref="x",
                    yref="paper",
                    text=(
                        f"{str(entry['kind']).title()} thr logBF={logbf_thr_text}, sig={sig_thr_text}"
                    ),
                    showarrow=False,
                    font={"size": 9, "color": colors["annotation"]},
                    yshift=-12 if entry["kind"] == "dip" else -24,
                )

            fig.add_trace(
                go.Scatter(
                    x=[float(entry["x0"])],
                    y=[y_ref],
                    mode="markers",
                    marker={"size": 9, "color": color, "symbol": "diamond"},
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{str(entry['kind']).title()} event</b><br>"
                        f"t0 (JD): {float(entry['t0']):.5f}<br>"
                        f"width param: {float(entry['half_width']) / 2.0:.3f}<br>"
                        f"logBF: {bf_text}<br>"
                        f"logBF threshold: {logbf_thr_text}<br>"
                        f"significance threshold: {sig_thr_text}<br>"
                        f"morph: {entry['morph'] or 'n/a'}<br>"
                        f"confidence: {float(entry['confidence']):.2f}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

    if phase_enabled and phase_row is not None and phase_period is not None:
        phase_df = _phase_fold_df(df, phase_period)
        for band in (0, 1):
            bdf = phase_df[phase_df["v_g_band"] == band]
            if bdf.empty:
                continue
            for cam in selected:
                cdf = bdf[bdf["camera_label"] == cam]
                if cdf.empty:
                    continue
                color = _stable_camera_color(cam)
                err = cdf["error"].to_numpy() if "error" in cdf.columns else np.full(len(cdf), np.nan)
                hover = np.column_stack([cdf["JD"].to_numpy(), err])
                fig.add_trace(
                    go.Scatter(
                        x=cdf["phase"],
                        y=cdf["mag"],
                        mode="markers",
                        showlegend=False,
                        marker={
                            "size": 6,
                            "symbol": band_markers[band],
                            "color": color,
                            "line": {"width": 0.7, "color": "rgba(10,10,10,0.95)"},
                        },
                        error_y={"type": "data", "array": err, "visible": True, "thickness": 1, "width": 0, "color": color},
                        customdata=hover,
                        hovertemplate=(
                            "<b>Phase-folded</b><br>"
                            "Phase: %{x:.4f}<br>"
                            "Mag: %{y:.4f}<br>"
                            "JD: %{customdata[0]:.5f}<br>"
                            "Err: %{customdata[1]:.4f}<extra></extra>"
                        ),
                    ),
                    row=phase_row,
                    col=1,
                )

        fig.add_vline(x=0.0, line_color=colors["guide_line"], line_dash="dot", line_width=1.0, row=phase_row, col=1)
        fig.add_vline(x=1.0, line_color=colors["guide_line"], line_dash="dot", line_width=1.0, row=phase_row, col=1)
        fig.add_vline(x=2.0, line_color=colors["guide_line"], line_dash="dot", line_width=1.0, row=phase_row, col=1)

    fig.update_yaxes(title_text="Magnitude [mag]", row=1, col=1, autorange="reversed")
    if show_residuals:
        fig.update_yaxes(title_text="Residual [mag]", row=residual_row, col=1, autorange="reversed")
        fig.add_hline(y=0.0, line_color=colors["guide_line"], line_dash="dot", row=residual_row, col=1)

    if phase_enabled and phase_row is not None and phase_period is not None:
        if show_residuals:
            fig.update_xaxes(title_text=f"JD - {int(jd_offset)}", row=residual_row, col=1)
            fig.update_xaxes(matches="x", row=residual_row, col=1)
        else:
            fig.update_xaxes(title_text=f"JD - {int(jd_offset)}", row=1, col=1)
        fig.update_xaxes(title_text=f"Phase (P={phase_period:.5f} d)", row=phase_row, col=1, range=[-0.02, 2.02])
        fig.update_yaxes(title_text="Phase mag [mag]", row=phase_row, col=1, autorange="reversed")
    else:
        fig.update_xaxes(title_text=f"JD - {int(jd_offset)}", row=n_rows, col=1)

    fig.update_layout(
        title=_build_title(payload, df),
        title_font={"size": 14, "color": colors["title"]},
        paper_bgcolor=colors["paper_bg"],
        plot_bgcolor=colors["plot_bg"],
        margin={"l": 55, "r": 20, "t": 54, "b": 44},
        font={"color": colors["text"], "family": "Monaco, Courier New, monospace", "size": 11},
        hovermode="closest",
        legend={
            "bgcolor": colors["legend_bg"],
            "bordercolor": colors["legend_border"],
            "borderwidth": 1,
            "font": {"size": 10},
        },
        height=760 if phase_enabled and show_residuals else (640 if phase_enabled else (650 if show_residuals else 480)),
        uirevision=uirevision_key,
    )
    fig.update_xaxes(showgrid=True, gridcolor=colors["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=colors["grid"], zeroline=False)

    camera_options = [{"label": f"{cam}", "value": str(cam)} for cam in camera_ids]
    return {
        "figure": fig,
        "camera_options": camera_options,
        "camera_values": selected,
        "stat_rows": _build_stat_rows(payload, df, filtered_cameras),
        "status": "ok",
        "status_message": "",
        "camera_diagnostics": camera_diagnostics,
        "warnings": warnings,
    }
