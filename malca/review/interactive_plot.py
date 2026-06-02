"""Interactive light-curve plotting for the review GUI."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from malca.baseline import (
    global_median_baseline,
    per_camera_gp_baseline,
    per_camera_gp_baseline_masked,
    per_camera_median_baseline,
)
from malca.lightcurve_io import load_lightcurve_df, stable_camera_color
from malca.phase import BAND_LABELS, phase_fold_dataframe, phase_time_dataframe, resolve_phase_epoch, resolve_phase_period
from malca.utils import (
    clean_lc,
    identify_bad_cameras,
    identify_catastrophic_outlier_cameras,
    identify_offset_cameras,
)
from malca.config import (
    JD_OFFSET, MJD_TO_JD, GAIA_TCB_EPOCH_JD, TESS_BTJD_OFFSET, KEPLER_BKJD_OFFSET,
    REVIEW_CACHE_LIMIT, REVIEW_MAX_EXTERNAL_POINTS, REVIEW_RESIDUAL_FRACTION,
)
from malca.config import (
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    CLEAN_LC_MAX_ERROR_ABSOLUTE,
    CLEAN_LC_MAX_ERROR_SIGMA,
    OFFSET_CAMERA_SIGMA_THRESHOLD,
)


BASELINE_FUNCTIONS = {
    "global_median": global_median_baseline,
    "per_camera_median": per_camera_median_baseline,
    "per_camera_gp": per_camera_gp_baseline,
    "gp": per_camera_gp_baseline,
    "gp_masked": per_camera_gp_baseline_masked,
    "per_camera_gp_masked": per_camera_gp_baseline_masked,
}

REQUIRED_COLUMNS = {"JD", "mag", "v_g_band"}
DIP_EVENT_COLOR = "#ff6b6b"
JUMP_EVENT_COLOR = "#0096FF"
PHASE_TIME_COLORSCALE = [
    [0.0, JUMP_EVENT_COLOR],
    [0.5, "#ffffff"],
    [1.0, DIP_EVENT_COLOR],
]


def _coerce_finite_float(value: object) -> float | None:
    """Return a finite float from a run-param value, or None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _baseline_config_from_run_params(run_params: dict | None) -> tuple[str, dict[str, float], list[str]]:
    """Resolve review baseline mode and GP kwargs from run parameters."""
    params = dict(run_params or {})
    requested_name = str(params.get("baseline_func") or "per_camera_gp").strip()
    baseline_name = requested_name or "per_camera_gp"
    warnings: list[str] = []

    if baseline_name not in BASELINE_FUNCTIONS:
        warnings.append(
            f"Unknown baseline_func '{baseline_name}' in run_params; falling back to per_camera_gp."
        )
        baseline_name = "per_camera_gp"

    baseline_kwargs: dict[str, float] = {}
    if baseline_name in {"gp", "per_camera_gp", "gp_masked", "per_camera_gp_masked"}:
        for run_key, arg_key in (
            ("baseline_s0", "S0"),
            ("baseline_w0", "w0"),
            ("baseline_q", "q"),
            ("baseline_jitter", "jitter"),
            ("baseline_sigma_floor", "sigma_floor"),
        ):
            value = _coerce_finite_float(params.get(run_key))
            if value is not None:
                baseline_kwargs[arg_key] = float(value)

    return baseline_name, baseline_kwargs, warnings


def _freeze_baseline_kwargs(baseline_kwargs: dict[str, object] | None) -> tuple[tuple[str, object], ...]:
    """Convert baseline kwargs into a stable cache key fragment."""
    frozen: list[tuple[str, object]] = []
    for key, value in sorted((baseline_kwargs or {}).items()):
        if isinstance(value, (float, np.floating)):
            frozen.append((str(key), float(value)))
        elif isinstance(value, (int, np.integer, bool, np.bool_)):
            frozen.append((str(key), int(value)))
        elif value is None:
            frozen.append((str(key), None))
        else:
            frozen.append((str(key), str(value)))
    return tuple(frozen)


def _mag_to_flux(mag: np.ndarray) -> np.ndarray:
    """Convert magnitude to flux: flux = 10^(-0.4 * mag)."""
    return np.power(10.0, -0.4 * mag)


def _flux_err_from_mag_err(flux: np.ndarray, mag_err: np.ndarray) -> np.ndarray:
    """Propagate magnitude error to flux: flux_err ≈ 0.921 * flux * mag_err."""
    return np.where(np.isfinite(flux) & np.isfinite(mag_err), 0.921 * flux * mag_err, np.nan)


def _robust_color_bounds(values: np.ndarray) -> tuple[float | None, float | None]:
    """Return percentile color bounds with a finite fallback."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    lo, hi = np.nanpercentile(vals, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None, None
    if lo == hi:
        pad = max(abs(float(lo)) * 0.05, 0.05)
        lo = float(lo) - pad
        hi = float(hi) + pad
    return float(lo), float(hi)


def _zero_centered_color_bounds(values: np.ndarray) -> tuple[float | None, float | None]:
    """Return symmetric bounds so Δm=0 maps to the color midpoint."""
    lo, hi = _robust_color_bounds(values)
    if lo is None or hi is None:
        return None, None
    limit = max(abs(float(lo)), abs(float(hi)))
    if not np.isfinite(limit) or limit <= 0:
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        limit = float(np.nanmax(np.abs(vals))) if vals.size else 0.0
    if not np.isfinite(limit) or limit <= 0:
        limit = 0.05
    return -limit, limit


# Keep plotting caches bounded; large values can inflate long-running GUI memory.
_CACHE_LIMIT = REVIEW_CACHE_LIMIT
_MAX_EXTERNAL_TRACE_POINTS = REVIEW_MAX_EXTERNAL_POINTS
_CLEAN_CACHE: OrderedDict[tuple, tuple[pd.DataFrame, set[int], dict[str, list[str]]]] = OrderedDict()
_BASELINE_CACHE: OrderedDict[tuple, dict[int, pd.DataFrame]] = OrderedDict()
_EVENT_CACHE: OrderedDict[tuple, list[dict[str, object]]] = OrderedDict()
_EXTERNAL_LC_CACHE: OrderedDict[tuple, pd.DataFrame] = OrderedDict()


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
    bundle_dir = None
    if plot_dir is not None:
        bundle_dir = plot_dir.parent / "bundle_assets" / "lightcurves"

    candidate_names: list[str] = []

    keys = ("path", "lc_path")
    for key in keys:
        raw_path = payload.get(key)
        if not raw_path:
            continue
        candidate = Path(str(raw_path)).expanduser()
        candidate_names.append(candidate.name)
        if candidate.suffix in (".dat", ".dat2", ".dat3"):
            candidate_names.append(candidate.with_suffix(".raw2").name)
        if candidate.exists():
            return candidate

    candidate_id = payload.get("candidate_id")
    if candidate_id is not None:
        cid = str(candidate_id).strip()
        if cid:
            candidate_names.extend([f"{cid}.dat3", f"{cid}.raw2", f"{cid}.dat2", f"{cid}.dat"])

    asas_sn_id = payload.get("asas_sn_id")
    if asas_sn_id is not None:
        sid = str(asas_sn_id).strip()
        if sid:
            candidate_names.extend([f"{sid}.dat3", f"{sid}.raw2", f"{sid}.dat2", f"{sid}.dat"])

    if bundle_dir is not None:
        seen: set[str] = set()
        for name in candidate_names:
            if not name or name in seen:
                continue
            seen.add(name)
            bundle_candidate = bundle_dir / name
            if bundle_candidate.exists():
                return bundle_candidate

    return None


def _camera_labels(df: pd.DataFrame, payload: dict | None = None) -> pd.Series:
    if "camera_name" in df.columns:
        base_labels = pd.Series(df["camera_name"].astype(str), index=df.index)
    elif "camera#" in df.columns:
        base_labels = pd.Series(df["camera#"].astype(str), index=df.index)
    elif "camera" in df.columns:
        base_labels = pd.Series(df["camera"].astype(str), index=df.index)
    else:
        base_labels = pd.Series(["unknown"] * len(df), index=df.index)

    if payload and "by_camera" in payload:
        by_cam = payload["by_camera"]
        if isinstance(by_cam, pd.DataFrame) and "camera" in by_cam.columns and "loo_corr" in by_cam.columns:
            loo_map = dict(zip(by_cam["camera"].astype(str), by_cam["loo_corr"]))
            
            def _format_label(name):
                corr = loo_map.get(str(name))
                if corr is not None and np.isfinite(corr):
                    return f"{name} (LOO Corr: {corr:.2f})"
                return str(name)
                
            return base_labels.map(_format_label)
            
    return base_labels


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
        offset_bad, _ = identify_offset_cameras(df, offset_sigma_threshold=OFFSET_CAMERA_SIGMA_THRESHOLD, remove_full_camera=True)
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


def _compute_baseline_bands(
    df: pd.DataFrame,
    baseline_name: str,
    cache_key: tuple,
    *,
    baseline_kwargs: dict[str, object] | None = None,
) -> dict[int, pd.DataFrame]:
    key = (cache_key, baseline_name, _freeze_baseline_kwargs(baseline_kwargs))
    cached = _cache_get(_BASELINE_CACHE, key)
    if cached is not None:
        return {k: v.copy() for k, v in cached.items()}

    baseline_func = BASELINE_FUNCTIONS.get(baseline_name, per_camera_gp_baseline)
    call_kwargs = dict(baseline_kwargs or {})
    if baseline_name in {"gp", "per_camera_gp", "gp_masked", "per_camera_gp_masked"}:
        call_kwargs.setdefault("add_sigma_eff_col", True)

    band_dfs: dict[int, pd.DataFrame] = {}
    for band in (0, 1):
        bdf = df[df["v_g_band"] == band].copy()
        if bdf.empty:
            continue
        try:
            out = baseline_func(bdf, **call_kwargs)
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


def warm_caches_for_candidate(
    payload: dict,
    plot_dir: Path | None,
    *,
    run_params: dict | None = None,
) -> None:
    """Preload LC and baseline into caches so the next time this candidate is shown it loads instantly."""
    plot_dir = Path(plot_dir) if plot_dir else None
    lc_path = resolve_lightcurve_path(payload, plot_dir)
    if lc_path is None or not lc_path.exists():
        return
    scatter_ratio = (
        float(run_params.get("bad_camera_scatter_ratio", BAD_CAMERA_SCATTER_RATIO_THRESHOLD))
        if run_params else BAD_CAMERA_SCATTER_RATIO_THRESHOLD
    )
    clean_abs = (
        float(run_params.get("clean_max_error_absolute", CLEAN_LC_MAX_ERROR_ABSOLUTE))
        if run_params else CLEAN_LC_MAX_ERROR_ABSOLUTE
    )
    clean_sig = (
        float(run_params.get("clean_max_error_sigma", CLEAN_LC_MAX_ERROR_SIGMA))
        if run_params else CLEAN_LC_MAX_ERROR_SIGMA
    )
    try:
        df, _, _ = _load_cleaned_df(
            lc_path,
            filter_bad_cameras=True,
            scatter_ratio=scatter_ratio,
            clean_max_error_absolute=clean_abs,
            clean_max_error_sigma=clean_sig,
        )
    except Exception:
        return
    cache_key = (str(lc_path.resolve()), True, scatter_ratio, clean_abs, clean_sig)
    baseline_name, baseline_kwargs, _ = _baseline_config_from_run_params(run_params)
    try:
        _compute_baseline_bands(df, baseline_name, cache_key, baseline_kwargs=baseline_kwargs)
    except Exception:
        pass


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
    _ = df

    def _include_key(key: str, value: object) -> bool:
        if key.startswith("stats_"):
            return True
        if not key.startswith("ltv_"):
            return False
        if key.endswith("_name"):
            return False
        if isinstance(value, str):
            return False
        return True

    def _format_value(value: object) -> str:
        if isinstance(value, (bool, np.bool_)):
            return "True" if bool(value) else "False"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        if isinstance(value, (float, np.floating)):
            f = float(value)
            if not np.isfinite(f):
                return ""
            if f != 0.0 and (abs(f) < 1e-3 or abs(f) >= 1e4):
                mantissa, exponent = f"{f:.6e}".split("e")
                mantissa = mantissa.rstrip("0").rstrip(".")
                exp = int(exponent)
                return rf"${mantissa}\times10^{{{exp}}}$"
            return f"{f:.6g}"
        return str(value)

    rows: list[tuple[str, str]] = []
    for key in sorted(
        k for k in payload.keys()
        if isinstance(k, str) and _include_key(k, payload.get(k))
    ):
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray, dict, list, tuple, set)):
            continue
        display = _format_value(value)
        if not display or display in {"nan", "NaN", "None"}:
            continue
        rows.append((key, display))

    if filtered_cameras:
        rows.append(("filtered_cams", ",".join(str(c) for c in sorted(filtered_cameras))))
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
    for prefix, color in (("dip", DIP_EVENT_COLOR), ("jump", JUMP_EVENT_COLOR)):
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


def _theme_palette(theme: str) -> dict[str, str]:
    mode = str(theme or "black").lower()
    if mode == "black":
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
    if mode == "gray":
        return {
            "text": "#d8dee9",
            "title": "#eceff4",
            "paper_bg": "#2e3440",
            "plot_bg": "#2e3440",
            "grid": "rgba(129, 161, 193, 0.15)",
            "legend_bg": "rgba(59, 66, 82, 0.9)",
            "legend_border": "rgba(129, 161, 193, 0.3)",
            "annotation": "#88c0d0",
            "marker_line": "rgba(236, 239, 244, 0.8)",
            "guide_line": "rgba(129, 161, 193, 0.3)",
        }
    if mode == "white":
        return {
            "text": "#15202b",
            "title": "#15202b",
            "paper_bg": "#f5f7fa",
            "plot_bg": "#f5f7fa",
            "grid": "rgba(104, 128, 149, 0.22)",
            "legend_bg": "rgba(255, 255, 255, 0.92)",
            "legend_border": "rgba(120, 140, 158, 0.35)",
            "annotation": "#245f8f",
            "marker_line": "rgba(255, 255, 255, 0.95)",
            "guide_line": "rgba(86, 112, 137, 0.28)",
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


def _status_figure(message: str, theme: str = "black") -> go.Figure:
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


def _subplot_axis_ref(row: int, axis: Literal["x", "y"]) -> str:
    """Return Plotly's axis reference string for a one-column subplot row."""
    return axis if row <= 1 else f"{axis}{row}"


def _subplot_domain_ref(row: int, axis: Literal["x", "y"]) -> str:
    """Return the subplot domain reference for layout-only overlays."""
    return f"{_subplot_axis_ref(row, axis)} domain"


def _event_annotation_y(
    x_values: np.ndarray,
    y_values: np.ndarray,
    event_x: float,
    half_width: float,
    *,
    kind: str,
    is_flux: bool,
    pad: float,
) -> tuple[float, float]:
    """Return marker and label y-values that sit just outside nearby data."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return 0.0, 0.0

    x = x[finite]
    y = y[finite]
    event_x = float(event_x)
    half_width = float(half_width) if np.isfinite(half_width) else 0.0
    local_window = max(half_width, 0.5)
    local = np.abs(x - event_x) <= local_window
    if local.any():
        local_y = y[local]
    else:
        nearest = np.argsort(np.abs(x - event_x))[: min(5, x.size)]
        local_y = y[nearest]

    top = float(np.nanmax(y) if is_flux else np.nanmin(y))
    local_bottom = float(np.nanmin(local_y) if is_flux else np.nanmax(local_y))
    marker_gap = 0.30 * pad
    label_gap = 0.66 * pad

    if str(kind).lower() == "dip":
        return (
            local_bottom - marker_gap if is_flux else local_bottom + marker_gap,
            local_bottom - label_gap if is_flux else local_bottom + label_gap,
        )

    return (
        top + marker_gap if is_flux else top - marker_gap,
        top + label_gap if is_flux else top - label_gap,
    )


_EXTERNAL_LC_SPECS: dict[str, dict] = {
    "atlas": {
        "time_col": "mjd",
        "time_offset": MJD_TO_JD,
        "jd_system": "mjd",
        "bands": {
            "c": {"color": "#00ccff", "marker": "diamond", "label": "ATLAS c"},
            "o": {"color": "#ff8c42", "marker": "diamond", "label": "ATLAS o"},
        },
        "filter_col": "filter",
        "mag_col": "mag",
        "err_col": "mag_err",
    },
    "ztf": {
        "time_col": "mjd",
        "jd_system": "mjd",
        "bands": {
            "zg": {"color": "#44aa44", "marker": "triangle-up", "label": "ZTF g"},
            "zr": {"color": "#dd4444", "marker": "triangle-up", "label": "ZTF r"},
            "zi": {"color": "#8844cc", "marker": "triangle-up", "label": "ZTF i"},
        },
        "filter_col": "band",
        "mag_col": "mag",
        "err_col": "mag_err",
    },
    "gaia_epoch": {
        "time_col": "time",
        "jd_system": "bjd_gaia",  # Gaia TCB in days since J2010.0 (JD 2455197.5)
        "bands": {
            "G": {"color": "#e8c547", "marker": "star", "label": "Gaia G"},
        },
        "filter_col": "band",
        "mag_col": "mag",
        "err_col": "mag_err",
    },
    "tess": {
        "time_col": "time",
        "jd_system": "btjd",  # BTJD = BJD - 2457000.0
        "is_flux": True,
        "bands": {
            "TESS": {"color": "#cc66ff", "marker": "hexagon", "label": "TESS"},
        },
        "filter_col": None,
        "mag_col": "flux",
        "err_col": "flux_err",
    },
    "kepler": {
        "time_col": "time",
        "jd_system": "bkjd",  # BKJD = BJD - 2454833.0
        "is_flux": True,
        "bands": {
            "Kepler": {"color": "#ffb6c1", "marker": "hexagon", "label": "Kepler/K2"},
        },
        "filter_col": None,
        "mag_col": "flux",
        "err_col": "flux_err",
    },
    "aavso": {
        "time_col": "mjd",
        "jd_system": "mjd",
        "bands": {
            "V": {"color": "#00ff00", "marker": "circle", "label": "AAVSO V"},
            "B": {"color": "#0000ff", "marker": "circle", "label": "AAVSO B"},
            "CV": {"color": "#aaaaaa", "marker": "circle", "label": "AAVSO CV"},
        },
        "filter_col": "filter",
        "mag_col": "mag",
        "err_col": "mag_err",
    },
    "ps1": {
        "time_col": "mjd",
        "jd_system": "mjd",
        "bands": {
            "g_ps": {"color": "#44aa44", "marker": "star", "label": "PS1 g"},
            "r_ps": {"color": "#dd4444", "marker": "star", "label": "PS1 r"},
            "i_ps": {"color": "#8844cc", "marker": "star", "label": "PS1 i"},
            "z_ps": {"color": "#ccaa44", "marker": "star", "label": "PS1 z"},
            "y_ps": {"color": "#aaaa33", "marker": "star", "label": "PS1 y"},
        },
        "filter_col": "filter",
        "mag_col": "mag",
        "err_col": "mag_err",
    },
    "crts": {
        "time_col": "mjd",
        "jd_system": "mjd",
        "bands": {
            "CV": {"color": "#bbbbbb", "marker": "square", "label": "CRTS CV"},
        },
        "filter_col": None,
        "mag_col": "mag",
        "err_col": "mag_err",
    },
}


def _rename_first_present(df: pd.DataFrame, canonical: str, aliases: tuple[str, ...]) -> pd.DataFrame:
    """Rename the first matching alias to *canonical* if needed."""
    if canonical in df.columns:
        return df
    for alias in aliases:
        if alias in df.columns:
            return df.rename(columns={alias: canonical})
    return df


def _coerce_numeric_column(df: pd.DataFrame, column: str) -> None:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")


def _normalize_mjd_column(df: pd.DataFrame, column: str = "mjd") -> None:
    """Normalize a time column to true MJD, tolerating JD-valued inputs."""
    if column not in df.columns:
        return
    df[column] = pd.to_numeric(df[column], errors="coerce")
    finite = df[column].to_numpy()
    finite = finite[np.isfinite(finite)]
    if finite.size and float(np.nanmedian(finite)) > 1_000_000.0:
        df[column] = df[column] - MJD_TO_JD


def normalize_external_lc_dataframe(source_name: str, df_ext: pd.DataFrame) -> pd.DataFrame:
    """Normalize heterogeneous external-LC schemas to the viewer's expected columns."""
    if df_ext is None or df_ext.empty:
        return df_ext

    source = str(source_name or "").strip().lower()
    df = df_ext.copy()

    if source == "atlas":
        df = _rename_first_present(df, "mjd", ("MJD", "mjd", "JD"))
        df = _rename_first_present(df, "filter", ("F", "filter"))
        df = _rename_first_present(df, "mag", ("m", "mag"))
        df = _rename_first_present(df, "mag_err", ("dm", "mag_err", "magerr"))
        _normalize_mjd_column(df)
        if "filter" in df.columns:
            df["filter"] = df["filter"].astype(str).str.strip().str.lower()
    elif source == "ztf":
        df = _rename_first_present(df, "mjd", ("mjd", "hjd"))
        df = _rename_first_present(df, "band", ("band", "filtercode"))
        df = _rename_first_present(df, "mag", ("mag",))
        df = _rename_first_present(df, "mag_err", ("mag_err", "magerr"))
        _normalize_mjd_column(df)
        if "band" in df.columns:
            band_map = {
                "1": "zg", "1.0": "zg",
                "2": "zr", "2.0": "zr",
                "3": "zi", "3.0": "zi",
                "zg": "zg", "zr": "zr", "zi": "zi",
            }
            df["band"] = (
                df["band"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(lambda v: band_map.get(v, v))
            )
    elif source == "gaia_epoch":
        df = _rename_first_present(df, "time", ("time", "g_transit_time"))
        df = _rename_first_present(df, "band", ("band",))
        df = _rename_first_present(df, "mag", ("mag", "g_transit_mag"))
        df = _rename_first_present(df, "mag_err", ("mag_err", "mag_error", "g_transit_mag_error"))
        _coerce_numeric_column(df, "time")
        if "band" in df.columns:
            df["band"] = df["band"].astype(str).str.strip().str.upper()
    elif source == "tess":
        df = _rename_first_present(df, "time", ("time",))
        df = _rename_first_present(df, "flux", ("flux",))
        df = _rename_first_present(df, "flux_err", ("flux_err",))
        _coerce_numeric_column(df, "time")
    elif source == "kepler":
        df = _rename_first_present(df, "time", ("time",))
        df = _rename_first_present(df, "flux", ("flux",))
        df = _rename_first_present(df, "flux_err", ("flux_err",))
        _coerce_numeric_column(df, "time")
    elif source == "aavso":
        df = _rename_first_present(df, "mjd", ("mjd", "JD"))
        df = _rename_first_present(df, "filter", ("filter", "Filter"))
        df = _rename_first_present(df, "mag", ("mag", "Mag"))
        df = _rename_first_present(df, "mag_err", ("mag_err", "Err"))
        _normalize_mjd_column(df)
        if "filter" in df.columns:
            df["filter"] = df["filter"].astype(str).str.strip().str.upper()
    elif source == "ps1":
        df = _rename_first_present(df, "mjd", ("mjd", "obsTime"))
        df = _rename_first_present(df, "filter", ("filter", "filterID"))
        df = _rename_first_present(df, "flux_psf", ("flux_psf", "psfFlux"))
        df = _rename_first_present(df, "flux_psf_err", ("flux_psf_err", "psfFluxErr"))
        df = _rename_first_present(df, "mag", ("mag",))
        df = _rename_first_present(df, "mag_err", ("mag_err", "magerr"))
        _normalize_mjd_column(df)
        if "filter" in df.columns:
            filter_map = {
                "1": "g_ps", "1.0": "g_ps",
                "2": "r_ps", "2.0": "r_ps",
                "3": "i_ps", "3.0": "i_ps",
                "4": "z_ps", "4.0": "z_ps",
                "5": "y_ps", "5.0": "y_ps",
                "g": "g_ps", "r": "r_ps", "i": "i_ps", "z": "z_ps", "y": "y_ps",
                "g_ps": "g_ps", "r_ps": "r_ps", "i_ps": "i_ps", "z_ps": "z_ps", "y_ps": "y_ps",
            }
            df["filter"] = (
                df["filter"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(lambda v: filter_map.get(v, v))
            )
        if "mag" not in df.columns and "flux_psf" in df.columns:
            flux = pd.to_numeric(df["flux_psf"], errors="coerce")
            valid_flux = flux > 0
            df["mag"] = np.nan
            df.loc[valid_flux, "mag"] = -2.5 * np.log10(flux[valid_flux]) + 8.90
        if "mag_err" not in df.columns and "flux_psf" in df.columns and "flux_psf_err" in df.columns:
            flux = pd.to_numeric(df["flux_psf"], errors="coerce")
            flux_err = pd.to_numeric(df["flux_psf_err"], errors="coerce")
            df["mag_err"] = np.nan
            valid_flux = flux > 0
            df.loc[valid_flux, "mag_err"] = 1.08 * (flux_err[valid_flux] / flux[valid_flux])
    elif source == "crts":
        df = _rename_first_present(df, "mjd", ("mjd", "ObsTime"))
        df = _rename_first_present(df, "mag", ("mag", "Mag"))
        df = _rename_first_present(df, "mag_err", ("mag_err", "magerr", "e_Mag"))
        _normalize_mjd_column(df)

    return df


def _load_external_lc_frame(source_name: str, lc_path: Path) -> pd.DataFrame:
    """Load and normalize an external LC parquet with a small in-memory cache."""
    try:
        lc_path = Path(lc_path)
        stat = lc_path.stat()
        key = (str(source_name), str(lc_path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    except Exception:
        return pd.DataFrame()

    cached = _cache_get(_EXTERNAL_LC_CACHE, key)
    if cached is not None:
        return cached.copy()

    try:
        df = normalize_external_lc_dataframe(source_name, pd.read_parquet(lc_path))
    except Exception:
        return pd.DataFrame()

    _cache_put(_EXTERNAL_LC_CACHE, key, df.copy())
    return df


def _overlay_external_lcs(
    fig: go.Figure,
    raw_row: int,
    external_lcs: dict[str, Path],
    jd_offset: float,
    colors: dict,
    theme: str,
    is_flux: bool,
    ext_source_ranges: dict[str, tuple[int, int]],
    trace_offset: int,
    mag_anchor: float | None = None,
    warnings: list[str] | None = None,
) -> None:
    """Load external LC parquets and overlay traces on the raw magnitude panel."""
    current_trace = len(fig.data)
    # The raw-panel x-axis is always intended to be JD - 2458000, even when the
    # native ASAS-SN file stores reduced JD and the main trace uses an internal
    # 8000 shift to reach that same frame.
    plot_jd_offset = JD_OFFSET

    for source_name, lc_path in external_lcs.items():
        spec = _EXTERNAL_LC_SPECS.get(source_name)
        if spec is None:
            continue
        is_flux_source = bool(spec.get("is_flux", False))
        try:
            lc_path = Path(lc_path)
            if not lc_path.exists():
                continue
            df_ext = _load_external_lc_frame(source_name, lc_path)
            if df_ext.empty:
                continue
        except Exception:
            continue

        start_idx = len(fig.data)

        # Resolve time column
        time_col = spec["time_col"]
        actual_time = None
        for c in df_ext.columns:
            if c.lower() == time_col.lower():
                actual_time = c
                break
        if actual_time is None:
            continue

        t = pd.to_numeric(df_ext[actual_time], errors="coerce").to_numpy()

        # Convert to JD_plot (JD - jd_offset, same as ASAS-SN)
        jd_sys = spec.get("jd_system", "mjd")
        if jd_sys == "mjd":
            # Most sources store MJD here, but tolerate mislabeled JD-valued columns.
            finite_t = t[np.isfinite(t)]
            if finite_t.size and float(np.nanmedian(finite_t)) > 1_000_000.0:
                jd = t
            else:
                jd = t + MJD_TO_JD
            x_plot = jd - plot_jd_offset
        elif jd_sys == "bjd_gaia":
            # Gaia TCB days since J2010.0 → JD → JD_plot
            jd = t + GAIA_TCB_EPOCH_JD
            x_plot = jd - plot_jd_offset
        elif jd_sys == "btjd":
            # BTJD → JD → JD_plot
            jd = t + TESS_BTJD_OFFSET
            x_plot = jd - plot_jd_offset
        elif jd_sys == "bkjd":
            # BKJD → JD → JD_plot
            jd = t + KEPLER_BKJD_OFFSET
            x_plot = jd - plot_jd_offset
        else:
            x_plot = t - jd_offset

        filter_col = spec.get("filter_col")
        mag_col = spec["mag_col"]
        err_col = spec.get("err_col", "")

        # Resolve actual column names (case-insensitive)
        col_lookup = {c.lower(): c for c in df_ext.columns}
        actual_mag = col_lookup.get(mag_col.lower())
        actual_err = col_lookup.get(err_col.lower()) if err_col else None
        actual_filt = col_lookup.get(filter_col.lower()) if filter_col else None

        if actual_mag is None:
            continue

        source_transform_warned = False
        for band_key, band_info in spec["bands"].items():
            if actual_filt:
                mask = df_ext[actual_filt].astype(str) == band_key
                band_df = df_ext[mask]
                band_x = x_plot[mask.to_numpy()]
            else:
                band_df = df_ext
                band_x = x_plot

            if band_df.empty:
                continue

            raw_y = pd.to_numeric(band_df[actual_mag], errors="coerce").to_numpy(dtype=float)
            good = np.isfinite(band_x) & np.isfinite(raw_y)
            if not good.any():
                continue

            display_label = str(band_info["label"])
            flux_to_relative_mag = bool(is_flux_source and not is_flux)
            y = raw_y.copy()
            ref_flux = np.nan
            anchor_mag = np.nan
            if flux_to_relative_mag:
                positive = good & (raw_y > 0)
                if not positive.any():
                    continue
                ref_flux = float(np.nanmedian(raw_y[positive]))
                if not np.isfinite(ref_flux) or ref_flux <= 0:
                    continue
                anchor_mag = float(mag_anchor) if mag_anchor is not None and np.isfinite(float(mag_anchor)) else 0.0
                y = np.full(raw_y.shape, np.nan, dtype=float)
                y[positive] = anchor_mag - 2.5 * np.log10(raw_y[positive] / ref_flux)
                good = np.isfinite(band_x) & np.isfinite(y)
                if not good.any():
                    continue
                display_label = f"{display_label} rel. Δm"
                if warnings is not None and not source_transform_warned:
                    warnings.append(
                        f"{band_info['label']} flux plotted as relative magnitude anchored to m={anchor_mag:.4f}; "
                        f"not calibrated {band_info['label']}-band magnitude."
                    )
                    source_transform_warned = True
            elif is_flux_source and is_flux:
                display_label = f"{display_label} flux"
            elif not is_flux_source and is_flux:
                y = np.power(10.0, -0.4 * y)

            err_full = None
            if actual_err and actual_err in band_df.columns:
                ev = pd.to_numeric(band_df[actual_err], errors="coerce").to_numpy(dtype=float)
                valid_err = good & np.isfinite(ev)
                if valid_err.any():
                    err_full = np.full(y.shape, np.nan, dtype=float)
                    if flux_to_relative_mag:
                        valid_flux_err = valid_err & (raw_y > 0)
                        err_full[valid_flux_err] = (2.5 / np.log(10.0)) * ev[valid_flux_err] / raw_y[valid_flux_err]
                    elif not is_flux_source and is_flux:
                        err_full[valid_err] = 0.921 * y[valid_err] * ev[valid_err]
                    else:
                        err_full[valid_err] = ev[valid_err]

            good_idx = np.flatnonzero(good)
            if good_idx.size > _MAX_EXTERNAL_TRACE_POINTS:
                step = int(np.ceil(good_idx.size / float(_MAX_EXTERNAL_TRACE_POINTS)))
                good_idx = good_idx[::step]

            x_vals = band_x[good_idx]
            y_vals = y[good_idx]
            jd_vals = x_vals + plot_jd_offset
            err_array = err_full[good_idx] if err_full is not None else None
            if err_array is not None and not np.isfinite(err_array).any():
                err_array = None

            if flux_to_relative_mag:
                raw_flux_vals = raw_y[good_idx]
                err_vals = err_array if err_array is not None else np.full(jd_vals.shape, np.nan, dtype=float)
                custom_ext = np.column_stack(
                    [
                        jd_vals,
                        raw_flux_vals,
                        np.full(jd_vals.shape, ref_flux, dtype=float),
                        np.full(jd_vals.shape, anchor_mag, dtype=float),
                        err_vals,
                    ]
                )
                hover_ext = (
                    f"<b>{display_label}</b><br>"
                    "JD: %{customdata[0]:.5f}<br>"
                    f"JD - {int(plot_jd_offset)}: %{{x:.5f}}<br>"
                    "m<sub>rel</sub>: %{y:.4f}<br>"
                    "raw flux: %{customdata[1]:.4e}<br>"
                    "median flux: %{customdata[2]:.4e}<br>"
                    "anchor m: %{customdata[3]:.4f}<br>"
                    + ("σ<sub>m,rel</sub>: %{customdata[4]:.4f}<extra></extra>" if err_array is not None else "<extra></extra>")
                )
            elif err_array is not None:
                custom_ext = np.column_stack([jd_vals, err_array])
                hover_ext = (
                    f"<b>{display_label}</b><br>"
                    "JD: %{customdata[0]:.5f}<br>"
                    f"JD - {int(plot_jd_offset)}: %{{x:.5f}}<br>"
                    + ("F: %{y:.4e}<br>" if is_flux else "m: %{y:.4f}<br>")
                    + ("σ<sub>F</sub>: %{customdata[1]:.3e}<extra></extra>" if is_flux else "σ<sub>m</sub>: %{customdata[1]:.4f}<extra></extra>")
                )
            else:
                custom_ext = jd_vals.reshape(-1, 1)
                hover_ext = (
                    f"<b>{display_label}</b><br>"
                    "JD: %{customdata[0]:.5f}<br>"
                    f"JD - {int(plot_jd_offset)}: %{{x:.5f}}<br>"
                    + ("F: %{y:.4e}<br>" if is_flux else "m: %{y:.4f}<br>")
                    + "<extra></extra>"
                )

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers",
                    name=display_label,
                    marker={
                        "size": 6,
                        "symbol": band_info["marker"],
                        "color": band_info["color"],
                        "opacity": 0.8,
                        "line": {"width": 0.5, "color": "rgba(255,255,255,0.5)"},
                    },
                    error_y={"type": "data", "array": err_array, "visible": err_array is not None, "thickness": 0.7, "width": 0, "color": band_info["color"]} if err_array is not None else None,
                    customdata=custom_ext,
                    hovertemplate=hover_ext,
                    legendgroup=source_name,
                ),
                row=raw_row,
                col=1,
            )

        end_idx = len(fig.data)
        if end_idx > start_idx:
            ext_source_ranges[str(source_name).strip().lower()] = (start_idx, end_idx)


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
    phase_panel_mode: Literal["fold", "time"] = "fold",
    show_raw_mag: bool = True,
    override_period: float | None = None,
    override_period_source: str = "manual/search",
    phase_period_pending: bool = False,
    suppress_catalog_phase_period: bool = False,
    show_diagnostics: bool,
    confidence_colors: bool,
    run_params: dict | None,
    uirevision_key: str,
    theme: str = "black",
    residual_fraction: float = REVIEW_RESIDUAL_FRACTION,
    baseline_opacity: float = 0.5,
    yaxis_mode: Literal["mag", "flux"] = "mag",
    external_lcs: dict[str, Path] | None = None,
    external_source_view: str = "all",
    selected_bands: list[str] | None = None,
) -> dict:
    """Build a native Plotly light-curve figure for review mode."""
    colors = _theme_palette(theme)
    phase_panel_mode = "time" if str(phase_panel_mode or "fold").strip().lower() == "time" else "fold"

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

    scatter_ratio = float(run_params.get("bad_camera_scatter_ratio", BAD_CAMERA_SCATTER_RATIO_THRESHOLD)) if run_params else BAD_CAMERA_SCATTER_RATIO_THRESHOLD
    clean_abs = float(run_params.get("clean_max_error_absolute", CLEAN_LC_MAX_ERROR_ABSOLUTE)) if run_params else CLEAN_LC_MAX_ERROR_ABSOLUTE
    clean_sig = float(run_params.get("clean_max_error_sigma", CLEAN_LC_MAX_ERROR_SIGMA)) if run_params else CLEAN_LC_MAX_ERROR_SIGMA

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
    jd_offset = JD_OFFSET if median_jd > 2000000 else 8000.0
    df = df[np.isfinite(df["JD"]) & np.isfinite(df["mag"])].copy()
    df["JD_plot"] = df["JD"] - jd_offset
    df["camera_label"] = _camera_labels(df, payload)

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

    band_labels = BAND_LABELS
    available_band_labels = [
        label
        for band, label in band_labels.items()
        if int((df["v_g_band"] == band).sum()) > 0
    ]
    selected_band_lookup = {
        str(value).strip().lower()
        for value in (selected_bands if selected_bands is not None else available_band_labels)
        if str(value).strip()
    }
    active_bands = [
        band
        for band, label in band_labels.items()
        if label.lower() in selected_band_lookup and label in available_band_labels
    ]
    if not active_bands:
        return {
            "figure": _status_figure("No native bands selected. Re-enable g or V in the sidebar band controls.", theme=theme),
            "camera_options": [{"label": f"{cam}", "value": str(cam)} for cam in camera_ids],
            "camera_values": selected,
            "stat_rows": [],
            "status": "empty-band-selection",
            "status_message": "Current band selection has no visible points.",
            "camera_diagnostics": camera_diagnostics,
            "warnings": ["No points for selected bands"],
        }

    baseline_name, baseline_kwargs, baseline_warnings = _baseline_config_from_run_params(run_params)
    baseline_cache_key = (
        str(lc_path.resolve()),
        tuple(sorted(str(c) for c in selected)),
        bool(filter_bad_cameras),
        float(scatter_ratio),
        float(clean_abs),
        float(clean_sig),
    )
    band_dfs = _compute_baseline_bands(
        df,
        baseline_name,
        baseline_cache_key,
        baseline_kwargs=baseline_kwargs,
    )

    warnings: list[str] = list(baseline_warnings)
    phase_requested = bool(show_phase_fold)
    if phase_requested:
        period_payload = {} if suppress_catalog_phase_period else payload
        phase_period, phase_source = resolve_phase_period(
            period_payload,
            override_period=override_period,
            override_source=override_period_source or "manual/search",
        )
    else:
        phase_period = None
        phase_source = ""
    phase_enabled = bool(phase_requested and phase_period is not None)
    if phase_requested and not phase_enabled:
        if phase_period_pending:
            warnings.append("Auto PDM period search is running; the phase panel will update when it finishes.")
        elif suppress_catalog_phase_period:
            warnings.append("Auto PDM did not return a valid period. Run Find Period or enter Manual P.")
        else:
            warnings.append("Phase panel requested, but no valid period was found. Use Find Period to search manually.")

    try:
        residual_fraction = float(residual_fraction)
    except Exception:
        residual_fraction = REVIEW_RESIDUAL_FRACTION
    if not np.isfinite(residual_fraction):
        residual_fraction = REVIEW_RESIDUAL_FRACTION
    residual_fraction = float(np.clip(residual_fraction, 0.15, 0.85))

    # Dynamic row allocation
    # Row indices are 1-based
    row_map = {}
    current_row = 1
    
    if show_raw_mag:
        row_map['raw'] = current_row
        current_row += 1
    
    if show_residuals:
        row_map['resid'] = current_row
        current_row += 1
        
    if phase_requested:
        row_map['phase'] = current_row
        current_row += 1
        
    n_rows = current_row - 1
    if n_rows == 0:
        return {
            "figure": _status_figure("No panels selected. Enable Raw, Residuals, or Phase-fold.", theme=theme),
            "camera_options": [{"label": f"{cam}", "value": str(cam)} for cam in camera_ids],
            "camera_values": selected,
            "stat_rows": [],
            "status": "ok",
            "status_message": "No panels selected.",
            "camera_diagnostics": camera_diagnostics,
            "warnings": ["No panels selected"],
        }

    # Calculate row heights
    # Logic:
    # - If 1 row: 1.0
    # - If 2 rows: raw/resid split or raw/phase split or resid/phase split
    # - If 3 rows: raw/resid/phase
    
    row_heights = []
    
    if n_rows == 1:
        row_heights = [1.0]
    elif n_rows == 2:
        if show_raw_mag and show_residuals:
            row_heights = [1.0 - residual_fraction, residual_fraction]
        elif show_raw_mag and phase_requested:
            row_heights = [0.5, 0.5]
        else:
            # Resid + Phase or just two unknown panels (unlikely with current logic)
            row_heights = [0.5, 0.5]
    elif n_rows == 3:
        lower_fraction = float(np.clip(residual_fraction, 0.15, 0.425))
        main_fraction = 1.0 - (2.0 * lower_fraction)
        row_heights = [main_fraction, lower_fraction, lower_fraction]

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=(not phase_requested) if show_raw_mag else False,
        vertical_spacing=0.05,
        row_heights=row_heights,
    )

    band_markers = {0: "circle", 1: "square"}
    is_flux = yaxis_mode == "flux"

    for band in active_bands:
        bdf = band_dfs.get(band)
        if bdf is None or bdf.empty:
            continue
        for cam in selected:
            cdf = bdf[bdf["camera_label"] == cam]
            if cdf.empty:
                continue
            trace_name = f"{cam} ({band_labels[band]})"
            color = stable_camera_color(cam)
            err = cdf["error"].to_numpy() if "error" in cdf.columns else np.full(len(cdf), np.nan)
            resid = cdf["resid"].to_numpy() if "resid" in cdf.columns else np.full(len(cdf), np.nan)
            baseline = cdf["baseline"].to_numpy() if "baseline" in cdf.columns else np.full(len(cdf), np.nan)
            jd_full = cdf["JD_plot"].to_numpy() + JD_OFFSET
            mag_raw = cdf["mag"].to_numpy()

            if show_raw_mag:
                y_raw = _mag_to_flux(mag_raw) if is_flux else mag_raw
                err_raw = _flux_err_from_mag_err(y_raw, err) if is_flux else err
                hover_raw = np.column_stack([jd_full, err, resid, baseline, err_raw, mag_raw])
                raw_hovertemplate = (
                    "<b>%{fullData.name}</b><br>"
                    "JD: %{customdata[0]:.5f}<br>"
                    f"JD - {int(JD_OFFSET)}: %{{x:.5f}}<br>"
                )
                if is_flux:
                    raw_hovertemplate += (
                        "F: %{y:.4e}<br>"
                        "σ<sub>F</sub>: %{customdata[4]:.3e}<br>"
                        "m: %{customdata[5]:.4f}<br>"
                        "σ<sub>m</sub>: %{customdata[1]:.4f}<br>"
                        "Δm: %{customdata[2]:.4f}<br>"
                        "m<sub>base</sub>: %{customdata[3]:.4f}<extra></extra>"
                    )
                else:
                    raw_hovertemplate += (
                        "m: %{y:.4f}<br>"
                        "σ<sub>m</sub>: %{customdata[1]:.4f}<br>"
                        "Δm: %{customdata[2]:.4f}<br>"
                        "m<sub>base</sub>: %{customdata[3]:.4f}<extra></extra>"
                    )

                fig.add_trace(
                    go.Scatter(
                        x=cdf["JD_plot"],
                        y=y_raw,
                        mode="markers",
                        name=trace_name,
                        marker={
                            "size": 7,
                            "symbol": band_markers[band],
                            "color": color,
                            "line": {"width": 0.8, "color": colors["marker_line"]},
                        },
                        error_y={"type": "data", "array": err_raw, "visible": True, "thickness": 1, "width": 0, "color": color},
                        customdata=hover_raw,
                        hovertemplate=raw_hovertemplate,
                    ),
                    row=row_map['raw'],
                    col=1,
                )

            if show_residuals:
                y_resid = (_mag_to_flux(resid) - 1.0) if is_flux else resid
                err_resid = _flux_err_from_mag_err(_mag_to_flux(resid), err) if is_flux else err
                hover_resid = np.column_stack([jd_full, err, resid, baseline, err_resid, mag_raw])
                resid_hovertemplate = (
                    "<b>%{fullData.name}</b><br>"
                    "JD: %{customdata[0]:.5f}<br>"
                    f"JD - {int(JD_OFFSET)}: %{{x:.5f}}<br>"
                )
                if is_flux:
                    resid_hovertemplate += (
                        "ΔF/F: %{y:.4f}<br>"
                        "σ<sub>F</sub>: %{customdata[4]:.3e}<br>"
                        "Δm: %{customdata[2]:.4f}<extra></extra>"
                    )
                else:
                    resid_hovertemplate += (
                        "Δm: %{y:.4f}<br>"
                        "σ<sub>m</sub>: %{customdata[1]:.4f}<extra></extra>"
                    )

                fig.add_trace(
                    go.Scatter(
                        x=cdf["JD_plot"],
                        y=y_resid,
                        mode="markers",
                        name=trace_name,
                        showlegend=False,
                        marker={
                            "size": 6,
                            "symbol": band_markers[band],
                            "color": color,
                            "line": {"width": 0.8, "color": colors["marker_line"]},
                        },
                        customdata=hover_resid,
                        hovertemplate=resid_hovertemplate,
                    ),
                    row=row_map['resid'],
                    col=1,
                )

        if show_raw_mag and show_baseline and "baseline" in bdf.columns:
            for cam in selected:
                cbase = bdf[(bdf["camera_label"] == cam) & np.isfinite(bdf["baseline"])].sort_values("JD_plot")
                if cbase.empty:
                    continue
                y_base = _mag_to_flux(cbase["baseline"].to_numpy()) if is_flux else cbase["baseline"].to_numpy()
                fig.add_trace(
                    go.Scatter(
                        x=cbase["JD_plot"],
                        y=y_base,
                        mode="lines",
                        showlegend=False,
                        line={"width": 1.6, "color": stable_camera_color(cam)},
                        opacity=baseline_opacity,
                        customdata=cbase["JD_plot"].to_numpy() + JD_OFFSET,
                        hovertemplate=(
                            f"JD: %{{customdata:.5f}}<br>JD - {int(JD_OFFSET)}: %{{x:.5f}}<br>F<sub>base</sub>: %{{y:.4e}}<extra></extra>"
                            if is_flux
                            else f"JD: %{{customdata:.5f}}<br>JD - {int(JD_OFFSET)}: %{{x:.5f}}<br>m<sub>base</sub>: %{{y:.4f}}<extra></extra>"
                        ),
                    ),
                    row=row_map['raw'],
                    col=1,
                )

    event_entries = _event_entries(payload, jd_offset, run_params)
    if show_event_markers and show_raw_mag:
        raw_row = row_map['raw']
        raw_xref = _subplot_axis_ref(raw_row, "x")
        raw_yref = _subplot_axis_ref(raw_row, "y")
        raw_y_domain_ref = _subplot_domain_ref(raw_row, "y")
        visible_raw_df = df[df["v_g_band"].isin(active_bands)].copy() if "v_g_band" in df.columns else df.copy()
        if visible_raw_df.empty:
            visible_raw_df = df.copy()
        raw_event_x_values = visible_raw_df["JD_plot"].to_numpy(dtype=float)
        raw_event_y_values = (
            _mag_to_flux(visible_raw_df["mag"].to_numpy(dtype=float))
            if is_flux
            else visible_raw_df["mag"].to_numpy(dtype=float)
        )
        finite_raw_event_y = raw_event_y_values[np.isfinite(raw_event_y_values)]
        if finite_raw_event_y.size:
            raw_event_span = float(np.nanmax(finite_raw_event_y) - np.nanmin(finite_raw_event_y))
        else:
            raw_event_span = 1.0
        raw_event_pad = raw_event_span * 0.10
        if not np.isfinite(raw_event_pad) or raw_event_pad <= 0:
            raw_event_pad = 0.05 * max(1.0, abs(float(finite_raw_event_y[0])) if finite_raw_event_y.size else 1.0)
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
                    color = f"rgba(255,107,107,{alpha:.3f})"
                else:
                    color = f"rgba(0,150,255,{alpha:.3f})"

            event_x = float(entry["x0"])
            marker_y, label_y = _event_annotation_y(
                raw_event_x_values,
                raw_event_y_values,
                event_x,
                float(entry["half_width"]),
                kind=str(entry["kind"]),
                is_flux=is_flux,
                pad=raw_event_pad,
            )
            label_yanchor = "top" if entry["kind"] == "dip" else "bottom"
            hover_text = (
                f"{str(entry['kind']).title()} event<br>"
                f"t0 [JD]: {float(entry['t0']):.5f}<br>"
                f"w: {float(entry['half_width']) / 2.0:.3f}<br>"
                f"log BF: {bf_text}<br>"
                f"log BF thr: {logbf_thr_text}<br>"
                f"sig thr: {sig_thr_text}<br>"
                f"morph: {entry['morph'] or 'n/a'}<br>"
                f"c: {float(entry['confidence']):.2f}"
            )
            fig.add_shape(
                type="line",
                x0=event_x,
                x1=event_x,
                y0=0.0,
                y1=1.0,
                xref=raw_xref,
                yref=raw_y_domain_ref,
                line={"color": color, "dash": "dash", "width": 1.8},
            )

            if show_diagnostics and float(entry["half_width"]) > 0:
                half_width = float(entry["half_width"])
                fig.add_shape(
                    type="rect",
                    x0=event_x - half_width,
                    x1=event_x + half_width,
                    y0=0.0,
                    y1=1.0,
                    xref=raw_xref,
                    yref=raw_y_domain_ref,
                    fillcolor=color,
                    opacity=0.11,
                    line={"width": 0},
                    layer="below",
                )
                fig.add_annotation(
                    x=event_x,
                    y=label_y,
                    xref=raw_xref,
                    yref=raw_yref,
                    text=(
                        f"{str(entry['kind']).title()} thr logBF={logbf_thr_text}, sig={sig_thr_text}"
                    ),
                    showarrow=False,
                    font={"size": 9, "color": colors["annotation"]},
                    yanchor=label_yanchor,
                    bgcolor=colors["paper_bg"],
                    opacity=0.92,
                )

            fig.add_annotation(
                x=event_x,
                y=marker_y,
                xref=raw_xref,
                yref=raw_yref,
                text="◆",
                showarrow=False,
                font={"size": 18, "color": color},
                hovertext=hover_text,
            )

    phase_diag: dict[str, object] = {}
    if phase_enabled and 'phase' in row_map and phase_period is not None:
        # Phase diagnostics use residuals (mag - baseline) from band_dfs.
        phase_zero_jd = resolve_phase_epoch(df)
        phase_inputs = [
            band_dfs[band]
            for band in active_bands
            if band in band_dfs and band_dfs[band] is not None and not band_dfs[band].empty
        ]
        if phase_inputs:
            phase_source_df = pd.concat(phase_inputs, ignore_index=True)
        else:
            phase_source_df = pd.DataFrame()

        if phase_panel_mode == "time":
            phase_time_df = pd.DataFrame()
            if not phase_source_df.empty:
                phase_time_df, phase_diag = phase_time_dataframe(
                    phase_source_df,
                    float(phase_period),
                    epoch_jd=phase_zero_jd,
                    value_mode="resid",
                    duplicate_cycles=True,
                )
            color_values = (
                pd.to_numeric(phase_time_df.get("phase_value"), errors="coerce").to_numpy(dtype=float)
                if not phase_time_df.empty and "phase_value" in phase_time_df.columns
                else np.array([], dtype=float)
            )
            cmin, cmax = _zero_centered_color_bounds(color_values)
            colorbar_shown = False
            for band in active_bands:
                if phase_time_df.empty or "v_g_band" not in phase_time_df.columns:
                    continue
                band_phase_df = phase_time_df[pd.to_numeric(phase_time_df["v_g_band"], errors="coerce") == band]
                if band_phase_df.empty:
                    continue
                for cam in selected:
                    cdf = band_phase_df[band_phase_df["camera_label"] == cam]
                    if cdf.empty:
                        continue
                    err = cdf["error"].to_numpy() if "error" in cdf.columns else np.full(len(cdf), np.nan)
                    resid = pd.to_numeric(cdf["phase_value"], errors="coerce").to_numpy(dtype=float)
                    jd_full_phase = cdf["JD_plot"].to_numpy() + JD_OFFSET
                    hover_phase = np.column_stack([
                        jd_full_phase,
                        err,
                        pd.to_numeric(cdf["mag"], errors="coerce").to_numpy(dtype=float),
                        resid,
                        pd.to_numeric(cdf["cycle"], errors="coerce").to_numpy(dtype=float),
                        pd.to_numeric(cdf["v_g_band"], errors="coerce").to_numpy(dtype=float),
                    ])
                    marker = {
                        "size": 6,
                        "symbol": band_markers[band],
                        "color": resid,
                        "colorscale": PHASE_TIME_COLORSCALE,
                        "line": {"width": 0.45, "color": colors["marker_line"]},
                        "showscale": not colorbar_shown,
                        "colorbar": {"title": "Δm", "len": 0.34},
                    }
                    if cmin is not None and cmax is not None:
                        marker["cmin"] = cmin
                        marker["cmax"] = cmax
                    fig.add_trace(
                        go.Scatter(
                            x=cdf["phase"],
                            y=cdf["cycle"],
                            mode="markers",
                            name=f"{cam} ({band_labels[band]})",
                            showlegend=False,
                            marker=marker,
                            customdata=hover_phase,
                            hovertemplate=(
                                "<b>%{fullData.name}</b><br>"
                                "φ: %{x:.4f}<br>"
                                "cycle: %{customdata[4]:.0f}<br>"
                                "JD: %{customdata[0]:.5f}<br>"
                                "Δm: %{customdata[3]:.4f}<br>"
                                "m: %{customdata[2]:.4f}<br>"
                                "σ<sub>m</sub>: %{customdata[1]:.4f}<br>"
                                "band: %{customdata[5]:.0f}<extra></extra>"
                            ),
                        ),
                        row=row_map['phase'],
                        col=1,
                    )
                    colorbar_shown = True
        else:
            phase_bdf = pd.DataFrame()
            if not phase_source_df.empty:
                phase_bdf, phase_diag = phase_fold_dataframe(
                    phase_source_df,
                    float(phase_period),
                    epoch_jd=phase_zero_jd,
                    value_mode="resid",
                    duplicate_cycles=True,
                )

            for band in active_bands:
                if phase_bdf.empty or "v_g_band" not in phase_bdf.columns:
                    continue
                band_phase_df = phase_bdf[pd.to_numeric(phase_bdf["v_g_band"], errors="coerce") == band]
                if band_phase_df.empty:
                    continue
                for cam in selected:
                    cdf = band_phase_df[band_phase_df["camera_label"] == cam]
                    if cdf.empty:
                        continue
                    color = stable_camera_color(cam)
                    err = cdf["error"].to_numpy() if "error" in cdf.columns else np.full(len(cdf), np.nan)
                    resid = cdf["phase_value"].to_numpy()
                    y_phase = (_mag_to_flux(resid) - 1.0) if is_flux else resid
                    err_phase = _flux_err_from_mag_err(_mag_to_flux(resid), err) if is_flux else err
                    jd_full_phase = cdf["JD_plot"].to_numpy() + JD_OFFSET
                    hover_phase = np.column_stack([jd_full_phase, err, cdf["mag"].to_numpy(), resid, err_phase])
                    phase_hovertemplate = (
                        "<b>%{fullData.name}</b><br>"
                        "φ: %{x:.4f}<br>"
                        "JD: %{customdata[0]:.5f}<br>"
                    )
                    if is_flux:
                        phase_hovertemplate += (
                            "ΔF/F: %{y:.4f}<br>"
                            "σ<sub>F</sub>: %{customdata[4]:.3e}<br>"
                            "Δm: %{customdata[3]:.4f}<br>"
                            "m: %{customdata[2]:.4f}<br>"
                            "σ<sub>m</sub>: %{customdata[1]:.4f}<extra></extra>"
                        )
                    else:
                        phase_hovertemplate += (
                            "Δm: %{y:.4f}<br>"
                            "σ<sub>m</sub>: %{customdata[1]:.4f}<br>"
                            "m: %{customdata[2]:.4f}<extra></extra>"
                        )

                    fig.add_trace(
                        go.Scatter(
                            x=cdf["phase"],
                            y=y_phase,
                            mode="markers",
                            name=f"{cam} ({band_labels[band]})",
                            showlegend=False,
                            marker={
                                "size": 6,
                                "symbol": band_markers[band],
                                "color": color,
                                "line": {"width": 0.7, "color": "rgba(10,10,10,0.95)"},
                            },
                            error_y={"type": "data", "array": err_phase, "visible": True, "thickness": 1, "width": 0, "color": color},
                            customdata=hover_phase,
                            hovertemplate=phase_hovertemplate,
                        ),
                        row=row_map['phase'],
                        col=1,
                    )

            phase_lag = float(phase_diag.get("phase_lag_g_v_cycles", np.nan))
            phase_lag_abs = float(phase_diag.get("phase_lag_g_v_abs_cycles", np.nan))
            if np.isfinite(phase_lag):
                lag_text = f"g-V lag {phase_lag:+.3f} cyc"
                if np.isfinite(phase_lag_abs):
                    lag_text += f" (|lag| {phase_lag_abs:.3f})"
                fig.add_annotation(
                    text=lag_text,
                    x=0.99,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    xanchor="right",
                    yanchor="top",
                    showarrow=False,
                    font={"size": 11, "color": colors["text"]},
                    bgcolor=colors["paper_bg"],
                    bordercolor=colors["grid"],
                    borderwidth=1,
                    opacity=0.9,
                )

        fig.add_vline(x=0.0, line_color=colors["guide_line"], line_dash="dot", line_width=1.0, row=row_map['phase'], col=1)
        fig.add_vline(x=1.0, line_color=colors["guide_line"], line_dash="dot", line_width=1.0, row=row_map['phase'], col=1)
        fig.add_vline(x=2.0, line_color=colors["guide_line"], line_dash="dot", line_width=1.0, row=row_map['phase'], col=1)
        if phase_panel_mode == "fold":
            fig.add_hline(y=0.0, line_color=colors["guide_line"], line_dash="dot", row=row_map['phase'], col=1)

    if show_raw_mag:
        # Explicitly calculate range to ensure full visibility
        if is_flux:
            y_vals = _mag_to_flux(df["mag"].to_numpy())
            if show_baseline and "baseline" in df.columns:
                b_vals = df["baseline"].dropna().to_numpy()
                if b_vals.size > 0:
                    y_vals = np.concatenate([y_vals, _mag_to_flux(b_vals)])
        else:
            y_vals = df["mag"].to_numpy()
            if show_baseline and "baseline" in df.columns:
                b_vals = df["baseline"].dropna().to_numpy()
                if b_vals.size > 0:
                    y_vals = np.concatenate([y_vals, b_vals])
        
        if y_vals.size > 0:
            y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)
            y_pad_fraction = 0.10 if show_event_markers and event_entries else 0.05
            y_pad = (y_max - y_min) * y_pad_fraction
            if y_pad == 0:
                y_pad = 0.5 if not is_flux else y_max * 0.05
            if is_flux:
                fig.update_yaxes(
                    title_text=r"$F$ [arb]",
                    row=row_map['raw'],
                    col=1,
                    range=[max(0, y_min - y_pad), y_max + y_pad],
                )
            else:
                fig.update_yaxes(
                    title_text=r"$m$ [mag]",
                    row=row_map['raw'],
                    col=1,
                    range=[y_max + y_pad, y_min - y_pad],
                )
        else:
            fig.update_yaxes(
                title_text=r"$F$ [arb]" if is_flux else r"$m$ [mag]",
                row=row_map['raw'],
                col=1,
                autorange="reversed" if not is_flux else True,
            )
    
    if show_residuals:
        fig.update_yaxes(
            title_text=r"$\Delta F/F$" if is_flux else r"$\Delta m$ [mag]",
            row=row_map['resid'],
            col=1,
            autorange="reversed" if not is_flux else True,
        )
        fig.add_hline(y=0.0, line_color=colors["guide_line"], line_dash="dot", row=row_map['resid'], col=1)

    if phase_enabled and 'phase' in row_map and phase_period is not None:
        phase_axis_title = (
            rf"$\phi\ \mathrm{{vs.}}\ E\,(P={phase_period:.5f}\,\mathrm{{d}})$"
            if phase_panel_mode == "time"
            else rf"$\phi\,(P={phase_period:.5f}\,\mathrm{{d}})$"
        )
        fig.update_xaxes(title_text=phase_axis_title, row=row_map['phase'], col=1, range=[-0.02, 2.02])
        if phase_panel_mode == "time":
            fig.update_yaxes(title_text="Cycle E", row=row_map['phase'], col=1, autorange=True)
        else:
            fig.update_yaxes(
                title_text=r"$\Delta F/F$" if is_flux else r"$\Delta m$ [mag]",
                row=row_map['phase'],
                col=1,
                autorange="reversed" if not is_flux else True,
            )
    elif phase_requested and 'phase' in row_map:
        phase_row = row_map['phase']
        fig.update_xaxes(title_text=r"$\phi$", row=phase_row, col=1, range=[0.0, 1.0])
        fig.update_yaxes(
            title_text=r"$\Delta F/F$" if is_flux else r"$\Delta m$ [mag]",
            row=phase_row,
            col=1,
            visible=True,
        )
        if phase_period_pending:
            phase_placeholder = "Auto PDM period search is running..."
        elif suppress_catalog_phase_period:
            phase_placeholder = "No automatic PDM period available. Run Find Period or enter Manual P."
        else:
            phase_placeholder = "No phase period available. Run Find Period or enter Manual P."
        fig.add_annotation(
            text=phase_placeholder,
            x=0.5,
            y=0.5,
            xref=_subplot_domain_ref(phase_row, "x"),
            yref=_subplot_domain_ref(phase_row, "y"),
            showarrow=False,
            font={"size": 12, "color": colors["text"]},
        )

    # Set JD axis on the bottom-most plot that uses JD
    jd_axis_row = None
    if show_residuals:
        jd_axis_row = row_map['resid']
    elif show_raw_mag:
        jd_axis_row = row_map['raw']
        
    if jd_axis_row is not None:
        fig.update_xaxes(title_text=rf"$\mathrm{{JD}} - {int(JD_OFFSET)}$", row=jd_axis_row, col=1)
        # Link x-axes if both raw and resid are present
        if show_raw_mag and show_residuals:
             fig.update_xaxes(matches="x", row=row_map['resid'], col=1)

    # Overlay external light curves on raw magnitude panel
    ext_trace_start = len(fig.data)
    ext_source_ranges: dict[str, tuple[int, int]] = {}
    requested_source = str(external_source_view or "all").strip().lower()
    active_external_lcs: dict[str, Path] | None = None
    if external_lcs:
        if requested_source == "asassn":
            active_external_lcs = {}
        elif requested_source in {"", "all"}:
            active_external_lcs = dict(external_lcs)
        else:
            active_external_lcs = ({requested_source: external_lcs[requested_source]}
                                   if requested_source in external_lcs else {})
    if active_external_lcs and 'raw' in row_map:
        mag_anchor = _coerce_finite_float(payload.get("baseline_mag"))
        if mag_anchor is None:
            finite_mag = pd.to_numeric(df["mag"], errors="coerce").to_numpy(dtype=float)
            finite_mag = finite_mag[np.isfinite(finite_mag)]
            if finite_mag.size:
                mag_anchor = float(np.nanmedian(finite_mag))
        _overlay_external_lcs(
            fig,
            row_map['raw'],
            active_external_lcs,
            jd_offset,
            colors,
            theme,
            is_flux,
            ext_source_ranges,
            ext_trace_start,
            mag_anchor=mag_anchor,
            warnings=warnings,
        )
        if is_flux and any(bool(_EXTERNAL_LC_SPECS.get(src, {}).get("is_flux", False)) for src in active_external_lcs):
            fig.update_yaxes(autorange=True, row=row_map['raw'], col=1)

    fig.update_layout(
        title=_build_title(payload, df),
        title_font={"size": 14, "color": colors["title"]},
        paper_bgcolor=colors["paper_bg"],
        plot_bgcolor=colors["plot_bg"],
        margin={"l": 55, "r": 20, "t": 68, "b": 44},
        font={"color": colors["text"], "family": "Monaco, Courier New, monospace", "size": 11},
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

    # Apply external source visibility from sidebar control.
    if ext_source_ranges:
        n_total = len(fig.data)
        all_visible = [True] * n_total
        asassn_only = [True] * ext_trace_start + [False] * (n_total - ext_trace_start)
        visible = all_visible

        if requested_source == "asassn":
            visible = asassn_only
        elif requested_source not in {"", "all"}:
            source_range = ext_source_ranges.get(requested_source)
            if source_range is None:
                visible = asassn_only
                warnings.append(f"{requested_source.upper()} light curve is not available for this candidate.")
            else:
                start, end = source_range
                visible = [True] * ext_trace_start + [False] * (n_total - ext_trace_start)
                for i in range(start, end):
                    visible[i] = True

        for trace_idx, is_visible in enumerate(visible):
            fig.data[trace_idx].visible = bool(is_visible)
    elif external_lcs and requested_source not in {"", "all", "asassn"}:
        warnings.append(f"{requested_source.upper()} light curve is not available for this candidate.")

    fig.update_xaxes(showgrid=True, gridcolor=colors["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=colors["grid"], zeroline=False)

    status_message = ""
    if phase_enabled and phase_period is not None:
        phase_bits = [
            f"{'Phase-time' if phase_panel_mode == 'time' else 'Phase-fold'} P={float(phase_period):.5f} d"
        ]
        if phase_source:
            phase_bits.append(f"source={phase_source}")
        phase_lag = float(phase_diag.get("phase_lag_g_v_cycles", np.nan))
        phase_lag_abs = float(phase_diag.get("phase_lag_g_v_abs_cycles", np.nan))
        if np.isfinite(phase_lag):
            phase_bits.append(f"g-V lag={phase_lag:+.3f} cycles")
            if np.isfinite(phase_lag_abs):
                phase_bits.append(f"|lag|={phase_lag_abs:.3f}")
        status_message = "; ".join(phase_bits)
    elif phase_requested and phase_period_pending:
        status_message = "Auto PDM: searching..."
    elif phase_requested and suppress_catalog_phase_period:
        status_message = "Auto PDM: no valid period"

    camera_options = [{"label": f"{cam}", "value": str(cam)} for cam in camera_ids]
    return {
        "figure": fig,
        "camera_options": camera_options,
        "camera_values": selected,
        "stat_rows": _build_stat_rows(payload, df, filtered_cameras),
        "status": "ok",
        "status_message": status_message,
        "camera_diagnostics": camera_diagnostics,
        "warnings": warnings,
    }
