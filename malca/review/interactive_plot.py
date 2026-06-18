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
from malca.lightcurve_io import load_lightcurve_df, stable_camera_color, to_asassn_algorithm_frame
from malca.lightcurve_publication import PUBLICATION_PLOTLY_FONT
from malca.phase import BAND_LABELS, phase_fold_dataframe, phase_time_dataframe, resolve_phase_epoch, resolve_phase_period
from malca.utils import (
    clean_lc,
    identify_bad_cameras,
    identify_catastrophic_outlier_cameras,
    identify_offset_cameras,
)
from malca.config import (
    JD_OFFSET, MJD_TO_JD, GAIA_TCB_EPOCH_JD, TESS_BTJD_OFFSET, KEPLER_BKJD_OFFSET,
    SKYPATROL_JD_OFFSET,
    REVIEW_CACHE_LIMIT, REVIEW_MAX_EXTERNAL_POINTS, REVIEW_RESIDUAL_FRACTION,
)
from malca.config import (
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    CLEAN_LC_MAX_ERROR_ABSOLUTE,
    CLEAN_LC_MAX_ERROR_SIGMA,
    OFFSET_CAMERA_SIGMA_THRESHOLD,
)
from malca.review.lightcurve_sources import normalize_external_lc_dataframe


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


def _bundle_lightcurve_dir(plot_dir: Path | None) -> Path | None:
    """Resolve bundle_assets/lightcurves for either a run dir or its plots/ anchor."""
    if plot_dir is None:
        return None
    direct = plot_dir / "bundle_assets" / "lightcurves"
    if direct.is_dir():
        return direct
    sibling = plot_dir.parent / "bundle_assets" / "lightcurves"
    if sibling.is_dir():
        return sibling
    if plot_dir.name == "plots":
        return sibling
    return direct


def resolve_lightcurve_path(payload: dict, plot_dir: Path | None) -> Path | None:
    """Resolve a candidate light-curve path for native plotting."""
    bundle_dir = _bundle_lightcurve_dir(plot_dir)

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
    df = to_asassn_algorithm_frame(df)
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


def _event_time_to_lc_scale(value: object, lc_median: float | None, jd_offset: float) -> float | None:
    """Convert stored event times to the light-curve time scale before plotting."""
    t = _coerce_finite_float(value)
    if t is None:
        return None
    scale_anchor = lc_median
    if scale_anchor is None or not np.isfinite(scale_anchor):
        scale_anchor = jd_offset
    if scale_anchor > 2_000_000.0:
        if t > 2_000_000.0:
            return t
        if t > 50_000.0:
            return t + MJD_TO_JD
        return t + SKYPATROL_JD_OFFSET
    if scale_anchor > 50_000.0:
        if t > 2_000_000.0:
            return t - MJD_TO_JD
        if t < 50_000.0:
            return t + SKYPATROL_JD_OFFSET - MJD_TO_JD
        return t
    if t > 2_000_000.0:
        return t - SKYPATROL_JD_OFFSET
    if t > 50_000.0:
        return t + MJD_TO_JD - SKYPATROL_JD_OFFSET
    return t


def _event_entries(
    payload: dict,
    jd_offset: float,
    run_params: dict | None,
    *,
    lc_median: float | None = None,
) -> list[dict[str, object]]:
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
        None if lc_median is None else float(lc_median),
    )
    cached = _cache_get(_EVENT_CACHE, key)
    if cached is not None:
        return [dict(x) for x in cached]

    entries: list[dict[str, object]] = []
    for prefix, color in (("dip", DIP_EVENT_COLOR), ("jump", JUMP_EVENT_COLOR)):
        t0 = _parse_num(payload, f"{prefix}_best_t0")
        if t0 is None:
            continue
        t0_lc = _event_time_to_lc_scale(t0, lc_median, jd_offset)
        if t0_lc is None:
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
                "t0": t0_lc,
                "t0_raw": t0,
                "x0": t0_lc - jd_offset,
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
        font={"size": 13, "color": colors["text"], "family": PUBLICATION_PLOTLY_FONT},
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
    external_source_view: str | list[str] = "asassn",
    external_panel_mode: Literal["overlay", "split"] = "overlay",
    selected_bands: list[str] | None = None,
    native_color_mode: Literal["camera", "band"] = "camera",
    candidate_id: str | None = None,
) -> dict:
    """Build a native Plotly light-curve figure for review mode."""
    from malca.review.lightcurve_assembly import ReviewPlotRequest, assemble_review_lightcurve_plot
    from malca.review.lightcurve_plotly import render_review_lightcurve_plotly

    request = ReviewPlotRequest.from_kwargs(
        payload,
        plot_dir=plot_dir,
        selected_cameras=selected_cameras,
        filter_bad_cameras=filter_bad_cameras,
        show_baseline=show_baseline,
        show_event_markers=show_event_markers,
        show_residuals=show_residuals,
        show_phase_fold=show_phase_fold,
        phase_panel_mode=phase_panel_mode,
        show_raw_mag=show_raw_mag,
        override_period=override_period,
        override_period_source=override_period_source,
        phase_period_pending=phase_period_pending,
        suppress_catalog_phase_period=suppress_catalog_phase_period,
        show_diagnostics=show_diagnostics,
        confidence_colors=confidence_colors,
        run_params=run_params,
        residual_fraction=residual_fraction,
        baseline_opacity=baseline_opacity,
        yaxis_mode=yaxis_mode,
        external_lcs=external_lcs,
        external_source_view=external_source_view,
        external_panel_mode=external_panel_mode,
        selected_bands=selected_bands,
        native_color_mode=native_color_mode,
        candidate_id=candidate_id,
        discover_external=True,
    )
    spec = assemble_review_lightcurve_plot(request)
    return render_review_lightcurve_plotly(spec, theme=theme, uirevision_key=uirevision_key)

