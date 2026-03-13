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
from malca.utils import (
    clean_lc,
    identify_bad_cameras,
    identify_catastrophic_outlier_cameras,
    identify_offset_cameras,
)
from malca.config.config_pipeline import (
    JD_OFFSET, MJD_TO_JD, GAIA_TCB_EPOCH_JD, TESS_BTJD_OFFSET, KEPLER_BKJD_OFFSET,
    REVIEW_CACHE_LIMIT, REVIEW_MAX_EXTERNAL_POINTS, REVIEW_RESIDUAL_FRACTION,
)
from malca.config.config_filters import (
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
        if candidate.suffix == ".dat2":
            candidate_names.append(candidate.with_suffix(".raw2").name)
        if candidate.exists():
            return candidate

    candidate_id = payload.get("candidate_id")
    if candidate_id is not None:
        cid = str(candidate_id).strip()
        if cid:
            candidate_names.extend([f"{cid}.dat2", f"{cid}.raw2"])

    asas_sn_id = payload.get("asas_sn_id")
    if asas_sn_id is not None:
        sid = str(asas_sn_id).strip()
        if sid:
            candidate_names.extend([f"{sid}.dat2", f"{sid}.raw2"])

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
    """Return preferred phase-fold period from payload metadata.

    Checks in order of reliability:
    1. Validated phase period (consensus or significant LSP)
    2. Period consensus from post-filter catalog validation
    3. Specific catalog periods — checks both vetting names and post-filter names
    4. Generic catalog period
    5. Raw LSP period
    """
    # 1. Validated pipeline period
    p = _parse_num(payload, "phase_period_days")
    if p is not None and p > 0:
        return float(p)

    # 2. Post-filter consensus period
    p = _parse_num(payload, "period_consensus_days")
    if p is not None and p > 0:
        return float(p)

    # 3. Known catalog periods (check both naming conventions)
    # Each tuple: (vetting column name, post_filter column name)
    for keys in (
        ("vsx_period", "period_vsx_days"),
        ("asassn_var_period", "period_asassn_var_days"),
        ("gaia_eb_period", "period_gaia_eb_days"),
        ("ztf_var_period", "period_ztf_periodic_days"),
    ):
        for key in keys:
            p = _parse_num(payload, key)
            if p is not None and p > 0:
                return float(p)

    # 4. Generic catalog period match
    p = _parse_num(payload, "catalog_period")
    if p is not None and p > 0:
        return float(p)

    # 5. Raw LSP period (least reliable, often alias)
    p = _parse_num(payload, "lsp_period")
    if p is not None and p > 0:
        return float(p)

    return None


def _phase_fold_df(
    df: pd.DataFrame,
    period_days: float,
    *,
    phase_zero_jd: float | None = None,
) -> pd.DataFrame:
    """Phase-fold dataframe to 0-2 cycles with an optional shared phase epoch."""
    out = df.copy()
    out = out[np.isfinite(out["JD"]) & np.isfinite(out["mag"])].copy()
    if out.empty:
        return out
    jd0 = float(phase_zero_jd) if phase_zero_jd is not None else float(out["JD"].min())
    out["phase"] = ((out["JD"].to_numpy(dtype=float) - jd0) / float(period_days)) % 1.0
    wrap = out.copy()
    wrap["phase"] = wrap["phase"] + 1.0
    return pd.concat([out, wrap], ignore_index=True)


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
        if is_flux_source and not is_flux:
            continue
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

            y = pd.to_numeric(band_df[actual_mag], errors="coerce").to_numpy()
            good = np.isfinite(band_x) & np.isfinite(y)
            if not good.any():
                continue

            if not is_flux_source and is_flux:
                y = np.power(10.0, -0.4 * y)

            err_array = None
            if actual_err and actual_err in band_df.columns:
                ev = pd.to_numeric(band_df[actual_err], errors="coerce").to_numpy()
                if np.isfinite(ev[good]).any():
                    if not is_flux_source and is_flux:
                        err_array = 0.921 * y[good] * ev[good]
                    else:
                        err_array = ev[good]

            good_idx = np.flatnonzero(good)
            if good_idx.size > _MAX_EXTERNAL_TRACE_POINTS:
                step = int(np.ceil(good_idx.size / float(_MAX_EXTERNAL_TRACE_POINTS)))
                good_idx = good_idx[::step]

            x_vals = band_x[good_idx]
            y_vals = y[good_idx]
            jd_vals = x_vals + plot_jd_offset
            if err_array is not None:
                err_array = err_array[good_idx]

            if err_array is not None:
                custom_ext = np.column_stack([jd_vals, err_array])
                hover_ext = (
                    f"<b>{band_info['label']}</b><br>"
                    "JD: %{customdata[0]:.5f}<br>"
                    f"JD - {int(plot_jd_offset)}: %{{x:.5f}}<br>"
                    + ("F: %{y:.4e}<br>" if is_flux else "m: %{y:.4f}<br>")
                    + ("σ<sub>F</sub>: %{customdata[1]:.3e}<extra></extra>" if is_flux else "σ<sub>m</sub>: %{customdata[1]:.4f}<extra></extra>")
                )
            else:
                custom_ext = jd_vals.reshape(-1, 1)
                hover_ext = (
                    f"<b>{band_info['label']}</b><br>"
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
                    name=band_info["label"],
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
    show_raw_mag: bool = True,
    override_period: float | None = None,
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

    band_labels = {0: "g", 1: "V"}
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
    phase_source = ""
    if show_phase_fold:
        if override_period is not None and override_period > 0:
            phase_period = override_period
            phase_source = "manual/search"
        else:
            phase_period = _phase_period_days(payload)
            phase_source = "catalog/pipeline"
    else:
        phase_period = None
    phase_enabled = bool(show_phase_fold and phase_period is not None)
    if show_phase_fold and not phase_enabled:
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
        
    if phase_enabled:
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
        elif show_raw_mag and phase_enabled:
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
        shared_xaxes=(not phase_enabled) if show_raw_mag else False,
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
                        name=f"{cam} ({band_labels[band]})",
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
        if is_flux and not df.empty:
            y_ref = float(_mag_to_flux(np.array([df["mag"].min()]))[0])
        else:
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

            fig.add_vline(x=float(entry["x0"]), line_color=color, line_dash="dash", line_width=1.8, row=row_map['raw'], col=1)

            if show_diagnostics and float(entry["half_width"]) > 0:
                fig.add_vrect(
                    x0=float(entry["x0"]) - float(entry["half_width"]),
                    x1=float(entry["x0"]) + float(entry["half_width"]),
                    fillcolor=color,
                    opacity=0.11,
                    line_width=0,
                    layer="below",
                    row=row_map['raw'],
                    col=1,
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
                    row=row_map['raw'],
                    col=1,
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
                        f"t<sub>0</sub> (JD): {float(entry['t0']):.5f}<br>"
                        f"w: {float(entry['half_width']) / 2.0:.3f}<br>"
                        f"log BF: {bf_text}<br>"
                        f"log BF<sub>thr</sub>: {logbf_thr_text}<br>"
                        f"sig<sub>thr</sub>: {sig_thr_text}<br>"
                        f"morph: {entry['morph'] or 'n/a'}<br>"
                        f"c: {float(entry['confidence']):.2f}<extra></extra>"
                    ),
                ),
                row=row_map['raw'],
                col=1,
            )

    if phase_enabled and 'phase' in row_map and phase_period is not None:
        # Phase-fold uses residuals (mag - baseline) from band_dfs
        phase_zero_jd = float(df["JD"].min()) if not df.empty else None
        for band in (0, 1):
            src_bdf = band_dfs.get(band)
            if src_bdf is None or src_bdf.empty:
                continue
            phase_bdf = _phase_fold_df(src_bdf, phase_period, phase_zero_jd=phase_zero_jd)
            if phase_bdf.empty:
                continue
            for cam in selected:
                cdf = phase_bdf[phase_bdf["camera_label"] == cam]
                if cdf.empty:
                    continue
                color = stable_camera_color(cam)
                err = cdf["error"].to_numpy() if "error" in cdf.columns else np.full(len(cdf), np.nan)
                resid = cdf["resid"].to_numpy() if "resid" in cdf.columns else cdf["mag"].to_numpy()
                y_phase = (_mag_to_flux(resid) - 1.0) if is_flux else resid
                err_phase = _flux_err_from_mag_err(_mag_to_flux(resid), err) if is_flux else err
                jd_full_phase = cdf["JD_plot"].to_numpy() + JD_OFFSET
                hover_phase = np.column_stack([jd_full_phase, err, cdf["mag"].to_numpy(), resid, err_phase])
                phase_hovertemplate = (
                    "<b>Phase-folded residual</b><br>"
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

        fig.add_vline(x=0.0, line_color=colors["guide_line"], line_dash="dot", line_width=1.0, row=row_map['phase'], col=1)
        fig.add_vline(x=1.0, line_color=colors["guide_line"], line_dash="dot", line_width=1.0, row=row_map['phase'], col=1)
        fig.add_vline(x=2.0, line_color=colors["guide_line"], line_dash="dot", line_width=1.0, row=row_map['phase'], col=1)
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
            y_pad = (y_max - y_min) * 0.05
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
        phase_axis_title = rf"$\phi\,(P={phase_period:.5f}\,\mathrm{{d}})$"
        fig.update_xaxes(title_text=phase_axis_title, row=row_map['phase'], col=1, range=[-0.02, 2.02])
        fig.update_yaxes(
            title_text=r"$\Delta F/F$" if is_flux else r"$\Delta m$ [mag]",
            row=row_map['phase'],
            col=1,
            autorange="reversed" if not is_flux else True,
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
        _overlay_external_lcs(fig, row_map['raw'], active_external_lcs, jd_offset, colors, theme, is_flux, ext_source_ranges, ext_trace_start)

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
            "itemclick": False,
            "itemdoubleclick": False,
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
