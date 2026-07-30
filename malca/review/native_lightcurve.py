"""Plotly-free light-curve preparation shared by native review tools.

The browser review module historically owned these helpers, even though the
cleaning, baseline, period-search, and event-annotation policies do not depend
on Plotly.  Keeping the native copies together lets the terminal renderer and
period search use the same policy without importing the browser UI stack.

The implementations intentionally mirror :mod:`malca.review.interactive_plot`.
Browser review retains its own definitions for now so this extraction does not
change the browser execution path.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from malca.config import (
    MJD_TO_JD,
    OFFSET_CAMERA_SIGMA_THRESHOLD,
    REVIEW_CACHE_LIMIT,
    SKYPATROL_JD_OFFSET,
)
from malca.core.baseline import (
    global_median_baseline,
    per_camera_gp_baseline,
    per_camera_gp_baseline_masked,
    per_camera_median_baseline,
)
from malca.core.utils import (
    clean_lc,
    identify_bad_cameras,
    identify_catastrophic_outlier_cameras,
    identify_offset_cameras,
)
from malca.io.lightcurve_io import load_lightcurve_df, to_asassn_algorithm_frame


BASELINE_FUNCTIONS = {
    "global_median": global_median_baseline,
    "per_camera_median": per_camera_median_baseline,
    "per_camera_gp": per_camera_gp_baseline,
    "gp": per_camera_gp_baseline,
    "gp_masked": per_camera_gp_baseline_masked,
    "per_camera_gp_masked": per_camera_gp_baseline_masked,
}

DIP_EVENT_COLOR = "#ff6b6b"
JUMP_EVENT_COLOR = "#0096FF"

_CACHE_LIMIT = REVIEW_CACHE_LIMIT
_CLEAN_CACHE: OrderedDict[
    tuple, tuple[pd.DataFrame, set[int], dict[str, list[str]]]
] = OrderedDict()
_BASELINE_CACHE: OrderedDict[tuple, dict[int, pd.DataFrame]] = OrderedDict()
_EVENT_CACHE: OrderedDict[tuple, list[dict[str, object]]] = OrderedDict()


def _coerce_finite_float(value: object) -> float | None:
    """Return a finite float from a run-param value, or ``None``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _baseline_config_from_run_params(
    run_params: dict | None,
) -> tuple[str, dict[str, float], list[str]]:
    """Resolve review baseline mode and GP kwargs from run parameters."""
    params = dict(run_params or {})
    requested_name = str(params.get("baseline_func") or "per_camera_gp").strip()
    baseline_name = requested_name or "per_camera_gp"
    warnings: list[str] = []

    if baseline_name not in BASELINE_FUNCTIONS:
        warnings.append(
            f"Unknown baseline_func '{baseline_name}' in run_params; "
            "falling back to per_camera_gp."
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


def _freeze_baseline_kwargs(
    baseline_kwargs: dict[str, object] | None,
) -> tuple[tuple[str, object], ...]:
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
    """Resolve bundle assets from either a run directory or its plots anchor."""
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
    """Resolve a candidate light-curve path for native processing."""
    bundle_dir = _bundle_lightcurve_dir(plot_dir)
    candidate_names: list[str] = []

    for key in ("path", "lc_path"):
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
            candidate_names.extend(
                [f"{cid}.dat3", f"{cid}.raw2", f"{cid}.dat2", f"{cid}.dat"]
            )

    asas_sn_id = payload.get("asas_sn_id")
    if asas_sn_id is not None:
        sid = str(asas_sn_id).strip()
        if sid:
            candidate_names.extend(
                [f"{sid}.dat3", f"{sid}.raw2", f"{sid}.dat2", f"{sid}.dat"]
            )

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


def _get_camera_reason_diagnostics(
    df: pd.DataFrame,
    scatter_ratio: float,
) -> dict[str, list[str]]:
    """Return the browser review camera reason tags for native filtering."""
    if df.empty or "camera#" not in df.columns:
        return {}
    diagnostics: dict[str, set[str]] = {}
    try:
        scatter_bad = identify_bad_cameras(
            df,
            scatter_ratio_threshold=scatter_ratio,
        )
    except Exception:
        scatter_bad = set()
    try:
        offset_bad, _ = identify_offset_cameras(
            df,
            offset_sigma_threshold=OFFSET_CAMERA_SIGMA_THRESHOLD,
            remove_full_camera=True,
        )
    except Exception:
        offset_bad = set()
    try:
        catastrophic_bad = identify_catastrophic_outlier_cameras(df)
    except Exception:
        catastrophic_bad = set()

    for camera in scatter_bad:
        diagnostics.setdefault(str(camera), set()).add("scatter")
    for camera in offset_bad:
        diagnostics.setdefault(str(camera), set()).add("offset")
    for camera in catastrophic_bad:
        diagnostics.setdefault(str(camera), set()).add("catastrophic")

    return {camera: sorted(tags) for camera, tags in diagnostics.items()}


def _load_cleaned_df(
    lc_path: Path,
    *,
    filter_bad_cameras: bool,
    scatter_ratio: float,
    clean_max_error_absolute: float,
    clean_max_error_sigma: float,
) -> tuple[pd.DataFrame, set[int], dict[str, list[str]]]:
    """Load and clean an ASAS-SN light curve using browser review policy."""
    key = (
        str(lc_path.resolve()),
        bool(filter_bad_cameras),
        float(scatter_ratio),
        float(clean_max_error_absolute),
        float(clean_max_error_sigma),
    )
    cached = _cache_get(_CLEAN_CACHE, key)
    if cached is not None:
        cached_df, cameras, diagnostics = cached
        return cached_df.copy(), set(cameras), dict(diagnostics)

    df, filtered_cameras = load_lightcurve_df(
        lc_path,
        filter_bad_cameras_enabled=filter_bad_cameras,
        bad_camera_scatter_ratio=scatter_ratio,
        return_filtered_info=True,
    )
    df = to_asassn_algorithm_frame(df)
    diagnostics = _get_camera_reason_diagnostics(df, scatter_ratio)
    df = clean_lc(
        df,
        max_error_absolute=clean_max_error_absolute,
        max_error_sigma=clean_max_error_sigma,
    )
    _cache_put(
        _CLEAN_CACHE,
        key,
        (df.copy(), set(filtered_cameras), diagnostics),
    )
    return df, set(filtered_cameras), diagnostics


def _compute_baseline_bands(
    df: pd.DataFrame,
    baseline_name: str,
    cache_key: tuple,
    *,
    baseline_kwargs: dict[str, object] | None = None,
) -> dict[int, pd.DataFrame]:
    """Compute per-band baselines and residuals with bounded caching."""
    key = (cache_key, baseline_name, _freeze_baseline_kwargs(baseline_kwargs))
    cached = _cache_get(_BASELINE_CACHE, key)
    if cached is not None:
        return {band: frame.copy() for band, frame in cached.items()}

    baseline_func = BASELINE_FUNCTIONS.get(baseline_name, per_camera_gp_baseline)
    call_kwargs = dict(baseline_kwargs or {})
    if baseline_name in {"gp", "per_camera_gp", "gp_masked", "per_camera_gp_masked"}:
        call_kwargs.setdefault("add_sigma_eff_col", True)

    band_dfs: dict[int, pd.DataFrame] = {}
    for band in (0, 1):
        band_df = df[df["v_g_band"] == band].copy()
        if band_df.empty:
            continue
        try:
            output = baseline_func(band_df, **call_kwargs)
            if "baseline" in output.columns:
                band_df["baseline"] = output["baseline"].to_numpy()
                band_df["resid"] = band_df["mag"] - band_df["baseline"]
            else:
                band_df["baseline"] = np.nan
                band_df["resid"] = np.nan
        except Exception:
            band_df["baseline"] = np.nan
            band_df["resid"] = np.nan
        band_dfs[band] = band_df

    _cache_put(
        _BASELINE_CACHE,
        key,
        {band: frame.copy() for band, frame in band_dfs.items()},
    )
    return band_dfs


def _parse_num(payload: dict, key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if np.isfinite(number) else None


def _event_thresholds(run_params: dict | None) -> dict[str, float | None]:
    """Extract event-related thresholds from run parameters."""
    if not run_params:
        return {"dip_logbf": None, "jump_logbf": None, "sig": None}
    output: dict[str, float | None] = {}
    for output_key, run_key in (
        ("dip_logbf", "logbf_threshold_dip"),
        ("jump_logbf", "logbf_threshold_jump"),
        ("sig", "significance_threshold"),
    ):
        try:
            raw_value = run_params.get(run_key)
            output[output_key] = (
                float(raw_value) if raw_value is not None else None
            )
        except Exception:
            output[output_key] = None
    return output


def _event_time_to_lc_scale(
    value: object,
    lc_median: float | None,
    jd_offset: float,
) -> float | None:
    """Convert stored event times to the light-curve time scale."""
    event_time = _coerce_finite_float(value)
    if event_time is None:
        return None
    scale_anchor = lc_median
    if scale_anchor is None or not np.isfinite(scale_anchor):
        scale_anchor = jd_offset
    if scale_anchor > 2_000_000.0:
        if event_time > 2_000_000.0:
            return event_time
        if event_time > 50_000.0:
            return event_time + MJD_TO_JD
        return event_time + SKYPATROL_JD_OFFSET
    if scale_anchor > 50_000.0:
        if event_time > 2_000_000.0:
            return event_time - MJD_TO_JD
        if event_time < 50_000.0:
            return event_time + SKYPATROL_JD_OFFSET - MJD_TO_JD
        return event_time
    if event_time > 2_000_000.0:
        return event_time - SKYPATROL_JD_OFFSET
    if event_time > 50_000.0:
        return event_time + MJD_TO_JD - SKYPATROL_JD_OFFSET
    return event_time


def _event_entries(
    payload: dict,
    jd_offset: float,
    run_params: dict | None,
    *,
    lc_median: float | None = None,
) -> list[dict[str, object]]:
    """Build dip/jump event annotations using browser review semantics."""
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
        return [dict(entry) for entry in cached]

    entries: list[dict[str, object]] = []
    for prefix, color in (("dip", DIP_EVENT_COLOR), ("jump", JUMP_EVENT_COLOR)):
        t0 = _parse_num(payload, f"{prefix}_best_t0")
        if t0 is None:
            continue
        t0_lc = _event_time_to_lc_scale(t0, lc_median, jd_offset)
        if t0_lc is None:
            continue
        width = _parse_num(payload, f"{prefix}_best_width_param")
        bayes_factor = _parse_num(payload, f"{prefix}_bayes_factor")
        morphology = str(payload.get(f"{prefix}_best_morph") or "")
        threshold = (
            thresholds["dip_logbf"]
            if prefix == "dip"
            else thresholds["jump_logbf"]
        )
        confidence_base = threshold if threshold is not None else 3.0
        confidence = (
            0.0
            if bayes_factor is None
            else float(
                np.clip(
                    (bayes_factor - confidence_base) / max(confidence_base, 8.0),
                    0.0,
                    1.0,
                )
            )
        )
        approximate_half_width = 0.0 if width is None else max(width * 2.0, 0.25)
        entries.append(
            {
                "kind": prefix,
                "t0": t0_lc,
                "t0_raw": t0,
                "x0": t0_lc - jd_offset,
                "half_width": approximate_half_width,
                "bf": bayes_factor,
                "morph": morphology,
                "confidence": confidence,
                "base_color": color,
                "logbf_threshold": threshold,
                "sig_threshold": thresholds["sig"],
            }
        )

    _cache_put(_EVENT_CACHE, key, [dict(entry) for entry in entries])
    return entries
