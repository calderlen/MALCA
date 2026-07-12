"""Event-relative multi-survey feature extraction for MALCA candidates."""

from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from malca.products.candidates import select_passing_candidates_if_present
from malca.config import GAIA_TCB_EPOCH_JD, MJD_TO_JD, SKYPATROL_JD_OFFSET, TESS_BTJD_OFFSET
from malca.products.feature_layers import with_feature_columns
from malca.io.lightcurve_io import load_lightcurve_df
from malca.io.table_io import read_feature_table, write_feature_table


MS_FEATURE_VERSION = "1"
EVENT_WINDOW_COLUMNS = (
    "failed_any",
    "dip_best_t0",
    "dip_significant",
    "dip_bayes_factor",
    "dip_best_width_param",
    "dip_max_run_duration",
    "jump_best_t0",
    "jump_significant",
    "jump_bayes_factor",
    "jump_best_width_param",
    "jump_max_run_duration",
)

MS_FEATURE_COLUMN_SPECS: tuple[tuple[str, str, str], ...] = (
    ("ms_feature_status", "TEXT", "text"),
    ("ms_feature_version", "TEXT", "text"),
    ("ms_event_type", "TEXT", "text"),
    ("ms_event_t0_jd", "REAL", "float"),
    ("ms_event_start_jd", "REAL", "float"),
    ("ms_event_end_jd", "REAL", "float"),
    ("ms_event_half_width_days", "REAL", "float"),
    ("ms_asassn_n_event_g", "INTEGER", "float"),
    ("ms_asassn_n_baseline_g", "INTEGER", "float"),
    ("ms_asassn_event_g_median", "REAL", "float"),
    ("ms_asassn_baseline_g_median", "REAL", "float"),
    ("ms_asassn_delta_g", "REAL", "float"),
    ("ms_asassn_n_event_v", "INTEGER", "float"),
    ("ms_asassn_n_baseline_v", "INTEGER", "float"),
    ("ms_asassn_event_v_median", "REAL", "float"),
    ("ms_asassn_baseline_v_median", "REAL", "float"),
    ("ms_asassn_delta_v", "REAL", "float"),
    ("ms_asassn_g_minus_v_event", "REAL", "float"),
    ("ms_asassn_g_minus_v_baseline", "REAL", "float"),
    ("ms_asassn_g_minus_v_delta", "REAL", "float"),
    ("ms_ztf_n_event_zg", "INTEGER", "float"),
    ("ms_ztf_n_baseline_zg", "INTEGER", "float"),
    ("ms_ztf_event_zg_median", "REAL", "float"),
    ("ms_ztf_baseline_zg_median", "REAL", "float"),
    ("ms_ztf_delta_zg", "REAL", "float"),
    ("ms_ztf_n_event_zr", "INTEGER", "float"),
    ("ms_ztf_n_baseline_zr", "INTEGER", "float"),
    ("ms_ztf_event_zr_median", "REAL", "float"),
    ("ms_ztf_baseline_zr_median", "REAL", "float"),
    ("ms_ztf_delta_zr", "REAL", "float"),
    ("ms_ztf_gr_event", "REAL", "float"),
    ("ms_ztf_gr_baseline", "REAL", "float"),
    ("ms_ztf_gr_delta", "REAL", "float"),
    ("ms_ztf_gr_event_pairs", "INTEGER", "float"),
    ("ms_ztf_gr_baseline_pairs", "INTEGER", "float"),
    ("ms_neowise_n_near", "INTEGER", "float"),
    ("ms_neowise_n_baseline", "INTEGER", "float"),
    ("ms_neowise_w1_near_median", "REAL", "float"),
    ("ms_neowise_w1_baseline_median", "REAL", "float"),
    ("ms_neowise_w1_delta", "REAL", "float"),
    ("ms_neowise_w2_near_median", "REAL", "float"),
    ("ms_neowise_w2_baseline_median", "REAL", "float"),
    ("ms_neowise_w2_delta", "REAL", "float"),
    ("ms_neowise_w1_w2_near", "REAL", "float"),
    ("ms_neowise_w1_w2_baseline", "REAL", "float"),
    ("ms_neowise_w1_w2_delta", "REAL", "float"),
    ("ms_tess_event_overlap", "INTEGER", "bool"),
    ("ms_tess_n_event", "INTEGER", "float"),
    ("ms_tess_n_baseline", "INTEGER", "float"),
    ("ms_tess_flux_event_median", "REAL", "float"),
    ("ms_tess_flux_baseline_median", "REAL", "float"),
    ("ms_tess_flux_frac_delta", "REAL", "float"),
    ("ms_tess_half_depth_duration_days", "REAL", "float"),
    ("ms_tess_ingress_slope_per_day", "REAL", "float"),
    ("ms_tess_egress_slope_per_day", "REAL", "float"),
    ("ms_tess_asymmetry", "REAL", "float"),
    ("ms_tess_boxiness", "REAL", "float"),
    ("ms_gaia_epoch_n_event", "INTEGER", "float"),
    ("ms_gaia_epoch_n_baseline", "INTEGER", "float"),
    ("ms_gaia_epoch_g_event_median", "REAL", "float"),
    ("ms_gaia_epoch_g_baseline_median", "REAL", "float"),
    ("ms_gaia_epoch_g_delta", "REAL", "float"),
    ("ms_candidate_period_days", "REAL", "float"),
    ("ms_gaia_eb_period_ratio", "REAL", "float"),
    ("ms_gaia_var_flag", "INTEGER", "bool"),
    ("ms_gaia_var_class", "TEXT", "text"),
    ("ms_gaia_var_score", "REAL", "float"),
    ("ms_gaia_eb_period", "REAL", "float"),
    ("ms_gaia_eb_morph", "TEXT", "text"),
    ("ms_gaia_eb_global_ranking", "REAL", "float"),
    ("ms_radial_velocity", "REAL", "float"),
    ("ms_rv_amplitude_robust", "REAL", "float"),
)

MS_FEATURE_COLUMNS: tuple[str, ...] = tuple(col for col, _sql, _kind in MS_FEATURE_COLUMN_SPECS)

_COUNT_COLUMNS = {
    col
    for col, _sql, kind in MS_FEATURE_COLUMN_SPECS
    if col.startswith(("ms_asassn_n_", "ms_ztf_n_", "ms_neowise_n_", "ms_tess_n_", "ms_gaia_epoch_n_"))
    or col.endswith("_pairs")
    or kind == "bool"
}


def _safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _safe_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value) and not (isinstance(value, float) and np.isnan(value))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


def _median(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.nanmedian(arr)) if arr.size else np.nan


def _default_features() -> dict[str, object]:
    out: dict[str, object] = {}
    for col, _sql, kind in MS_FEATURE_COLUMN_SPECS:
        if kind == "text":
            out[col] = ""
        elif col in _COUNT_COLUMNS:
            out[col] = False if kind == "bool" else 0
        else:
            out[col] = np.nan
    out["ms_feature_status"] = "no_event"
    out["ms_feature_version"] = MS_FEATURE_VERSION
    out["ms_event_type"] = "none"
    return out


def _time_to_jd(values: Any, system: str) -> np.ndarray:
    t = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = t[np.isfinite(t)]
    median = float(np.nanmedian(finite)) if finite.size else np.nan
    if system == "mjd":
        if np.isfinite(median) and median > 2_000_000.0:
            return t
        return t + MJD_TO_JD
    if system == "btjd":
        if np.isfinite(median) and median > 2_000_000.0:
            return t
        return t + TESS_BTJD_OFFSET
    if system == "bjd_gaia":
        if np.isfinite(median) and median > 2_000_000.0:
            return t
        return t + GAIA_TCB_EPOCH_JD
    if system == "asassn":
        if np.isfinite(median) and median > 2_000_000.0:
            return t
        if np.isfinite(median) and median > 50_000.0:
            return t + MJD_TO_JD
        return t + SKYPATROL_JD_OFFSET
    return t


def _event_time_to_jd(value: Any) -> float:
    t = _safe_float(value)
    if not np.isfinite(t):
        return np.nan
    if t > 2_000_000.0:
        return float(t)
    if t > 50_000.0:
        return float(t + MJD_TO_JD)
    return float(t + SKYPATROL_JD_OFFSET)


def _derive_event_window(row: pd.Series | dict[str, Any]) -> dict[str, object] | None:
    events: list[dict[str, object]] = []
    for event_type in ("dip", "jump"):
        t0 = _event_time_to_jd(row.get(f"{event_type}_best_t0"))
        if not np.isfinite(t0):
            continue
        significant = _safe_bool(row.get(f"{event_type}_significant"))
        bayes = _safe_float(row.get(f"{event_type}_bayes_factor"))
        width = abs(_safe_float(row.get(f"{event_type}_best_width_param")))
        duration = abs(_safe_float(row.get(f"{event_type}_max_run_duration")))
        half_candidates = [7.0]
        if np.isfinite(width) and width > 0:
            half_candidates.append(3.0 * width)
        if np.isfinite(duration) and duration > 0:
            half_candidates.append(0.5 * duration)
        half_width = min(max(max(half_candidates), 7.0), 120.0)
        events.append(
            {
                "event_type": event_type,
                "t0_jd": t0,
                "half_width_days": float(half_width),
                "significant": significant,
                "bayes_factor": bayes if np.isfinite(bayes) else -np.inf,
            }
        )
    if not events:
        return None
    events.sort(key=lambda item: (bool(item["significant"]), float(item["bayes_factor"])), reverse=True)
    best = events[0]
    t0 = float(best["t0_jd"])
    half_width = float(best["half_width_days"])
    return {
        "event_type": str(best["event_type"]),
        "t0_jd": t0,
        "start_jd": t0 - half_width,
        "end_jd": t0 + half_width,
        "half_width_days": half_width,
    }


def _candidate_lookup_keys(row: pd.Series | dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key in ("candidate_id", "asas_sn_id", "source_id"):
        value = row.get(key)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            keys.append(str(value).strip())
    for key in ("path", "lc_path", "local_lightcurve_path"):
        value = row.get(key)
        if value:
            keys.append(Path(str(value)).stem)
    seen: set[str] = set()
    return [key for key in keys if key and not (key in seen or seen.add(key))]


def _index_external_lc_paths(root: Path | str | None) -> dict[str, dict[str, Path]]:
    prefixes = ("ztf", "tess", "neowise", "gaia_epoch")
    index: dict[str, dict[str, Path]] = {prefix: {} for prefix in prefixes}
    if root is None:
        return index
    root_path = Path(root).expanduser()
    if not root_path.exists():
        return index
    for prefix in prefixes:
        file_prefix = f"{prefix}_lc_"
        for path in root_path.rglob(f"{file_prefix}*.parquet"):
            key = path.stem.replace(file_prefix, "", 1)
            if key:
                index[prefix][key] = path
    return index


def _lookup_external_path(index: dict[str, dict[str, Path]], prefix: str, row: pd.Series | dict[str, Any]) -> Path | None:
    by_key = index.get(prefix, {})
    for key in _candidate_lookup_keys(row):
        path = by_key.get(str(key))
        if path is not None and path.exists():
            return path
    return None


def _first_column(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lookup = {str(col).lower(): str(col) for col in df.columns}
    for name in names:
        found = lookup.get(name.lower())
        if found is not None:
            return found
    return None


def _load_asassn_lc(row: pd.Series | dict[str, Any]) -> pd.DataFrame:
    for key in ("lc_path", "path", "local_lightcurve_path"):
        value = row.get(key)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if not path.exists():
            continue
        raw = load_lightcurve_df(path)
        if raw is None or raw.empty:
            continue
        if "jd" not in raw.columns or "mag" not in raw.columns:
            continue
        out = pd.DataFrame()
        out["jd"] = pd.to_numeric(raw["jd"], errors="coerce")
        out["mag"] = pd.to_numeric(raw["mag"], errors="coerce")
        out["band"] = raw["band"].astype(str).str.strip() if "band" in raw.columns else "all"
        return out[np.isfinite(out["jd"]) & np.isfinite(out["mag"])].reset_index(drop=True)
    return pd.DataFrame(columns=["jd", "mag", "band"])


def _load_external_table(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _normalize_ztf_lc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["jd", "mag", "band"])
    time_col = _first_column(df, ("mjd", "hjd", "jd", "time"))
    mag_col = _first_column(df, ("mag", "m"))
    band_col = _first_column(df, ("band", "filtercode", "filter"))
    if time_col is None or mag_col is None:
        return pd.DataFrame(columns=["jd", "mag", "band"])
    out = pd.DataFrame()
    out["jd"] = _time_to_jd(df[time_col], "mjd")
    out["mag"] = pd.to_numeric(df[mag_col], errors="coerce")
    if band_col:
        band_map = {"1": "zg", "1.0": "zg", "2": "zr", "2.0": "zr", "3": "zi", "3.0": "zi"}
        out["band"] = df[band_col].astype(str).str.strip().str.lower().map(lambda value: band_map.get(value, value))
    else:
        out["band"] = "all"
    return out[np.isfinite(out["jd"]) & np.isfinite(out["mag"])].reset_index(drop=True)


def _normalize_neowise_lc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["jd", "w1", "w2"])
    time_col = _first_column(df, ("mjd", "MJD", "jd", "time"))
    if time_col is None:
        return pd.DataFrame(columns=["jd", "w1", "w2"])
    out = pd.DataFrame()
    out["jd"] = _time_to_jd(df[time_col], "mjd")
    out["w1"] = pd.to_numeric(df.get("w1mpro"), errors="coerce") if "w1mpro" in df.columns else np.nan
    out["w2"] = pd.to_numeric(df.get("w2mpro"), errors="coerce") if "w2mpro" in df.columns else np.nan
    return out[np.isfinite(out["jd"]) & (np.isfinite(out["w1"]) | np.isfinite(out["w2"]))].reset_index(drop=True)


def _normalize_tess_lc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["jd", "flux"])
    time_col = _first_column(df, ("time", "btjd", "bjd", "jd"))
    flux_col = _first_column(df, ("flux", "f"))
    if time_col is None or flux_col is None:
        return pd.DataFrame(columns=["jd", "flux"])
    out = pd.DataFrame()
    out["jd"] = _time_to_jd(df[time_col], "btjd")
    out["flux"] = pd.to_numeric(df[flux_col], errors="coerce")
    if "quality" in df.columns:
        quality = pd.to_numeric(df["quality"], errors="coerce").fillna(0)
        out = out.loc[quality.to_numpy() == 0].copy()
    return out[np.isfinite(out["jd"]) & np.isfinite(out["flux"])].reset_index(drop=True)


def _normalize_gaia_epoch_lc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["jd", "mag"])
    time_col = _first_column(df, ("time", "g_transit_time"))
    mag_col = _first_column(df, ("mag", "g_transit_mag"))
    if time_col is None or mag_col is None:
        return pd.DataFrame(columns=["jd", "mag"])
    out = pd.DataFrame()
    out["jd"] = _time_to_jd(df[time_col], "bjd_gaia")
    out["mag"] = pd.to_numeric(df[mag_col], errors="coerce")
    return out[np.isfinite(out["jd"]) & np.isfinite(out["mag"])].reset_index(drop=True)


def _add_band_median_features(
    features: dict[str, object],
    df: pd.DataFrame,
    *,
    prefix: str,
    band_col: str,
    value_col: str,
    bands: tuple[str, ...],
    start_jd: float,
    end_jd: float,
) -> None:
    event_mask = (df["jd"] >= start_jd) & (df["jd"] <= end_jd)
    for band in bands:
        band_key = band.lower()
        band_mask = df[band_col].astype(str).str.lower() == band.lower()
        event_values = df.loc[event_mask & band_mask, value_col]
        base_values = df.loc[(~event_mask) & band_mask, value_col]
        features[f"{prefix}_n_event_{band_key}"] = int(pd.to_numeric(event_values, errors="coerce").notna().sum())
        features[f"{prefix}_n_baseline_{band_key}"] = int(pd.to_numeric(base_values, errors="coerce").notna().sum())
        event_median = _median(event_values)
        base_median = _median(base_values)
        features[f"{prefix}_event_{band_key}_median"] = event_median
        features[f"{prefix}_baseline_{band_key}_median"] = base_median
        features[f"{prefix}_delta_{band_key}"] = event_median - base_median if np.isfinite(event_median) and np.isfinite(base_median) else np.nan


def _add_median_color(
    features: dict[str, object],
    *,
    prefix: str,
    blue_key: str,
    red_key: str,
    color_name: str,
) -> None:
    event_blue = _safe_float(features.get(f"{prefix}_event_{blue_key}_median"))
    event_red = _safe_float(features.get(f"{prefix}_event_{red_key}_median"))
    base_blue = _safe_float(features.get(f"{prefix}_baseline_{blue_key}_median"))
    base_red = _safe_float(features.get(f"{prefix}_baseline_{red_key}_median"))
    event_color = event_blue - event_red if np.isfinite(event_blue) and np.isfinite(event_red) else np.nan
    base_color = base_blue - base_red if np.isfinite(base_blue) and np.isfinite(base_red) else np.nan
    features[f"{prefix}_{color_name}_event"] = event_color
    features[f"{prefix}_{color_name}_baseline"] = base_color
    features[f"{prefix}_{color_name}_delta"] = event_color - base_color if np.isfinite(event_color) and np.isfinite(base_color) else np.nan


def _nearest_gr_color(df: pd.DataFrame, mask: pd.Series, max_days: float = 1.0) -> tuple[float, int]:
    subset = df.loc[mask].copy()
    g = subset[subset["band"].astype(str).str.lower() == "zg"].sort_values("jd")
    r = subset[subset["band"].astype(str).str.lower() == "zr"].sort_values("jd")
    if g.empty or r.empty:
        return np.nan, 0
    r_t = r["jd"].to_numpy(dtype=float)
    r_mag = r["mag"].to_numpy(dtype=float)
    colors: list[float] = []
    for t, mag in zip(g["jd"].to_numpy(dtype=float), g["mag"].to_numpy(dtype=float)):
        pos = int(np.searchsorted(r_t, t))
        candidates: list[int] = []
        if pos < len(r_t):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        if not candidates:
            continue
        best_idx = min(candidates, key=lambda idx: abs(r_t[idx] - t))
        if abs(r_t[best_idx] - t) <= max_days:
            colors.append(float(mag - r_mag[best_idx]))
    if not colors:
        return np.nan, 0
    return float(np.nanmedian(colors)), len(colors)


def _add_ztf_color(features: dict[str, object], ztf: pd.DataFrame, start_jd: float, end_jd: float) -> None:
    if ztf.empty:
        return
    event_mask = (ztf["jd"] >= start_jd) & (ztf["jd"] <= end_jd)
    event_color, event_pairs = _nearest_gr_color(ztf, event_mask)
    base_color, base_pairs = _nearest_gr_color(ztf, ~event_mask)
    features["ms_ztf_gr_event"] = event_color
    features["ms_ztf_gr_baseline"] = base_color
    features["ms_ztf_gr_delta"] = event_color - base_color if np.isfinite(event_color) and np.isfinite(base_color) else np.nan
    features["ms_ztf_gr_event_pairs"] = int(event_pairs)
    features["ms_ztf_gr_baseline_pairs"] = int(base_pairs)


def _add_neowise_features(features: dict[str, object], neowise: pd.DataFrame, t0_jd: float) -> None:
    if neowise.empty:
        return
    near_mask = (neowise["jd"] >= t0_jd - 180.0) & (neowise["jd"] <= t0_jd + 180.0)
    base_mask = ~near_mask
    features["ms_neowise_n_near"] = int(near_mask.sum())
    features["ms_neowise_n_baseline"] = int(base_mask.sum())
    for band in ("w1", "w2"):
        near = _median(neowise.loc[near_mask, band])
        base = _median(neowise.loc[base_mask, band])
        features[f"ms_neowise_{band}_near_median"] = near
        features[f"ms_neowise_{band}_baseline_median"] = base
        features[f"ms_neowise_{band}_delta"] = near - base if np.isfinite(near) and np.isfinite(base) else np.nan
    near_color = _median(neowise.loc[near_mask, "w1"] - neowise.loc[near_mask, "w2"])
    base_color = _median(neowise.loc[base_mask, "w1"] - neowise.loc[base_mask, "w2"])
    features["ms_neowise_w1_w2_near"] = near_color
    features["ms_neowise_w1_w2_baseline"] = base_color
    features["ms_neowise_w1_w2_delta"] = near_color - base_color if np.isfinite(near_color) and np.isfinite(base_color) else np.nan


def _linear_slope(x: pd.Series, y: pd.Series) -> float:
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return np.nan
    x_fit = x_arr[mask]
    y_fit = y_arr[mask]
    if float(np.nanmax(x_fit) - np.nanmin(x_fit)) <= 0:
        return np.nan
    slope, _intercept = np.polyfit(x_fit, y_fit, 1)
    return float(slope)


def _add_tess_features(features: dict[str, object], tess: pd.DataFrame, event: dict[str, object]) -> None:
    if tess.empty:
        return
    start_jd = float(event["start_jd"])
    end_jd = float(event["end_jd"])
    t0_jd = float(event["t0_jd"])
    half_width = float(event["half_width_days"])
    event_mask = (tess["jd"] >= start_jd) & (tess["jd"] <= end_jd)
    base_mask = ~event_mask
    event_flux = tess.loc[event_mask, "flux"]
    base_flux = tess.loc[base_mask, "flux"]
    features["ms_tess_event_overlap"] = bool(int(event_mask.sum()) > 0)
    features["ms_tess_n_event"] = int(pd.to_numeric(event_flux, errors="coerce").notna().sum())
    features["ms_tess_n_baseline"] = int(pd.to_numeric(base_flux, errors="coerce").notna().sum())
    event_med = _median(event_flux)
    base_med = _median(base_flux)
    features["ms_tess_flux_event_median"] = event_med
    features["ms_tess_flux_baseline_median"] = base_med
    if np.isfinite(event_med) and np.isfinite(base_med) and base_med != 0:
        features["ms_tess_flux_frac_delta"] = event_med / base_med - 1.0

    event_df = tess.loc[event_mask].copy()
    if event_df.empty or not np.isfinite(event_med) or not np.isfinite(base_med):
        return
    threshold = base_med + 0.5 * (event_med - base_med)
    if event_med < base_med:
        half_depth = event_df[event_df["flux"] <= threshold]
    else:
        half_depth = event_df[event_df["flux"] >= threshold]
    if len(half_depth) >= 2:
        duration = float(half_depth["jd"].max() - half_depth["jd"].min())
        features["ms_tess_half_depth_duration_days"] = duration
        denom = max(2.0 * half_width, 1e-12)
        features["ms_tess_boxiness"] = float(np.clip(duration / denom, 0.0, 1.0))
    before = event_df[event_df["jd"] <= t0_jd]
    after = event_df[event_df["jd"] >= t0_jd]
    ingress = _linear_slope(before["jd"], before["flux"])
    egress = _linear_slope(after["jd"], after["flux"])
    features["ms_tess_ingress_slope_per_day"] = ingress
    features["ms_tess_egress_slope_per_day"] = egress
    if np.isfinite(ingress) and np.isfinite(egress):
        denom = abs(ingress) + abs(egress)
        features["ms_tess_asymmetry"] = abs(abs(ingress) - abs(egress)) / denom if denom > 0 else np.nan


def _add_gaia_epoch_features(features: dict[str, object], gaia: pd.DataFrame, start_jd: float, end_jd: float) -> None:
    if gaia.empty:
        return
    event_mask = (gaia["jd"] >= start_jd) & (gaia["jd"] <= end_jd)
    base_mask = ~event_mask
    event_mag = gaia.loc[event_mask, "mag"]
    base_mag = gaia.loc[base_mask, "mag"]
    features["ms_gaia_epoch_n_event"] = int(pd.to_numeric(event_mag, errors="coerce").notna().sum())
    features["ms_gaia_epoch_n_baseline"] = int(pd.to_numeric(base_mag, errors="coerce").notna().sum())
    event_median = _median(event_mag)
    base_median = _median(base_mag)
    features["ms_gaia_epoch_g_event_median"] = event_median
    features["ms_gaia_epoch_g_baseline_median"] = base_median
    features["ms_gaia_epoch_g_delta"] = event_median - base_median if np.isfinite(event_median) and np.isfinite(base_median) else np.nan


def _candidate_period_days(row: pd.Series | dict[str, Any]) -> float:
    for col in (
        "periodicity_period",
        "pdm_corrected_period",
        "ce_corrected_period",
        "pdm_period",
        "ce_period",
        "period_consensus_days",
        "pre_periodicity_selected_period",
        "asassn_var_period",
        "ztf_var_period",
        "vsx_period",
    ):
        value = _safe_float(row.get(col))
        if np.isfinite(value) and value > 0:
            return float(value)
    return np.nan


def _add_gaia_payload_features(features: dict[str, object], row: pd.Series | dict[str, Any]) -> None:
    features["ms_gaia_var_flag"] = _safe_bool(row.get("gaia_var_flag"))
    features["ms_gaia_var_class"] = str(row.get("gaia_var_class") or "")
    features["ms_gaia_var_score"] = _safe_float(row.get("gaia_var_score"))
    features["ms_gaia_eb_period"] = _safe_float(row.get("gaia_eb_period"))
    features["ms_gaia_eb_morph"] = str(row.get("gaia_eb_morph") or "")
    features["ms_gaia_eb_global_ranking"] = _safe_float(row.get("gaia_eb_global_ranking"))
    features["ms_radial_velocity"] = _safe_float(row.get("radial_velocity"))
    features["ms_rv_amplitude_robust"] = _safe_float(row.get("rv_amplitude_robust"))
    candidate_period = _candidate_period_days(row)
    features["ms_candidate_period_days"] = candidate_period
    gaia_period = _safe_float(row.get("gaia_eb_period"))
    if np.isfinite(candidate_period) and candidate_period > 0 and np.isfinite(gaia_period) and gaia_period > 0:
        features["ms_gaia_eb_period_ratio"] = candidate_period / gaia_period


def compute_candidate_multi_survey_features(
    row: pd.Series | dict[str, Any],
    *,
    external_lc_dir: Path | str | None = None,
    external_index: dict[str, dict[str, Path]] | None = None,
) -> dict[str, object]:
    features = _default_features()
    event = _derive_event_window(row)
    _add_gaia_payload_features(features, row)
    if event is None:
        return features

    features["ms_feature_status"] = "ok"
    features["ms_event_type"] = str(event["event_type"])
    features["ms_event_t0_jd"] = float(event["t0_jd"])
    features["ms_event_start_jd"] = float(event["start_jd"])
    features["ms_event_end_jd"] = float(event["end_jd"])
    features["ms_event_half_width_days"] = float(event["half_width_days"])
    start_jd = float(event["start_jd"])
    end_jd = float(event["end_jd"])
    t0_jd = float(event["t0_jd"])

    index = external_index if external_index is not None else _index_external_lc_paths(external_lc_dir)

    asassn = _load_asassn_lc(row)
    if not asassn.empty:
        _add_band_median_features(
            features,
            asassn,
            prefix="ms_asassn",
            band_col="band",
            value_col="mag",
            bands=("g", "V"),
            start_jd=start_jd,
            end_jd=end_jd,
        )
        _add_median_color(features, prefix="ms_asassn", blue_key="g", red_key="v", color_name="g_minus_v")

    ztf = _normalize_ztf_lc(_load_external_table(_lookup_external_path(index, "ztf", row)))
    if not ztf.empty:
        _add_band_median_features(
            features,
            ztf,
            prefix="ms_ztf",
            band_col="band",
            value_col="mag",
            bands=("zg", "zr"),
            start_jd=start_jd,
            end_jd=end_jd,
        )
        _add_ztf_color(features, ztf, start_jd, end_jd)

    neowise = _normalize_neowise_lc(_load_external_table(_lookup_external_path(index, "neowise", row)))
    _add_neowise_features(features, neowise, t0_jd)

    tess = _normalize_tess_lc(_load_external_table(_lookup_external_path(index, "tess", row)))
    _add_tess_features(features, tess, event)

    gaia = _normalize_gaia_epoch_lc(_load_external_table(_lookup_external_path(index, "gaia_epoch", row)))
    _add_gaia_epoch_features(features, gaia, start_jd, end_jd)

    return features


def compute_multi_survey_features(
    df: pd.DataFrame,
    *,
    external_lc_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Append ``ms_*`` multi-survey feature columns to a candidate table."""
    out = df.copy()
    external_index = _index_external_lc_paths(external_lc_dir)
    rows: list[dict[str, object]] = []
    for _, row in out.iterrows():
        rows.append(
            compute_candidate_multi_survey_features(
                row,
                external_lc_dir=external_lc_dir,
                external_index=external_index,
            )
        )
    features = pd.DataFrame(rows, index=out.index)
    for col in MS_FEATURE_COLUMNS:
        out[col] = features[col] if col in features.columns else np.nan
    return out


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_multi_survey_features.parquet")


def _ensure_candidate_id(df: pd.DataFrame) -> pd.DataFrame:
    if "candidate_id" in df.columns:
        return df
    if "asas_sn_id" not in df.columns:
        return df
    df = df.copy()
    df["candidate_id"] = df["asas_sn_id"].astype(str)
    return df


def _merge_frame(out: pd.DataFrame) -> pd.DataFrame:
    id_cols = [col for col in ("candidate_id", "asas_sn_id") if col in out.columns]
    value_cols = [col for col in MS_FEATURE_COLUMNS if col in out.columns]
    return out[id_cols + value_cols].copy()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malca multi-survey-features",
        description="Compute event-relative multi-survey features for MALCA candidates.",
    )
    parser.add_argument("input", type=Path, help="Input candidate Parquet file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path (default: <input>_multi_survey_features.parquet)",
    )
    parser.add_argument(
        "--external-lc-dir",
        type=Path,
        default=None,
        help="Directory containing ztf_lc_*, tess_lc_*, neowise_lc_*, and gaia_epoch_lc_* parquet files.",
    )
    parser.add_argument(
        "--review-db",
        type=Path,
        default=None,
        help="Optional review SQLite DB to merge ms_* fields into",
    )
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="Compute features for all input rows instead of only failed_any=False passers.",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    input_path = args.input.expanduser()
    output_path = (args.output or _default_output_path(input_path)).expanduser()
    external_lc_dir = (args.external_lc_dir or input_path.parent).expanduser()

    df = with_feature_columns(_ensure_candidate_id(read_feature_table(input_path)), EVENT_WINDOW_COLUMNS)
    if not getattr(args, "all_candidates", False):
        df = select_passing_candidates_if_present(df, printer=print)
    print(f"Loaded {len(df)} candidates from {input_path}")
    print(f"Reading external LC files from {external_lc_dir}")
    out = compute_multi_survey_features(df, external_lc_dir=external_lc_dir)

    write_feature_table(out, output_path)
    print(f"Saved multi-survey feature table to {output_path}")

    if args.review_db:
        from malca.review.store import db_connect, merge_candidate_results

        review_db = args.review_db.expanduser()
        with closing(db_connect(review_db)) as conn:
            updated = merge_candidate_results(conn, _merge_frame(out))
        print(f"Merged ms_* fields into {review_db} ({updated} candidates updated)")

    return output_path


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
