"""Shared phase-folding helpers for plotting and periodicity diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


BAND_LABELS: dict[int, str] = {0: "g", 1: "V"}

_PERIOD_SOURCE_KEYS: tuple[tuple[str, str], ...] = (
    ("phase_period_days", "phase_period_days"),
    ("period_consensus_days", "period_consensus_days"),
    ("vsx_period", "vsx_period"),
    ("period_vsx_days", "period_vsx_days"),
    ("asassn_var_period", "asassn_var_period"),
    ("period_asassn_var_days", "period_asassn_var_days"),
    ("gaia_eb_period", "gaia_eb_period"),
    ("period_gaia_eb_days", "period_gaia_eb_days"),
    ("ztf_var_period", "ztf_var_period"),
    ("period_ztf_periodic_days", "period_ztf_periodic_days"),
    ("catalog_period", "catalog_period"),
    ("pre_periodicity_selected_period", "pre_periodicity_selected_period"),
    ("lsp_period", "lsp_period"),
    ("pdm_period", "pdm_period"),
    ("ce_period", "ce_period"),
    ("stats_variability_lomb_scargle_best_period_days", "stats_variability_lomb_scargle_best_period_days"),
)

_PERIODOGRAM_PERIOD_KEYS = {
    "lsp_period",
    "pdm_period",
    "ce_period",
    "stats_variability_lomb_scargle_best_period_days",
}


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def resolve_phase_period(
    payload: dict | None,
    *,
    override_period: float | None = None,
    override_source: str = "manual/search",
    include_lsp: bool = True,
    include_periodogram_periods: bool = True,
) -> tuple[float | None, str]:
    """Return the preferred period and source label for phase folding."""
    manual = _finite_float(override_period)
    if manual is not None and manual > 0:
        return float(manual), str(override_source)

    payload = payload or {}
    for key, source in _PERIOD_SOURCE_KEYS:
        if key == "lsp_period" and not include_lsp:
            continue
        if key in _PERIODOGRAM_PERIOD_KEYS and not include_periodogram_periods:
            continue
        period = _finite_float(payload.get(key))
        if period is not None and period > 0:
            if key == "phase_period_days":
                source = str(payload.get("phase_source") or source)
            return float(period), source
    return None, ""


def resolve_phase_epoch(
    df: pd.DataFrame,
    *,
    explicit_epoch_jd: float | None = None,
    jd_col: str = "JD",
) -> float | None:
    """Return the finite phase epoch, defaulting to the minimum finite JD."""
    explicit = _finite_float(explicit_epoch_jd)
    if explicit is not None:
        return float(explicit)
    if df.empty or jd_col not in df.columns:
        return None
    jd = pd.to_numeric(df[jd_col], errors="coerce").to_numpy(dtype=float)
    jd = jd[np.isfinite(jd)]
    if jd.size == 0:
        return None
    return float(np.min(jd))


def camera_labels(df: pd.DataFrame) -> pd.Series:
    """Return stable camera labels using the common light-curve columns."""
    if "camera_label" in df.columns:
        return pd.Series(df["camera_label"].astype(str), index=df.index)
    if "camera_name" in df.columns:
        return pd.Series(df["camera_name"].astype(str), index=df.index)
    if "camera#" in df.columns:
        return pd.Series(df["camera#"].astype(str), index=df.index)
    if "camera" in df.columns:
        return pd.Series(df["camera"].astype(str), index=df.index)
    return pd.Series(["unknown"] * len(df), index=df.index)


def band_labels(df: pd.DataFrame, *, band_col: str = "v_g_band") -> pd.Series:
    """Return g/V labels for numeric ASAS-SN bands, falling back to strings."""
    if band_col not in df.columns:
        return pd.Series(["unknown"] * len(df), index=df.index)
    band = pd.to_numeric(df[band_col], errors="coerce")
    labels = band.map(lambda value: BAND_LABELS.get(int(value), str(value)) if pd.notna(value) else "unknown")
    return pd.Series(labels.astype(str), index=df.index)


def align_v_to_g_magnitude(
    df: pd.DataFrame,
    *,
    min_points_per_band: int = 5,
    band_col: str = "v_g_band",
    mag_col: str = "mag",
) -> tuple[pd.DataFrame, float]:
    """Align V-band magnitudes to g-band using one global median offset."""
    if df.empty or band_col not in df.columns or mag_col not in df.columns:
        return df, np.nan

    band = pd.to_numeric(df[band_col], errors="coerce").to_numpy()
    mag = pd.to_numeric(df[mag_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(mag)
    g_mask = finite & (band == 0)
    v_mask = finite & (band == 1)

    if np.count_nonzero(g_mask) < int(min_points_per_band) or np.count_nonzero(v_mask) < int(min_points_per_band):
        return df, np.nan

    g_med = float(np.median(mag[g_mask]))
    v_med = float(np.median(mag[v_mask]))
    offset_v_minus_g = float(v_med - g_med)
    if not np.isfinite(offset_v_minus_g):
        return df, np.nan

    out = df.copy()
    mag_aligned = mag.copy()
    mag_aligned[v_mask] = mag_aligned[v_mask] - offset_v_minus_g
    out[mag_col] = mag_aligned
    return out, offset_v_minus_g


def phase_template(
    phase: np.ndarray,
    values: np.ndarray,
    *,
    n_bins: int = 48,
    min_bin_points: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a median template over phase bins."""
    template = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    phase = np.asarray(phase, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(phase) & np.isfinite(values)
    if np.count_nonzero(valid) == 0:
        return template, counts

    phase_valid = np.mod(phase[valid], 1.0)
    values_valid = values[valid]
    idx = np.floor(phase_valid * n_bins).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)

    for bin_idx in range(n_bins):
        vals = values_valid[idx == bin_idx]
        if vals.size >= min_bin_points:
            template[bin_idx] = float(np.median(vals))
            counts[bin_idx] = int(vals.size)
    return template, counts


def template_phase_lag(
    template_a: np.ndarray,
    template_b: np.ndarray,
    *,
    signed: bool = False,
) -> float:
    """Return template lag in phase cycles.

    By default this returns the absolute lag for compatibility with existing
    harmonic scoring. With ``signed=True``, positive means template_b occurs
    later in phase than template_a.
    """
    template_a = np.asarray(template_a, dtype=float)
    template_b = np.asarray(template_b, dtype=float)
    if template_a.size == 0 or template_a.size != template_b.size:
        return np.nan

    n = int(template_a.size)
    best_corr = -np.inf
    best_shift = 0
    min_overlap = max(6, n // 4)

    for shift in range(n):
        shifted = np.roll(template_b, shift)
        mask = np.isfinite(template_a) & np.isfinite(shifted)
        if np.count_nonzero(mask) < min_overlap:
            continue
        a = template_a[mask] - np.mean(template_a[mask])
        b = shifted[mask] - np.mean(shifted[mask])
        sa = float(np.std(a))
        sb = float(np.std(b))
        if sa <= 0 or sb <= 0:
            continue
        corr = float(np.mean((a / sa) * (b / sb)))
        if corr > best_corr:
            best_corr = corr
            best_shift = int(shift)

    if not np.isfinite(best_corr):
        return np.nan
    if not signed:
        lag_bins = min(best_shift, n - best_shift)
        return float(lag_bins / n)

    lag_bins = (-best_shift) % n
    if lag_bins > n / 2:
        lag_bins -= n
    return float(lag_bins / n)


def compute_band_phase_lag(
    df: pd.DataFrame,
    *,
    band_a: int = 0,
    band_b: int = 1,
    phase_col: str = "phase",
    value_col: str = "phase_value",
    band_col: str = "v_g_band",
    n_bins: int = 48,
    min_bin_points: int = 3,
) -> dict[str, float]:
    """Compute signed and absolute phase lag between two bands."""
    if df.empty or phase_col not in df.columns or value_col not in df.columns or band_col not in df.columns:
        return {"phase_lag_g_v_cycles": np.nan, "phase_lag_g_v_abs_cycles": np.nan}

    templates: dict[int, np.ndarray] = {}
    for band in (band_a, band_b):
        mask = pd.to_numeric(df[band_col], errors="coerce") == int(band)
        bdf = df[mask]
        if bdf.empty:
            continue
        template, _ = phase_template(
            pd.to_numeric(bdf[phase_col], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(bdf[value_col], errors="coerce").to_numpy(dtype=float),
            n_bins=n_bins,
            min_bin_points=min_bin_points,
        )
        templates[int(band)] = template

    if int(band_a) not in templates or int(band_b) not in templates:
        return {"phase_lag_g_v_cycles": np.nan, "phase_lag_g_v_abs_cycles": np.nan}

    signed_lag = template_phase_lag(templates[int(band_a)], templates[int(band_b)], signed=True)
    abs_lag = abs(float(signed_lag)) if np.isfinite(signed_lag) else np.nan
    return {"phase_lag_g_v_cycles": float(signed_lag), "phase_lag_g_v_abs_cycles": float(abs_lag)}


def phase_fold_dataframe(
    df: pd.DataFrame,
    period_days: float,
    *,
    epoch_jd: float | None = None,
    jd_col: str = "JD",
    mag_col: str = "mag",
    error_col: str = "error",
    band_col: str = "v_g_band",
    resid_col: str = "resid",
    value_mode: str = "mag",
    align_v_to_g: bool = False,
    min_points_per_band: int = 5,
    duplicate_cycles: bool = True,
    n_bins: int = 48,
    min_bin_points: int = 3,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Phase-fold a light curve and return diagnostics.

    ``value_mode`` selects the plotted/diagnostic value in ``phase_value``.
    The original magnitude and residual columns are preserved when present.
    """
    period = _finite_float(period_days)
    if period is None or period <= 0:
        raise ValueError("period_days must be positive and finite")

    out = df.copy()
    v_minus_g_offset = np.nan
    if align_v_to_g:
        out, v_minus_g_offset = align_v_to_g_magnitude(
            out,
            min_points_per_band=min_points_per_band,
            band_col=band_col,
            mag_col=mag_col,
        )

    if value_mode not in {"mag", "resid"}:
        raise ValueError("value_mode must be 'mag' or 'resid'")
    if value_mode == "resid" and resid_col in out.columns:
        value_col = resid_col
    else:
        value_col = mag_col

    if out.empty or jd_col not in out.columns or value_col not in out.columns:
        diagnostics = {
            "period_days": float(period),
            "epoch_jd": None,
            "value_mode": value_mode,
            "value_col": value_col,
            "v_minus_g_median_offset": float(v_minus_g_offset),
            "phase_lag_g_v_cycles": np.nan,
            "phase_lag_g_v_abs_cycles": np.nan,
            "n_phase_points": 0,
        }
        return out, diagnostics

    jd = pd.to_numeric(out[jd_col], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(out[value_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(jd) & np.isfinite(values)
    out = out.loc[finite].copy()
    if out.empty:
        diagnostics = {
            "period_days": float(period),
            "epoch_jd": None,
            "value_mode": value_mode,
            "value_col": value_col,
            "v_minus_g_median_offset": float(v_minus_g_offset),
            "phase_lag_g_v_cycles": np.nan,
            "phase_lag_g_v_abs_cycles": np.nan,
            "n_phase_points": 0,
        }
        return out, diagnostics

    epoch = resolve_phase_epoch(out, explicit_epoch_jd=epoch_jd, jd_col=jd_col)
    if epoch is None:
        diagnostics = {
            "period_days": float(period),
            "epoch_jd": None,
            "value_mode": value_mode,
            "value_col": value_col,
            "v_minus_g_median_offset": float(v_minus_g_offset),
            "phase_lag_g_v_cycles": np.nan,
            "phase_lag_g_v_abs_cycles": np.nan,
            "n_phase_points": 0,
        }
        return out.iloc[0:0].copy(), diagnostics

    jd = pd.to_numeric(out[jd_col], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(out[value_col], errors="coerce").to_numpy(dtype=float)
    out["phase"] = ((jd - float(epoch)) / float(period)) % 1.0
    out["phase_value"] = values
    out["band_label"] = band_labels(out, band_col=band_col)
    out["camera_label"] = camera_labels(out)

    lag_diag = compute_band_phase_lag(
        out,
        phase_col="phase",
        value_col="phase_value",
        band_col=band_col,
        n_bins=n_bins,
        min_bin_points=min_bin_points,
    )

    folded = out
    if duplicate_cycles:
        wrap = out.copy()
        wrap["phase"] = wrap["phase"] + 1.0
        folded = pd.concat([out, wrap], ignore_index=True)

    diagnostics = {
        "period_days": float(period),
        "epoch_jd": float(epoch),
        "value_mode": value_mode,
        "value_col": value_col,
        "v_minus_g_median_offset": float(v_minus_g_offset),
        "n_phase_points": int(len(out)),
        **lag_diag,
    }
    return folded, diagnostics


def phase_time_dataframe(
    df: pd.DataFrame,
    period_days: float,
    *,
    epoch_jd: float | None = None,
    jd_col: str = "JD",
    mag_col: str = "mag",
    error_col: str = "error",
    band_col: str = "v_g_band",
    resid_col: str = "resid",
    value_mode: str = "resid",
    duplicate_cycles: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return phase, cycle, and plot value for a phase-time diagnostic.

    ``phase`` is in cycles and ``cycle`` is the integer cycle number since the
    chosen epoch. ``phase_value`` uses residuals when requested and available,
    otherwise it falls back to the magnitude column.
    """
    period = _finite_float(period_days)
    if period is None or period <= 0:
        raise ValueError("period_days must be positive and finite")
    if value_mode not in {"mag", "resid"}:
        raise ValueError("value_mode must be 'mag' or 'resid'")

    out = df.copy()
    if value_mode == "resid" and resid_col in out.columns:
        value_col = resid_col
    else:
        value_col = mag_col

    diagnostics = {
        "period_days": float(period),
        "epoch_jd": None,
        "value_mode": value_mode,
        "value_col": value_col,
        "n_phase_time_points": 0,
    }
    if out.empty or jd_col not in out.columns or value_col not in out.columns:
        return out.iloc[0:0].copy(), diagnostics

    jd = pd.to_numeric(out[jd_col], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(out[value_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(jd) & np.isfinite(values)
    out = out.loc[finite].copy()
    if out.empty:
        return out, diagnostics

    epoch = resolve_phase_epoch(out, explicit_epoch_jd=epoch_jd, jd_col=jd_col)
    if epoch is None:
        return out.iloc[0:0].copy(), diagnostics

    jd = pd.to_numeric(out[jd_col], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(out[value_col], errors="coerce").to_numpy(dtype=float)
    cycles_float = (jd - float(epoch)) / float(period)
    out["phase"] = np.mod(cycles_float, 1.0)
    out["cycle"] = np.floor(cycles_float).astype(int)
    out["phase_value"] = values
    out["band_label"] = band_labels(out, band_col=band_col)
    out["camera_label"] = camera_labels(out)

    diagnostics = {
        "period_days": float(period),
        "epoch_jd": float(epoch),
        "value_mode": value_mode,
        "value_col": value_col,
        "n_phase_time_points": int(len(out)),
    }

    phase_time = out
    if duplicate_cycles:
        wrap = out.copy()
        wrap["phase"] = wrap["phase"] + 1.0
        phase_time = pd.concat([out, wrap], ignore_index=True)

    return phase_time, diagnostics
