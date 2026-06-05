from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd

from malca.lightcurve_io import normalize_asassn_lightcurve


def _robust_sigma(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) < 3:
        return math.nan
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med))
    return float(1.4826 * mad)


def standardize_asassn_lightcurve(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ASAS-SN light curves through the shared canonical loader."""
    return normalize_asassn_lightcurve(df, apply_quality=True)


def _feature_flux(clean: pd.DataFrame) -> pd.Series:
    """Pick the best flux-like series for nuclear morphology features."""
    for col in ("flux_density_mjy", "rel_flux", "flux"):
        if col not in clean.columns:
            continue
        series = pd.to_numeric(clean[col], errors="coerce")
        if series.notna().sum() >= 5:
            return series
    return pd.Series(np.nan, index=clean.index, dtype=float)


def _flare_groups(mask: np.ndarray, time: np.ndarray, *, max_gap_days: float = 120.0) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    groups: list[list[int]] = [[int(indices[0])]]
    for idx in indices[1:]:
        if time[int(idx)] - time[groups[-1][-1]] <= max_gap_days:
            groups[-1].append(int(idx))
        else:
            groups.append([int(idx)])
    return [np.asarray(group, dtype=int) for group in groups]


def _empty_tde_features() -> dict[str, float | int | str]:
    return {
        "n_flare_events": 0,
        "recurrence_count": 0,
        "preflare_rms": math.nan,
        "tde_single_flare_score": 0.0,
        "tde_quiet_baseline_score": 0.0,
        "tde_no_recurrence_score": 0.0,
        "tde_smooth_decline_score": 0.0,
        "fallback_fit_r2": math.nan,
    }


def compute_tde_flare_features(lc: pd.DataFrame, *, peak_mjd: float | None = None) -> dict[str, float | int | str]:
    if lc.empty:
        return _empty_tde_features()
    clean = standardize_asassn_lightcurve(lc)
    if clean.empty or len(clean) < 5:
        return _empty_tde_features()

    time = pd.to_numeric(clean["mjd"], errors="coerce").to_numpy(dtype=float)
    flux = _feature_flux(clean).to_numpy(dtype=float)
    finite = np.isfinite(time) & np.isfinite(flux)
    time = time[finite]
    flux = flux[finite]
    if len(time) < 5:
        return _empty_tde_features()

    median = np.nanmedian(flux)
    sigma = _robust_sigma(pd.Series(flux))
    sigma = sigma if math.isfinite(sigma) and sigma > 0 else float(np.nanstd(flux))
    threshold = median + 3.0 * max(sigma, 1e-12)
    groups = _flare_groups(flux > threshold, time)
    n_groups = len(groups)
    recurrence_count = max(0, n_groups - 1)

    if peak_mjd is None or not math.isfinite(float(peak_mjd)):
        peak_idx = int(np.nanargmax(flux))
        peak_mjd = float(time[peak_idx])
    pre = flux[time < float(peak_mjd) - 100.0]
    pre_sigma = _robust_sigma(pd.Series(pre)) if len(pre) >= 5 else math.nan
    preflare_rms = float(pre_sigma / max(abs(median), 1e-12)) if math.isfinite(pre_sigma) else math.nan
    quiet_score = float(np.clip(1.0 - (preflare_rms - 0.03) / 0.22, 0.0, 1.0)) if math.isfinite(preflare_rms) else 0.0

    post_mask = time >= float(peak_mjd)
    post_time = time[post_mask] - float(peak_mjd)
    post_flux = flux[post_mask]
    fallback_fit_r2 = math.nan
    smooth_score = 0.0
    if len(post_time) >= 5:
        x = np.log10(np.maximum(post_time, 1.0))
        y = np.log10(np.maximum(post_flux - np.nanmin(post_flux) + 1e-12, 1e-12))
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() >= 5:
            coeff = np.polyfit(x[good], y[good], deg=1)
            pred = np.polyval(coeff, x[good])
            ss_res = float(np.sum((y[good] - pred) ** 2))
            ss_tot = float(np.sum((y[good] - np.nanmean(y[good])) ** 2))
            fallback_fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
            monotonic = abs(pd.Series(post_time[good]).rank().corr(pd.Series(post_flux[good]).rank()))
            smooth_score = float(np.nanmax([np.clip((fallback_fit_r2 - 0.4) / 0.45, 0.0, 1.0), monotonic]))

    return {
        "n_flare_events": int(n_groups),
        "recurrence_count": int(recurrence_count),
        "preflare_rms": preflare_rms,
        "tde_single_flare_score": 1.0 if n_groups == 1 else 0.0,
        "tde_quiet_baseline_score": quiet_score,
        "tde_no_recurrence_score": 1.0 if recurrence_count == 0 else 0.0,
        "tde_smooth_decline_score": smooth_score,
        "fallback_fit_r2": fallback_fit_r2,
    }


def compute_clagn_transition_features(lc: pd.DataFrame) -> dict[str, float | str]:
    clean = standardize_asassn_lightcurve(lc)
    if clean.empty or len(clean) < 8:
        return {
            "clagn_state_change_mag": math.nan,
            "clagn_monotonicity_score": 0.0,
            "clagn_plateau_score": 0.0,
        }
    work = clean.copy()
    work["_feature_flux"] = _feature_flux(work)
    work = work.dropna(subset=["mjd", "_feature_flux"]).sort_values("mjd")
    if len(work) < 8:
        return {
            "clagn_state_change_mag": math.nan,
            "clagn_monotonicity_score": 0.0,
            "clagn_plateau_score": 0.0,
        }
    n = len(work)
    early = work.iloc[: max(3, n // 4)]["_feature_flux"]
    late = work.iloc[-max(3, n // 4) :]["_feature_flux"]
    early_med = float(np.nanmedian(early))
    late_med = float(np.nanmedian(late))
    flux_ratio = max(late_med, 1e-12) / max(early_med, 1e-12)
    state_change_mag = float(-2.5 * np.log10(flux_ratio))
    monotonic = abs(work["mjd"].rank().corr(work["_feature_flux"].rank()))
    late_sigma = _robust_sigma(late)
    full_sigma = _robust_sigma(work["_feature_flux"])
    plateau = 1.0 - (late_sigma / full_sigma) if math.isfinite(late_sigma) and math.isfinite(full_sigma) and full_sigma > 0 else 0.0
    return {
        "clagn_state_change_mag": state_change_mag,
        "clagn_monotonicity_score": float(np.clip(monotonic, 0.0, 1.0)) if math.isfinite(monotonic) else 0.0,
        "clagn_plateau_score": float(np.clip(plateau, 0.0, 1.0)),
    }


def compute_nuclear_lightcurve_features(lc: pd.DataFrame, *, peak_mjd: float | None = None) -> dict[str, float | int | str]:
    clean = standardize_asassn_lightcurve(lc)
    out: dict[str, float | int | str] = {
        "nuc_n_points": int(len(clean)),
        "nuc_time_span_days": float(clean["mjd"].max() - clean["mjd"].min()) if len(clean) else math.nan,
        "nuc_flux_frac_amp_p95_p05": math.nan,
        "nuc_flux_slope_snr": math.nan,
    }
    if len(clean) >= 5:
        flux = _feature_flux(clean).to_numpy(dtype=float)
        amp = (np.nanpercentile(flux, 95) - np.nanpercentile(flux, 5)) / max(abs(np.nanmedian(flux)), 1e-12)
        out["nuc_flux_frac_amp_p95_p05"] = float(amp)
        x = clean["mjd"].to_numpy(dtype=float)
        y = flux
        coeff = np.polyfit(x - np.nanmedian(x), y, deg=1)
        residual = y - np.polyval(coeff, x - np.nanmedian(x))
        slope_err = np.nanstd(residual) / max(np.sqrt(len(y)) * np.nanstd(x), 1e-12)
        out["nuc_flux_slope_snr"] = float(coeff[0] / slope_err) if slope_err > 0 else math.nan
    out.update(compute_tde_flare_features(clean, peak_mjd=peak_mjd))
    out.update(compute_clagn_transition_features(clean))
    return out


def compute_lightcurve_feature_table(
    manifest: pd.DataFrame,
    *,
    id_col: str = "candidate_id",
    path_col: str = "lc_path",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        candidate_id = str(row.get(id_col, "")).strip()
        path = str(row.get(path_col, "")).strip()
        payload: dict[str, object] = {id_col: candidate_id, "lc_feature_status": "missing"}
        if path:
            try:
                lc = pd.read_csv(Path(path).expanduser())
                payload.update(compute_nuclear_lightcurve_features(lc, peak_mjd=row.get("peak_mjd_ref", row.get("discovery_mjd"))))
                payload["lc_feature_status"] = "ok"
            except Exception as exc:
                payload["lc_feature_status"] = "error"
                payload["lc_feature_error"] = str(exc)
        rows.append(payload)
    return pd.DataFrame(rows)
