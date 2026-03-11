from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from malca.config.config_filters import (
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    CLEAN_LC_MAX_ERROR_ABSOLUTE,
    CLEAN_LC_MAX_ERROR_SIGMA,
)
from malca.periodogram import ce_find_period, lsp_find_period, pdm_find_period
from malca.review.interactive_plot import (
    _compute_baseline_bands,
    _load_cleaned_df,
    resolve_lightcurve_path,
)


def has_external_period(payload: dict | None) -> bool:
    payload = payload or {}
    for keys in (
        ("phase_period_days",),
        ("period_consensus_days",),
        ("vsx_period", "period_vsx_days"),
        ("asassn_var_period", "period_asassn_var_days"),
        ("gaia_eb_period", "period_gaia_eb_days"),
        ("ztf_var_period", "period_ztf_periodic_days"),
        ("catalog_period",),
    ):
        for key in keys:
            try:
                value = float(payload.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0:
                return True
    return False


def _robust_sigma(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return np.nan
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.nanstd(vals))
    return sigma if np.isfinite(sigma) and sigma > 0 else np.nan


def _phase_template(
    phase: np.ndarray,
    resid: np.ndarray,
    *,
    n_bins: int = 48,
    min_bin_points: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    template = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    phase = np.asarray(phase, dtype=float)
    resid = np.asarray(resid, dtype=float)
    valid = np.isfinite(phase) & np.isfinite(resid)
    if np.count_nonzero(valid) == 0:
        return template, counts

    phase_valid = np.mod(phase[valid], 1.0)
    resid_valid = resid[valid]
    idx = np.floor(phase_valid * n_bins).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)

    for b in range(n_bins):
        vals = resid_valid[idx == b]
        if vals.size >= min_bin_points:
            template[b] = float(np.median(vals))
            counts[b] = int(vals.size)
    return template, counts


def _template_phase_lag(template_a: np.ndarray, template_b: np.ndarray) -> float:
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
            best_shift = shift

    if not np.isfinite(best_corr):
        return np.nan
    lag_bins = min(best_shift, n - best_shift)
    return float(lag_bins / n)


def _score_period_harmonic_candidate(
    band_resid: dict[int, tuple[np.ndarray, np.ndarray]],
    period: float,
    *,
    n_bins: int = 48,
    lag_weight: float = 2.5,
) -> dict[str, float]:
    if not np.isfinite(period) or period <= 0:
        return {"objective": np.inf, "scatter_ratio": np.inf, "lag_phase": np.nan}

    all_jd = [jd for jd, _ in band_resid.values() if jd.size > 0]
    if not all_jd:
        return {"objective": np.inf, "scatter_ratio": np.inf, "lag_phase": np.nan}
    jd0 = float(min(np.min(jd) for jd in all_jd))

    templates: dict[int, np.ndarray] = {}
    scatter_ratios: list[float] = []
    for band, (jd, resid) in band_resid.items():
        phase = np.mod((jd - jd0) / float(period), 1.0)
        template, _ = _phase_template(phase, resid, n_bins=n_bins)
        templates[band] = template

        bin_idx = np.floor(phase * n_bins).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        model = template[bin_idx]
        valid = np.isfinite(model) & np.isfinite(resid)
        if np.count_nonzero(valid) < 20:
            continue

        raw_sigma = _robust_sigma(resid[valid])
        folded_sigma = _robust_sigma(resid[valid] - model[valid])
        if np.isfinite(raw_sigma) and raw_sigma > 0 and np.isfinite(folded_sigma):
            scatter_ratios.append(float(folded_sigma / raw_sigma))

    if not scatter_ratios:
        return {"objective": np.inf, "scatter_ratio": np.inf, "lag_phase": np.nan}

    scatter_ratio = float(np.mean(scatter_ratios))
    lag_phase = np.nan
    if 0 in templates and 1 in templates:
        lag_phase = _template_phase_lag(templates[0], templates[1])
    lag_term = 0.0 if not np.isfinite(lag_phase) else float(lag_phase)
    return {
        "objective": float(scatter_ratio + lag_weight * lag_term),
        "scatter_ratio": scatter_ratio,
        "lag_phase": lag_phase,
    }


def arbitrate_harmonic_period(
    band_dfs: dict[int, pd.DataFrame],
    base_period: float,
    *,
    min_period: float,
    max_period: float,
) -> tuple[float, float, dict[str, float]]:
    if not np.isfinite(base_period) or base_period <= 0:
        return float(base_period), 1.0, {"objective": np.nan, "base_objective": np.nan}

    band_resid: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for band in (0, 1):
        bdf = band_dfs.get(band)
        if bdf is None or bdf.empty or "resid" not in bdf.columns:
            continue
        jd = bdf["JD"].to_numpy(dtype=float)
        resid = bdf["resid"].to_numpy(dtype=float)
        mask = np.isfinite(jd) & np.isfinite(resid)
        if np.count_nonzero(mask) < 30:
            continue
        band_resid[band] = (jd[mask], resid[mask])

    if not band_resid:
        return float(base_period), 1.0, {"objective": np.nan, "base_objective": np.nan}

    candidates: list[tuple[float, float, dict[str, float]]] = []
    for factor in (1.0, 2.0, 0.5):
        p = float(base_period) * float(factor)
        if not np.isfinite(p) or p <= 0 or p < float(min_period) or p > float(max_period):
            continue
        if any(abs(p - prev_p) <= 1e-10 * max(1.0, abs(p), abs(prev_p)) for _, prev_p, _ in candidates):
            continue
        candidates.append((float(factor), p, _score_period_harmonic_candidate(band_resid, p)))

    if not candidates:
        return float(base_period), 1.0, {"objective": np.nan, "base_objective": np.nan}

    finite_candidates = [c for c in candidates if np.isfinite(c[2].get("objective", np.nan))]
    if not finite_candidates:
        return float(base_period), 1.0, {"objective": np.nan, "base_objective": np.nan}

    best_factor, best_period, best_score = min(finite_candidates, key=lambda x: float(x[2]["objective"]))
    base_entry = next((c for c in finite_candidates if abs(c[0] - 1.0) < 1e-12), None)
    base_objective = float(base_entry[2]["objective"]) if base_entry is not None else np.nan

    if base_entry is not None and abs(best_factor - 1.0) > 1e-12:
        improvement = (base_objective - float(best_score["objective"])) / max(abs(base_objective), 1e-9)
        if not np.isfinite(improvement) or improvement < 0.02:
            best_factor, best_period, best_score = base_entry

    diag = {
        "objective": float(best_score.get("objective", np.nan)),
        "scatter_ratio": float(best_score.get("scatter_ratio", np.nan)),
        "lag_phase": float(best_score.get("lag_phase", np.nan)),
        "base_objective": base_objective,
    }
    return float(best_period), float(best_factor), diag


def run_period_search_for_payload(
    payload: dict,
    *,
    plot_dir: Path | None,
    min_period: float,
    max_period: float,
    method: str,
) -> tuple[dict | None, str]:
    lc_path = resolve_lightcurve_path(payload, Path(plot_dir) if plot_dir else None)
    if lc_path is None:
        return None, "No LC file"

    df, _, _ = _load_cleaned_df(
        lc_path,
        filter_bad_cameras=True,
        scatter_ratio=BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
        clean_max_error_absolute=CLEAN_LC_MAX_ERROR_ABSOLUTE,
        clean_max_error_sigma=CLEAN_LC_MAX_ERROR_SIGMA,
    )
    if df is None or df.empty:
        return None, "Empty LC"

    baseline_cache_key = (
        str(lc_path.resolve()),
        (),
        True,
        BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
        CLEAN_LC_MAX_ERROR_ABSOLUTE,
        CLEAN_LC_MAX_ERROR_SIGMA,
    )
    band_dfs = _compute_baseline_bands(df, "per_camera_gp", baseline_cache_key)

    resid_parts = []
    for bdf in band_dfs.values():
        if "resid" not in bdf.columns:
            continue
        mask = np.isfinite(bdf["JD"].to_numpy()) & np.isfinite(bdf["resid"].to_numpy())
        resid_parts.append(bdf[mask][["JD", "resid"]])
    if not resid_parts:
        return None, "No residuals"

    resid_df = pd.concat(resid_parts, ignore_index=True)
    times = resid_df["JD"].to_numpy()
    values = resid_df["resid"].to_numpy()
    if len(times) < 10:
        return None, "Too few points"

    method_name = str(method or "pdm").lower()
    if method_name == "pdm":
        best_period, _, _ = pdm_find_period(times, values, min_period=min_period, max_period=max_period, refine=True)
        label = "PDM"
    elif method_name == "ce":
        best_period, _, _ = ce_find_period(times, values, min_period=min_period, max_period=max_period, refine=True)
        label = "CE"
    else:
        best_period, _, _ = lsp_find_period(times, values, min_period=min_period, max_period=max_period, refine=True)
        label = "LSP"

    raw_best_period = float(best_period)
    best_period, harmonic_factor, harmonic_diag = arbitrate_harmonic_period(
        band_dfs,
        raw_best_period,
        min_period=min_period,
        max_period=max_period,
    )
    summary = f"{label}: P={best_period:.5f} d"
    if abs(harmonic_factor - 1.0) > 1e-12:
        summary += f" (harmonic x{harmonic_factor:g}; base={raw_best_period:.5f} d)"

    return {
        "best_period": float(best_period),
        "method": label,
        "harmonic_factor": float(harmonic_factor),
        "base_period": raw_best_period,
        "harmonic_objective": float(harmonic_diag.get("objective", np.nan)),
        "harmonic_lag_phase": float(harmonic_diag.get("lag_phase", np.nan)),
        "harmonic_scatter_ratio": float(harmonic_diag.get("scatter_ratio", np.nan)),
    }, summary
