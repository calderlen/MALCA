from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from malca.config import (
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    CLEAN_LC_MAX_ERROR_ABSOLUTE,
    CLEAN_LC_MAX_ERROR_SIGMA,
)
from malca.config import LS_ALIAS_PERIODS, LS_ALIAS_TOLERANCE
from malca.core.phase import phase_template, resolve_phase_period, template_phase_lag
from malca.core.periodogram import ce_find_period, lsp_find_period, pdm_find_period
from malca.review.interactive_plot import (
    _compute_baseline_bands,
    _load_cleaned_df,
    resolve_lightcurve_path,
)


REVIEW_PERIOD_HARMONIC_FACTORS: tuple[float, ...] = (
    1.0,
    2.0,
    0.5,
    3.0,
    1.0 / 3.0,
    4.0,
    0.25,
)
REVIEW_HARMONIC_CHECK_DIVISORS: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
REVIEW_HARMONIC_CHECK_SCORE_TOLERANCE = 0.03
AUTO_PERIOD_METHODS: tuple[str, ...] = ("pdm", "ce")
_METHOD_LABELS: dict[str, str] = {
    "pdm": "PDM",
    "ce": "CE",
    "lsp": "LSP",
}
_METHOD_PRIORITY: dict[str, int] = {
    "pdm": 0,
    "ce": 1,
    "lsp": 2,
}


def has_external_period(payload: dict | None) -> bool:
    period, _ = resolve_phase_period(payload, include_lsp=False, include_periodogram_periods=False)
    return bool(period is not None and np.isfinite(period) and period > 0)


def resolve_stored_review_period(payload: dict | None) -> tuple[float | None, str]:
    """Return the stored/precomputed period used as the review harmonic-check seed."""
    period, source = resolve_phase_period(payload, include_periodogram_periods=False)
    if period is None or not np.isfinite(period) or period <= 0:
        return None, ""
    return float(period), str(source or "stored period")


def _harmonic_check_candidate_periods(
    base_period: float,
    *,
    min_period: float,
    max_period: float,
    divisors: tuple[float, ...] = REVIEW_HARMONIC_CHECK_DIVISORS,
) -> list[dict[str, float]]:
    """Return the live review harmonic candidates: P0, P0/2, P0/3, P0/4."""
    try:
        base = float(base_period)
        min_p = float(min_period)
        max_p = float(max_period)
    except (TypeError, ValueError):
        return []
    if not (np.isfinite(base) and base > 0 and np.isfinite(min_p) and np.isfinite(max_p)):
        return []
    if max_p < min_p:
        min_p, max_p = max_p, min_p

    candidates: list[dict[str, float]] = []
    for divisor in divisors:
        divisor = float(divisor)
        if not np.isfinite(divisor) or divisor <= 0:
            continue
        period = base / divisor
        if not np.isfinite(period) or period <= 0 or period < min_p or period > max_p:
            continue
        if any(abs(period - item["period"]) <= 1e-10 * max(1.0, abs(period), abs(item["period"])) for item in candidates):
            continue
        candidates.append({"factor": 1.0 / divisor, "divisor": divisor, "period": float(period)})
    return candidates


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


def _score_period_harmonic_candidate(
    band_resid: dict[int, tuple[np.ndarray, np.ndarray]],
    period: float,
    *,
    n_bins: int = 48,
    lag_weight: float = 2.5,
    alias_penalty: float = 0.2,
) -> dict[str, object]:
    if not np.isfinite(period) or period <= 0:
        return {
            "objective": np.inf,
            "raw_objective": np.inf,
            "scatter_ratio": np.inf,
            "lag_phase": np.nan,
            "alias_flag": False,
            "alias_matches": [],
        }

    all_jd = [jd for jd, _ in band_resid.values() if jd.size > 0]
    if not all_jd:
        return {
            "objective": np.inf,
            "raw_objective": np.inf,
            "scatter_ratio": np.inf,
            "lag_phase": np.nan,
            "alias_flag": False,
            "alias_matches": [],
        }
    jd0 = float(min(np.min(jd) for jd in all_jd))

    templates: dict[int, np.ndarray] = {}
    scatter_ratios: list[float] = []
    for band, (jd, resid) in band_resid.items():
        phase = np.mod((jd - jd0) / float(period), 1.0)
        template, _ = phase_template(phase, resid, n_bins=n_bins)
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
        return {
            "objective": np.inf,
            "raw_objective": np.inf,
            "scatter_ratio": np.inf,
            "lag_phase": np.nan,
            "alias_flag": False,
            "alias_matches": [],
        }

    scatter_ratio = float(np.mean(scatter_ratios))
    lag_phase = np.nan
    if 0 in templates and 1 in templates:
        lag_phase = template_phase_lag(templates[0], templates[1])
    lag_term = 0.0 if not np.isfinite(lag_phase) else float(lag_phase)
    raw_objective = float(scatter_ratio + lag_weight * lag_term)
    alias_matches = [
        float(alias_period)
        for alias_period in LS_ALIAS_PERIODS
        if np.isfinite(alias_period) and abs(float(period) - float(alias_period)) <= float(LS_ALIAS_TOLERANCE)
    ]
    alias_flag = bool(alias_matches)
    return {
        "objective": float(raw_objective + (alias_penalty if alias_flag else 0.0)),
        "raw_objective": raw_objective,
        "scatter_ratio": scatter_ratio,
        "lag_phase": lag_phase,
        "alias_flag": alias_flag,
        "alias_matches": alias_matches,
    }


def _band_resid_from_band_dfs(band_dfs: dict[int, pd.DataFrame]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
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
    return band_resid


def _select_harmonic_check_candidate(
    candidates: list[dict[str, object]],
    *,
    score_tolerance: float = REVIEW_HARMONIC_CHECK_SCORE_TOLERANCE,
) -> dict[str, object] | None:
    finite = [
        candidate
        for candidate in candidates
        if np.isfinite(float(candidate.get("selection_objective", np.nan)))
        and np.isfinite(float(candidate.get("period", np.nan)))
    ]
    if not finite:
        return None
    best_score = min(float(candidate["selection_objective"]) for candidate in finite)
    cutoff = best_score + max(abs(best_score) * float(score_tolerance), 1e-9)
    comparable = [
        candidate
        for candidate in finite
        if float(candidate["selection_objective"]) <= cutoff
    ]
    return min(
        comparable,
        key=lambda candidate: (
            float(candidate.get("period", np.inf)),
            float(candidate.get("selection_objective", np.inf)),
        ),
    )


def check_stored_period_harmonics(
    band_dfs: dict[int, pd.DataFrame],
    base_period: float,
    *,
    min_period: float,
    max_period: float,
    score_tolerance: float = REVIEW_HARMONIC_CHECK_SCORE_TOLERANCE,
) -> tuple[float, float, dict[str, object]]:
    """Score P0, P0/2, P0/3, and P0/4, preferring shorter comparable harmonics."""
    try:
        base = float(base_period)
    except (TypeError, ValueError):
        return np.nan, 1.0, {"objective": np.nan, "base_objective": np.nan, "candidates": []}
    if not np.isfinite(base) or base <= 0:
        return np.nan, 1.0, {"objective": np.nan, "base_objective": np.nan, "candidates": []}

    candidate_periods = _harmonic_check_candidate_periods(base, min_period=min_period, max_period=max_period)
    if not candidate_periods:
        return np.nan, 1.0, {"objective": np.nan, "base_objective": np.nan, "candidates": []}

    band_resid = _band_resid_from_band_dfs(band_dfs)
    if not band_resid:
        return np.nan, 1.0, {"objective": np.nan, "base_objective": np.nan, "candidates": candidate_periods}

    scored_candidates: list[dict[str, object]] = []
    for candidate in candidate_periods:
        score = dict(_score_period_harmonic_candidate(band_resid, candidate["period"], alias_penalty=0.0))
        selection_objective = float(score.get("raw_objective", score.get("objective", np.inf)))
        scored_candidates.append(
            {
                **candidate,
                "objective": float(score.get("objective", np.nan)),
                "selection_objective": selection_objective,
                "raw_objective": float(score.get("raw_objective", np.nan)),
                "scatter_ratio": float(score.get("scatter_ratio", np.nan)),
                "lag_phase": float(score.get("lag_phase", np.nan)),
                "alias_flag": bool(score.get("alias_flag", False)),
                "alias_matches": [float(v) for v in score.get("alias_matches", [])],
            }
        )

    selected = _select_harmonic_check_candidate(scored_candidates, score_tolerance=score_tolerance)
    base_entry = next((candidate for candidate in scored_candidates if abs(float(candidate.get("divisor", 0.0)) - 1.0) < 1e-12), None)
    base_objective = (
        float(base_entry.get("selection_objective", base_entry.get("objective", np.nan)))
        if base_entry is not None else np.nan
    )
    if selected is None:
        return np.nan, 1.0, {"objective": np.nan, "base_objective": base_objective, "candidates": scored_candidates}

    diag = {
        "objective": float(selected.get("objective", np.nan)),
        "selection_objective": float(selected.get("selection_objective", np.nan)),
        "raw_objective": float(selected.get("raw_objective", np.nan)),
        "scatter_ratio": float(selected.get("scatter_ratio", np.nan)),
        "lag_phase": float(selected.get("lag_phase", np.nan)),
        "base_objective": base_objective,
        "harmonic_penalty": 0.0,
        "harmonic_divisor": float(selected.get("divisor", 1.0)),
        "alias_flag": bool(selected.get("alias_flag", False)),
        "alias_matches": [float(v) for v in selected.get("alias_matches", [])],
        "candidates": scored_candidates,
    }
    return float(selected.get("period", base)), float(selected.get("factor", 1.0)), diag


def arbitrate_harmonic_period(
    band_dfs: dict[int, pd.DataFrame],
    base_period: float,
    *,
    min_period: float,
    max_period: float,
    harmonic_factors: tuple[float, ...] = REVIEW_PERIOD_HARMONIC_FACTORS,
    harmonic_penalty_scale: float = 0.02,
) -> tuple[float, float, dict[str, object]]:
    if not np.isfinite(base_period) or base_period <= 0:
        return float(base_period), 1.0, {"objective": np.nan, "base_objective": np.nan}

    band_resid = _band_resid_from_band_dfs(band_dfs)
    if not band_resid:
        return float(base_period), 1.0, {"objective": np.nan, "base_objective": np.nan}

    candidates: list[tuple[float, float, dict[str, float]]] = []
    for factor in harmonic_factors:
        p = float(base_period) * float(factor)
        if not np.isfinite(p) or p <= 0 or p < float(min_period) or p > float(max_period):
            continue
        if any(abs(p - prev_p) <= 1e-10 * max(1.0, abs(p), abs(prev_p)) for _, prev_p, _ in candidates):
            continue
        score = dict(_score_period_harmonic_candidate(band_resid, p))
        harmonic_penalty = float(harmonic_penalty_scale * abs(np.log2(float(factor)))) if factor > 0 else np.inf
        score["harmonic_penalty"] = harmonic_penalty
        base_objective = float(score.get("objective", np.inf))
        score["selection_objective"] = float(base_objective + harmonic_penalty) if np.isfinite(base_objective) else np.inf
        candidates.append((float(factor), p, score))

    if not candidates:
        return float(base_period), 1.0, {"objective": np.nan, "base_objective": np.nan}

    finite_candidates = [c for c in candidates if np.isfinite(c[2].get("objective", np.nan))]
    if not finite_candidates:
        return float(base_period), 1.0, {"objective": np.nan, "base_objective": np.nan}

    best_factor, best_period, best_score = min(
        finite_candidates,
        key=lambda x: float(x[2].get("selection_objective", x[2].get("objective", np.inf))),
    )
    base_entry = next((c for c in finite_candidates if abs(c[0] - 1.0) < 1e-12), None)
    base_objective = (
        float(base_entry[2].get("selection_objective", base_entry[2].get("objective", np.nan)))
        if base_entry is not None else np.nan
    )

    if base_entry is not None and abs(best_factor - 1.0) > 1e-12:
        best_selection_objective = float(best_score.get("selection_objective", best_score.get("objective", np.nan)))
        improvement = (base_objective - best_selection_objective) / max(abs(base_objective), 1e-9)
        if not np.isfinite(improvement) or improvement < 0.02:
            best_factor, best_period, best_score = base_entry

    diag = {
        "objective": float(best_score.get("objective", np.nan)),
        "selection_objective": float(best_score.get("selection_objective", np.nan)),
        "raw_objective": float(best_score.get("raw_objective", np.nan)),
        "scatter_ratio": float(best_score.get("scatter_ratio", np.nan)),
        "lag_phase": float(best_score.get("lag_phase", np.nan)),
        "base_objective": base_objective,
        "harmonic_penalty": float(best_score.get("harmonic_penalty", np.nan)),
        "alias_flag": bool(best_score.get("alias_flag", False)),
        "alias_matches": [float(v) for v in best_score.get("alias_matches", [])],
    }
    return float(best_period), float(best_factor), diag


def _normalize_method_names(method: str | None) -> list[str]:
    method_name = str(method or "pdm").strip().lower()
    if method_name in {"auto", "ce+pdm", "pdm+ce", "auto_ce_pdm"}:
        return list(AUTO_PERIOD_METHODS)
    if method_name in _METHOD_LABELS:
        return [method_name]
    return ["pdm"]


def run_harmonic_check_for_payload(
    payload: dict,
    *,
    plot_dir: Path | None,
    min_period: float,
    max_period: float,
    filter_bad_cameras: bool = True,
    scatter_ratio: float = BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    clean_max_error_absolute: float = CLEAN_LC_MAX_ERROR_ABSOLUTE,
    clean_max_error_sigma: float = CLEAN_LC_MAX_ERROR_SIGMA,
    baseline_name: str = "per_camera_gp",
    baseline_kwargs: dict | None = None,
) -> tuple[dict | None, str]:
    base_period, base_source = resolve_stored_review_period(payload)
    if base_period is None:
        return None, "No stored period; use Find Period."

    lc_path = resolve_lightcurve_path(payload, Path(plot_dir) if plot_dir else None)
    if lc_path is None:
        return None, "No LC file for harmonic check"

    df, _, _ = _load_cleaned_df(
        lc_path,
        filter_bad_cameras=bool(filter_bad_cameras),
        scatter_ratio=float(scatter_ratio),
        clean_max_error_absolute=float(clean_max_error_absolute),
        clean_max_error_sigma=float(clean_max_error_sigma),
    )
    if df is None or df.empty:
        return None, "Empty LC for harmonic check"

    baseline_cache_key = (
        str(lc_path.resolve()),
        (),
        bool(filter_bad_cameras),
        float(scatter_ratio),
        float(clean_max_error_absolute),
        float(clean_max_error_sigma),
    )
    band_dfs = _compute_baseline_bands(
        df,
        str(baseline_name or "per_camera_gp"),
        baseline_cache_key,
        baseline_kwargs=dict(baseline_kwargs or {}),
    )

    best_period, harmonic_factor, harmonic_diag = check_stored_period_harmonics(
        band_dfs,
        base_period,
        min_period=min_period,
        max_period=max_period,
    )
    if not np.isfinite(best_period) or best_period <= 0:
        return None, "No valid harmonic candidate"

    harmonic_divisor = float(harmonic_diag.get("harmonic_divisor", 1.0))
    summary = f"Auto harmonic check: P={best_period:.5f} d"
    if abs(harmonic_divisor - 1.0) > 1e-12:
        summary += f" (P0/{harmonic_divisor:g}; base={base_period:.5f} d from {base_source})"
    else:
        summary += f" (kept {base_source})"
    alias_matches = [float(v) for v in harmonic_diag.get("alias_matches", [])]
    if alias_matches:
        aliases_text = ", ".join(f"{alias:g}" for alias in alias_matches)
        summary += f" [alias~{aliases_text} d]"

    return {
        "best_period": float(best_period),
        "method": "Harmonic check",
        "searched_methods": ["alias_check"],
        "auto": True,
        "harmonic_factor": float(harmonic_factor),
        "harmonic_divisor": harmonic_divisor,
        "base_period": float(base_period),
        "base_period_source": str(base_source),
        "harmonic_objective": float(harmonic_diag.get("objective", np.nan)),
        "harmonic_selection_objective": float(harmonic_diag.get("selection_objective", np.nan)),
        "harmonic_raw_objective": float(harmonic_diag.get("raw_objective", np.nan)),
        "harmonic_lag_phase": float(harmonic_diag.get("lag_phase", np.nan)),
        "harmonic_scatter_ratio": float(harmonic_diag.get("scatter_ratio", np.nan)),
        "alias_flag": bool(harmonic_diag.get("alias_flag", False)),
        "alias_matches": alias_matches,
        "method_candidates": list(harmonic_diag.get("candidates", [])),
    }, summary


def _run_single_method_period_search(
    times: np.ndarray,
    values: np.ndarray,
    *,
    min_period: float,
    max_period: float,
    method_name: str,
) -> tuple[float, str]:
    if method_name == "pdm":
        best_period, _, _ = pdm_find_period(times, values, min_period=min_period, max_period=max_period, refine=True)
    elif method_name == "ce":
        best_period, _, _ = ce_find_period(times, values, min_period=min_period, max_period=max_period, refine=True)
    else:
        best_period, _, _ = lsp_find_period(times, values, min_period=min_period, max_period=max_period, refine=True)
    return float(best_period), _METHOD_LABELS.get(method_name, method_name.upper())


def run_period_search_for_payload(
    payload: dict,
    *,
    plot_dir: Path | None,
    min_period: float,
    max_period: float,
    method: str,
    filter_bad_cameras: bool = True,
    scatter_ratio: float = BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    clean_max_error_absolute: float = CLEAN_LC_MAX_ERROR_ABSOLUTE,
    clean_max_error_sigma: float = CLEAN_LC_MAX_ERROR_SIGMA,
    baseline_name: str = "per_camera_gp",
    baseline_kwargs: dict | None = None,
) -> tuple[dict | None, str]:
    lc_path = resolve_lightcurve_path(payload, Path(plot_dir) if plot_dir else None)
    if lc_path is None:
        return None, "No LC file"

    df, _, _ = _load_cleaned_df(
        lc_path,
        filter_bad_cameras=bool(filter_bad_cameras),
        scatter_ratio=float(scatter_ratio),
        clean_max_error_absolute=float(clean_max_error_absolute),
        clean_max_error_sigma=float(clean_max_error_sigma),
    )
    if df is None or df.empty:
        return None, "Empty LC"

    baseline_cache_key = (
        str(lc_path.resolve()),
        (),
        bool(filter_bad_cameras),
        float(scatter_ratio),
        float(clean_max_error_absolute),
        float(clean_max_error_sigma),
    )
    band_dfs = _compute_baseline_bands(
        df,
        str(baseline_name or "per_camera_gp"),
        baseline_cache_key,
        baseline_kwargs=dict(baseline_kwargs or {}),
    )

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
        return None, "Auto search: Too few points" if str(method or "").strip().lower() == "auto" else "Too few points"

    requested_method = str(method or "pdm").strip().lower()
    method_names = _normalize_method_names(requested_method)
    auto_mode = requested_method in {"auto", "ce+pdm", "pdm+ce", "auto_ce_pdm"}

    candidates: list[dict[str, object]] = []
    for method_name in method_names:
        raw_best_period, label = _run_single_method_period_search(
            times,
            values,
            min_period=min_period,
            max_period=max_period,
            method_name=method_name,
        )
        best_period, harmonic_factor, harmonic_diag = arbitrate_harmonic_period(
            band_dfs,
            raw_best_period,
            min_period=min_period,
            max_period=max_period,
        )
        candidates.append(
            {
                "method_name": method_name,
                "method": label,
                "best_period": float(best_period),
                "base_period": float(raw_best_period),
                "harmonic_factor": float(harmonic_factor),
                "harmonic_objective": float(harmonic_diag.get("objective", np.nan)),
                "harmonic_selection_objective": float(harmonic_diag.get("selection_objective", np.nan)),
                "harmonic_raw_objective": float(harmonic_diag.get("raw_objective", np.nan)),
                "harmonic_lag_phase": float(harmonic_diag.get("lag_phase", np.nan)),
                "harmonic_scatter_ratio": float(harmonic_diag.get("scatter_ratio", np.nan)),
                "alias_flag": bool(harmonic_diag.get("alias_flag", False)),
                "alias_matches": [float(v) for v in harmonic_diag.get("alias_matches", [])],
            }
        )

    finite_candidates = [
        candidate
        for candidate in candidates
        if np.isfinite(float(candidate.get("best_period", np.nan)))
        and np.isfinite(float(candidate.get("harmonic_selection_objective", candidate.get("harmonic_objective", np.nan))))
    ]
    if finite_candidates:
        chosen = min(
            finite_candidates,
            key=lambda candidate: (
                float(candidate.get("harmonic_selection_objective", candidate.get("harmonic_objective", np.inf))),
                bool(candidate.get("alias_flag", False)),
                _METHOD_PRIORITY.get(str(candidate.get("method_name", "")).lower(), 99),
            ),
        )
    else:
        viable = [
            candidate
            for candidate in candidates
            if np.isfinite(float(candidate.get("best_period", np.nan)))
        ]
        if not viable:
            return None, "Auto search: No valid period" if auto_mode else "No valid period"
        chosen = viable[0]

    best_period = float(chosen.get("best_period", np.nan))
    raw_best_period = float(chosen.get("base_period", np.nan))
    harmonic_factor = float(chosen.get("harmonic_factor", 1.0))
    label = str(chosen.get("method", "PDM"))
    prefix = "Auto CE/PDM" if auto_mode else label
    summary = f"{prefix}: P={best_period:.5f} d"
    if auto_mode:
        summary += f" via {label}"
    if abs(harmonic_factor - 1.0) > 1e-12:
        summary += f" (harmonic x{harmonic_factor:g}; base={raw_best_period:.5f} d)"
    alias_matches = [float(v) for v in chosen.get("alias_matches", [])]
    if alias_matches:
        aliases_text = ", ".join(f"{alias:g}" for alias in alias_matches)
        summary += f" [alias~{aliases_text} d]"

    return {
        "best_period": best_period,
        "method": label,
        "searched_methods": [str(candidate.get("method", "")) for candidate in candidates],
        "auto": bool(auto_mode),
        "harmonic_factor": harmonic_factor,
        "base_period": raw_best_period,
        "harmonic_objective": float(chosen.get("harmonic_objective", np.nan)),
        "harmonic_selection_objective": float(chosen.get("harmonic_selection_objective", np.nan)),
        "harmonic_raw_objective": float(chosen.get("harmonic_raw_objective", np.nan)),
        "harmonic_lag_phase": float(chosen.get("harmonic_lag_phase", np.nan)),
        "harmonic_scatter_ratio": float(chosen.get("harmonic_scatter_ratio", np.nan)),
        "alias_flag": bool(chosen.get("alias_flag", False)),
        "alias_matches": alias_matches,
        "method_candidates": candidates,
    }, summary
