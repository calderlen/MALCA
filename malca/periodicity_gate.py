from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from malca.config.config_filters import (
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    CLEAN_LC_MAX_ERROR_ABSOLUTE,
    CLEAN_LC_MAX_ERROR_SIGMA,
    PRE_PERIODICITY_AGREEMENT_REL_TOL,
    PRE_PERIODICITY_CE_MIN_ENTROPY,
    PRE_PERIODICITY_CE_SNR_THRESHOLD,
    PRE_PERIODICITY_LAG_PHASE_MAX,
    PRE_PERIODICITY_MAX_PERIOD,
    PRE_PERIODICITY_MIN_POINTS,
    PRE_PERIODICITY_MIN_PERIOD,
    PRE_PERIODICITY_N_BOOTSTRAP,
    PRE_PERIODICITY_N_PERIODS,
    PRE_PERIODICITY_PDM_MIN_THETA,
    PRE_PERIODICITY_PDM_SNR_THRESHOLD,
    PRE_PERIODICITY_SCATTER_RATIO_MAX,
    PRE_PERIODICITY_SIGNIFICANCE,
    PRE_PERIODICITY_STRONG_SINGLE_SCATTER_RATIO_MAX,
    PRE_PERIODICITY_STRONG_SINGLE_SIG,
)
from malca.config.config_pipeline import WORKERS
from malca.config.config_stats import LS_ALIAS_PERIODS, LS_ALIAS_TOLERANCE
from malca.lightcurve_io import load_lightcurve_df
from malca.stats import compute_ce_stats, compute_pdm_stats
from malca.utils import clean_lc, compute_n_cameras


PREGATE_HARMONIC_FACTORS: tuple[float, ...] = (
    1.0,
    2.0,
    0.5,
    3.0,
    1.0 / 3.0,
    4.0,
    0.25,
)
PREGATE_RESULT_COLUMNS: list[str] = [
    "pre_periodicity_checkpoint_key",
    "pre_periodicity_path",
    "pre_n_points",
    "pre_n_cameras",
    "pre_pdm_period",
    "pre_pdm_theta",
    "pre_pdm_snr",
    "pre_pdm_bootstrap_sig",
    "pre_pdm_support",
    "pre_ce_period",
    "pre_ce_entropy",
    "pre_ce_snr",
    "pre_ce_bootstrap_sig",
    "pre_ce_support",
    "pre_periodicity_method",
    "pre_periodicity_base_period",
    "pre_periodicity_selected_period",
    "pre_periodicity_harmonic_factor",
    "pre_periodicity_methods_agree",
    "pre_periodicity_agreement_factor",
    "pre_periodicity_agreement_rel_err",
    "pre_periodicity_support_count",
    "pre_periodicity_bootstrap_sig",
    "pre_periodicity_is_significant",
    "pre_periodicity_score",
    "pre_periodicity_scatter_ratio",
    "pre_periodicity_lag_phase",
    "pre_periodicity_alias_flag",
    "pre_periodicity_label",
    "pre_periodic_flag",
    "pre_periodicity_reason",
    "pre_periodicity_error",
]


def _empty_result(path: str, checkpoint_key: str, *, label: str = "non_periodic", reason: str = "", error: str | None = None) -> dict[str, object]:
    return {
        "pre_periodicity_checkpoint_key": checkpoint_key,
        "pre_periodicity_path": path,
        "pre_n_points": 0,
        "pre_n_cameras": 0,
        "pre_pdm_period": np.nan,
        "pre_pdm_theta": np.nan,
        "pre_pdm_snr": np.nan,
        "pre_pdm_bootstrap_sig": np.nan,
        "pre_pdm_support": False,
        "pre_ce_period": np.nan,
        "pre_ce_entropy": np.nan,
        "pre_ce_snr": np.nan,
        "pre_ce_bootstrap_sig": np.nan,
        "pre_ce_support": False,
        "pre_periodicity_method": None,
        "pre_periodicity_base_period": np.nan,
        "pre_periodicity_selected_period": np.nan,
        "pre_periodicity_harmonic_factor": np.nan,
        "pre_periodicity_methods_agree": False,
        "pre_periodicity_agreement_factor": np.nan,
        "pre_periodicity_agreement_rel_err": np.nan,
        "pre_periodicity_support_count": 0,
        "pre_periodicity_bootstrap_sig": np.nan,
        "pre_periodicity_is_significant": False,
        "pre_periodicity_score": np.nan,
        "pre_periodicity_scatter_ratio": np.nan,
        "pre_periodicity_lag_phase": np.nan,
        "pre_periodicity_alias_flag": False,
        "pre_periodicity_label": label,
        "pre_periodic_flag": bool(label == "periodic"),
        "pre_periodicity_reason": reason,
        "pre_periodicity_error": error,
    }


def _excluded_camera_set(value: object) -> set[int]:
    if isinstance(value, str):
        return {int(item.strip()) for item in value.split(",") if item.strip()}
    if isinstance(value, (set, list, tuple)):
        return {int(item) for item in value}
    return set()


def _checkpoint_key(path: str, excluded_cameras: object) -> str:
    token = ",".join(str(item) for item in sorted(_excluded_camera_set(excluded_cameras)))
    return f"{path}::{token}"


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

    for bin_idx in range(n_bins):
        vals = resid_valid[idx == bin_idx]
        if vals.size >= min_bin_points:
            template[bin_idx] = float(np.median(vals))
            counts[bin_idx] = int(vals.size)
    return template, counts


def _template_phase_lag(template_a: np.ndarray, template_b: np.ndarray) -> float:
    template_a = np.asarray(template_a, dtype=float)
    template_b = np.asarray(template_b, dtype=float)
    if template_a.size == 0 or template_a.size != template_b.size:
        return np.nan

    n_bins = int(template_a.size)
    best_corr = -np.inf
    best_shift = 0
    min_overlap = max(6, n_bins // 4)

    for shift in range(n_bins):
        shifted = np.roll(template_b, shift)
        mask = np.isfinite(template_a) & np.isfinite(shifted)
        if np.count_nonzero(mask) < min_overlap:
            continue
        a = template_a[mask] - np.mean(template_a[mask])
        b = shifted[mask] - np.mean(shifted[mask])
        sigma_a = float(np.std(a))
        sigma_b = float(np.std(b))
        if sigma_a <= 0 or sigma_b <= 0:
            continue
        corr = float(np.mean((a / sigma_a) * (b / sigma_b)))
        if corr > best_corr:
            best_corr = corr
            best_shift = shift

    if not np.isfinite(best_corr):
        return np.nan
    lag_bins = min(best_shift, n_bins - best_shift)
    return float(lag_bins / n_bins)


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
        lag_phase = _template_phase_lag(templates[0], templates[1])
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


def _arbitrate_harmonic_period(
    band_resid: dict[int, tuple[np.ndarray, np.ndarray]],
    base_period: float,
    *,
    min_period: float,
    max_period: float,
    harmonic_factors: tuple[float, ...] = PREGATE_HARMONIC_FACTORS,
    harmonic_penalty_scale: float = 0.02,
) -> tuple[float, float, dict[str, object]]:
    if not np.isfinite(base_period) or base_period <= 0 or not band_resid:
        return float(base_period), 1.0, {"objective": np.nan, "scatter_ratio": np.nan, "lag_phase": np.nan}

    candidates: list[tuple[float, float, dict[str, object]]] = []
    for factor in harmonic_factors:
        period = float(base_period) * float(factor)
        if not np.isfinite(period) or period <= 0 or period < float(min_period) or period > float(max_period):
            continue
        if any(abs(period - prev_period) <= 1e-10 * max(1.0, abs(period), abs(prev_period)) for _, prev_period, _ in candidates):
            continue
        score = dict(_score_period_harmonic_candidate(band_resid, period))
        harmonic_penalty = float(harmonic_penalty_scale * abs(np.log2(float(factor)))) if factor > 0 else np.inf
        base_objective = float(score.get("objective", np.inf))
        score["selection_objective"] = float(base_objective + harmonic_penalty) if np.isfinite(base_objective) else np.inf
        candidates.append((float(factor), period, score))

    if not candidates:
        return float(base_period), 1.0, {"objective": np.nan, "scatter_ratio": np.nan, "lag_phase": np.nan}

    factor, period, score = min(
        candidates,
        key=lambda item: (
            float(item[2].get("selection_objective", item[2].get("objective", np.inf))),
            bool(item[2].get("alias_flag", False)),
        ),
    )
    return float(period), float(factor), score


def _harmonic_period_agreement(
    period_a: float,
    period_b: float,
    *,
    rel_tol: float,
    harmonic_factors: tuple[float, ...] = PREGATE_HARMONIC_FACTORS,
) -> tuple[bool, float, float]:
    if not np.isfinite(period_a) or not np.isfinite(period_b) or period_a <= 0 or period_b <= 0:
        return False, np.nan, np.nan

    best_factor = np.nan
    best_rel_err = np.inf
    for factor in harmonic_factors:
        candidate = float(period_a) * float(factor)
        if not np.isfinite(candidate) or candidate <= 0:
            continue
        rel_err = abs(candidate - float(period_b)) / max(abs(candidate), abs(float(period_b)), 1e-12)
        if rel_err < best_rel_err:
            best_rel_err = rel_err
            best_factor = float(factor)
    return bool(best_rel_err <= float(rel_tol)), float(best_factor), float(best_rel_err)


def _build_band_residuals(df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    band_resid: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if df.empty or "v_g_band" not in df.columns:
        return band_resid

    for band_value, band_df in df.groupby("v_g_band"):
        try:
            band = int(band_value)
        except Exception:
            continue
        jd = band_df["JD"].to_numpy(dtype=float)
        mag = band_df["mag"].to_numpy(dtype=float)
        valid = np.isfinite(jd) & np.isfinite(mag)
        if np.count_nonzero(valid) < 20:
            continue
        mag_valid = mag[valid]
        resid = mag_valid - float(np.median(mag_valid))
        band_resid[band] = (jd[valid], resid)
    return band_resid


def _choose_period_method(
    pdm_result: dict[str, object],
    ce_result: dict[str, object],
) -> tuple[str | None, float]:
    candidates: list[tuple[float, float, str, float]] = []

    pdm_period = float(pdm_result.get("pdm_period", np.nan))
    if np.isfinite(pdm_period) and pdm_period > 0:
        pdm_sig = float(pdm_result.get("pdm_bootstrap_sig", np.nan))
        pdm_snr = float(pdm_result.get("pdm_snr", np.nan))
        candidates.append((pdm_sig if np.isfinite(pdm_sig) else np.inf, -pdm_snr if np.isfinite(pdm_snr) else np.inf, "pdm", pdm_period))

    ce_period = float(ce_result.get("ce_period", np.nan))
    if np.isfinite(ce_period) and ce_period > 0:
        ce_sig = float(ce_result.get("ce_bootstrap_sig", np.nan))
        ce_snr = float(ce_result.get("ce_snr", np.nan))
        candidates.append((ce_sig if np.isfinite(ce_sig) else np.inf, -ce_snr if np.isfinite(ce_snr) else np.inf, "ce", ce_period))

    if not candidates:
        return None, np.nan

    _, _, method_name, period = min(candidates)
    return str(method_name), float(period)


def _evaluate_periodicity_worker(args: tuple[object, ...]) -> dict[str, object]:
    (
        path_str,
        checkpoint_key,
        excluded_cameras,
        bad_camera_scatter_ratio,
        clean_max_error_absolute,
        clean_max_error_sigma,
        min_period,
        max_period,
        n_periods,
        n_bootstrap,
        significance_level,
        pdm_snr_threshold,
        ce_snr_threshold,
        pdm_theta_threshold,
        ce_entropy_threshold,
        min_points,
        agreement_rel_tol,
        scatter_ratio_max,
        strong_single_scatter_ratio_max,
        lag_phase_max,
        strong_single_sig,
    ) = args

    try:
        df_lc = load_lightcurve_df(
            path_str,
            filter_bad_cameras_enabled=True,
            bad_camera_scatter_ratio=float(bad_camera_scatter_ratio),
        )
        if df_lc.empty:
            return _empty_result(path_str, checkpoint_key, reason="empty_lc")

        excluded = _excluded_camera_set(excluded_cameras)
        if excluded and "camera#" in df_lc.columns:
            df_lc = df_lc[~df_lc["camera#"].isin(excluded)].reset_index(drop=True)

        df_lc = clean_lc(
            df_lc,
            max_error_absolute=float(clean_max_error_absolute),
            max_error_sigma=float(clean_max_error_sigma),
        )
        if df_lc.empty:
            return _empty_result(path_str, checkpoint_key, reason="empty_after_clean")

        n_points = int(len(df_lc))
        n_cameras = int(compute_n_cameras(df_lc))
        if n_points < int(min_points):
            result = _empty_result(path_str, checkpoint_key, reason="too_few_points")
            result["pre_n_points"] = n_points
            result["pre_n_cameras"] = n_cameras
            return result

        jd = df_lc["JD"].to_numpy(dtype=float)
        mag = df_lc["mag"].to_numpy(dtype=float)
        err = df_lc["error"].to_numpy(dtype=float)

        pdm_result = compute_pdm_stats(
            jd,
            mag,
            err,
            min_period=float(min_period),
            max_period=float(max_period),
            n_periods=int(n_periods),
            n_bootstrap=int(n_bootstrap),
            significance_level=float(significance_level),
        )
        ce_result = compute_ce_stats(
            jd,
            mag,
            err,
            min_period=float(min_period),
            max_period=float(max_period),
            n_periods=int(n_periods),
            n_bootstrap=int(n_bootstrap),
            significance_level=float(significance_level),
        )

        pdm_support = bool(
            np.isfinite(pdm_result.get("pdm_snr", np.nan))
            and np.isfinite(pdm_result.get("pdm_min_theta", np.nan))
            and float(pdm_result["pdm_snr"]) >= float(pdm_snr_threshold)
            and float(pdm_result["pdm_min_theta"]) <= float(pdm_theta_threshold)
        )
        ce_support = bool(
            np.isfinite(ce_result.get("ce_snr", np.nan))
            and np.isfinite(ce_result.get("ce_min_entropy", np.nan))
            and float(ce_result["ce_snr"]) >= float(ce_snr_threshold)
            and float(ce_result["ce_min_entropy"]) <= float(ce_entropy_threshold)
        )
        support_count = int(pdm_support) + int(ce_support)

        methods_agree, agreement_factor, agreement_rel_err = _harmonic_period_agreement(
            float(pdm_result.get("pdm_period", np.nan)),
            float(ce_result.get("ce_period", np.nan)),
            rel_tol=float(agreement_rel_tol),
        )

        method_name, base_period = _choose_period_method(pdm_result, ce_result)
        band_resid = _build_band_residuals(df_lc)
        selected_period, harmonic_factor, harmonic_diag = _arbitrate_harmonic_period(
            band_resid,
            float(base_period),
            min_period=float(min_period),
            max_period=float(max_period),
        )

        pdm_sig = float(pdm_result.get("pdm_bootstrap_sig", np.nan))
        ce_sig = float(ce_result.get("ce_bootstrap_sig", np.nan))
        finite_sigs = np.array([pdm_sig, ce_sig], dtype=float)
        finite_sigs = finite_sigs[np.isfinite(finite_sigs)]
        selected_sig = float(np.min(finite_sigs)) if finite_sigs.size > 0 else np.nan
        is_significant = bool(np.isfinite(selected_sig) and selected_sig <= float(significance_level))
        strong_single = bool(np.isfinite(selected_sig) and selected_sig <= float(strong_single_sig))

        scatter_ratio = float(harmonic_diag.get("scatter_ratio", np.nan))
        lag_phase = float(harmonic_diag.get("lag_phase", np.nan))
        alias_flag = bool(harmonic_diag.get("alias_flag", False))
        scatter_ok = bool(np.isfinite(scatter_ratio) and scatter_ratio <= float(scatter_ratio_max))
        strong_single_scatter_ok = bool(
            np.isfinite(scatter_ratio) and scatter_ratio <= float(strong_single_scatter_ratio_max)
        )
        lag_ok = bool((not np.isfinite(lag_phase)) or lag_phase <= float(lag_phase_max))

        label = "non_periodic"
        if is_significant and lag_ok and (
            (support_count >= 2 and methods_agree and scatter_ok)
            or (support_count == 1 and strong_single and strong_single_scatter_ok)
        ):
            label = "periodic"
        elif (
            (support_count >= 1 and np.isfinite(selected_sig) and selected_sig <= max(0.01, float(significance_level) * 10.0))
            or (support_count >= 2 and methods_agree)
            or (support_count >= 1 and scatter_ok)
        ):
            label = "ambiguous"

        score = np.nan
        if np.isfinite(selected_sig):
            min_sig = max(1.0 / float(max(int(n_bootstrap), 1)), 1e-12)
            score = float(-np.log10(np.clip(selected_sig, min_sig, 1.0)))

        reasons: list[str] = []
        if pdm_support:
            reasons.append("pdm")
        if ce_support:
            reasons.append("ce")
        if methods_agree:
            reasons.append("agreement")
        if scatter_ok:
            reasons.append("folded_scatter")
        if alias_flag:
            reasons.append("alias")
        if label == "periodic" and support_count == 1:
            reasons.append("strong_single")
        if not lag_ok:
            reasons.append("lag_mismatch")
        reason = ",".join(reasons) if reasons else "no_strong_periodicity"

        return {
            "pre_periodicity_checkpoint_key": checkpoint_key,
            "pre_periodicity_path": path_str,
            "pre_n_points": n_points,
            "pre_n_cameras": n_cameras,
            "pre_pdm_period": float(pdm_result.get("pdm_period", np.nan)),
            "pre_pdm_theta": float(pdm_result.get("pdm_min_theta", np.nan)),
            "pre_pdm_snr": float(pdm_result.get("pdm_snr", np.nan)),
            "pre_pdm_bootstrap_sig": pdm_sig,
            "pre_pdm_support": pdm_support,
            "pre_ce_period": float(ce_result.get("ce_period", np.nan)),
            "pre_ce_entropy": float(ce_result.get("ce_min_entropy", np.nan)),
            "pre_ce_snr": float(ce_result.get("ce_snr", np.nan)),
            "pre_ce_bootstrap_sig": ce_sig,
            "pre_ce_support": ce_support,
            "pre_periodicity_method": method_name,
            "pre_periodicity_base_period": float(base_period),
            "pre_periodicity_selected_period": float(selected_period),
            "pre_periodicity_harmonic_factor": float(harmonic_factor),
            "pre_periodicity_methods_agree": methods_agree,
            "pre_periodicity_agreement_factor": float(agreement_factor),
            "pre_periodicity_agreement_rel_err": float(agreement_rel_err),
            "pre_periodicity_support_count": support_count,
            "pre_periodicity_bootstrap_sig": selected_sig,
            "pre_periodicity_is_significant": is_significant,
            "pre_periodicity_score": score,
            "pre_periodicity_scatter_ratio": scatter_ratio,
            "pre_periodicity_lag_phase": lag_phase,
            "pre_periodicity_alias_flag": alias_flag,
            "pre_periodicity_label": label,
            "pre_periodic_flag": bool(label == "periodic"),
            "pre_periodicity_reason": reason,
            "pre_periodicity_error": None,
        }
    except Exception as exc:
        return _empty_result(path_str, checkpoint_key, reason="error", error=str(exc))


def _result_is_usable(result: dict[str, object] | None) -> bool:
    if not isinstance(result, dict) or not result:
        return False
    if str(result.get("pre_periodicity_error") or "").strip():
        return False
    return "pre_periodicity_label" in result


def _save_checkpoint(checkpoint_path: Path, results: dict[str, dict[str, object]]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(results.values()))
    if df.empty:
        return
    df = df.drop_duplicates(subset=["pre_periodicity_checkpoint_key"], keep="last")
    df.to_parquet(checkpoint_path, index=False)


def apply_pre_periodicity_gate(
    df: pd.DataFrame,
    *,
    path_col: str | None = None,
    excluded_cameras_col: str | None = "excluded_cameras",
    bad_camera_scatter_ratio: float = BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    clean_max_error_absolute: float = CLEAN_LC_MAX_ERROR_ABSOLUTE,
    clean_max_error_sigma: float = CLEAN_LC_MAX_ERROR_SIGMA,
    min_period: float = PRE_PERIODICITY_MIN_PERIOD,
    max_period: float = PRE_PERIODICITY_MAX_PERIOD,
    n_periods: int = PRE_PERIODICITY_N_PERIODS,
    n_bootstrap: int = PRE_PERIODICITY_N_BOOTSTRAP,
    significance_level: float = PRE_PERIODICITY_SIGNIFICANCE,
    pdm_snr_threshold: float = PRE_PERIODICITY_PDM_SNR_THRESHOLD,
    ce_snr_threshold: float = PRE_PERIODICITY_CE_SNR_THRESHOLD,
    pdm_theta_threshold: float = PRE_PERIODICITY_PDM_MIN_THETA,
    ce_entropy_threshold: float = PRE_PERIODICITY_CE_MIN_ENTROPY,
    min_points: int = PRE_PERIODICITY_MIN_POINTS,
    agreement_rel_tol: float = PRE_PERIODICITY_AGREEMENT_REL_TOL,
    scatter_ratio_max: float = PRE_PERIODICITY_SCATTER_RATIO_MAX,
    strong_single_scatter_ratio_max: float = PRE_PERIODICITY_STRONG_SINGLE_SCATTER_RATIO_MAX,
    lag_phase_max: float = PRE_PERIODICITY_LAG_PHASE_MAX,
    strong_single_sig: float = PRE_PERIODICITY_STRONG_SINGLE_SIG,
    workers: int = WORKERS,
    checkpoint_path: str | Path | None = None,
    show_tqdm: bool = False,
) -> pd.DataFrame:
    df_out = df.copy()
    existing_gate_cols = [column for column in PREGATE_RESULT_COLUMNS if column in df_out.columns]
    if existing_gate_cols:
        df_out = df_out.drop(columns=existing_gate_cols)
    if df_out.empty:
        for column in PREGATE_RESULT_COLUMNS:
            if column not in df_out.columns:
                df_out[column] = pd.Series(dtype="object")
        return df_out

    if path_col is None:
        if "dat_path" in df_out.columns:
            path_col = "dat_path"
        elif "path" in df_out.columns:
            path_col = "path"
        else:
            raise ValueError("Need 'dat_path' or 'path' column for pre-periodicity gate")

    df_out[path_col] = df_out[path_col].astype(str)
    checkpoint_keys = [
        _checkpoint_key(path_value, row.get(excluded_cameras_col) if excluded_cameras_col else None)
        for path_value, (_, row) in zip(df_out[path_col].tolist(), df_out.iterrows())
    ]
    df_out["_pre_periodicity_checkpoint_key"] = checkpoint_keys

    cached_results: dict[str, dict[str, object]] = {}
    checkpoint_file = Path(checkpoint_path).expanduser() if checkpoint_path is not None else None
    if checkpoint_file is not None and checkpoint_file.exists():
        try:
            checkpoint_df = pd.read_parquet(checkpoint_file)
            for _, row in checkpoint_df.iterrows():
                result = row.to_dict()
                key = str(result.get("pre_periodicity_checkpoint_key") or "").strip()
                if key:
                    cached_results[key] = result
        except Exception:
            cached_results = {}

    worker_args: list[tuple[object, ...]] = []
    result_map: dict[str, dict[str, object]] = {}
    for _, row in df_out.iterrows():
        path_value = str(row[path_col])
        checkpoint_key = str(row["_pre_periodicity_checkpoint_key"])
        cached = cached_results.get(checkpoint_key)
        if _result_is_usable(cached):
            result_map[checkpoint_key] = cached
            continue
        worker_args.append(
            (
                path_value,
                checkpoint_key,
                row.get(excluded_cameras_col) if excluded_cameras_col else None,
                float(bad_camera_scatter_ratio),
                float(clean_max_error_absolute),
                float(clean_max_error_sigma),
                float(min_period),
                float(max_period),
                int(n_periods),
                int(n_bootstrap),
                float(significance_level),
                float(pdm_snr_threshold),
                float(ce_snr_threshold),
                float(pdm_theta_threshold),
                float(ce_entropy_threshold),
                int(min_points),
                float(agreement_rel_tol),
                float(scatter_ratio_max),
                float(strong_single_scatter_ratio_max),
                float(lag_phase_max),
                float(strong_single_sig),
            )
        )

    if show_tqdm:
        tqdm.write(
            f"[pre_periodicity_gate] {len(result_map)} cached, processing {len(worker_args)} light curves"
        )

    checkpoint_interval = max(25, len(worker_args) // 20) if worker_args else 25
    processed_since_save = 0

    if workers > 1 and len(worker_args) > 1:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(worker_args))) as executor:
            futures = {
                executor.submit(_evaluate_periodicity_worker, item): item[1]
                for item in worker_args
            }
            iterator = as_completed(futures)
            if show_tqdm:
                iterator = tqdm(iterator, total=len(futures), desc="Pre-periodicity gate")
            for future in iterator:
                result = future.result()
                result_map[str(result["pre_periodicity_checkpoint_key"])] = result
                processed_since_save += 1
                if checkpoint_file is not None and processed_since_save >= checkpoint_interval:
                    _save_checkpoint(checkpoint_file, result_map)
                    processed_since_save = 0
    else:
        iterator = worker_args
        if show_tqdm and worker_args:
            iterator = tqdm(worker_args, desc="Pre-periodicity gate")
        for item in iterator:
            result = _evaluate_periodicity_worker(item)
            result_map[str(result["pre_periodicity_checkpoint_key"])] = result
            processed_since_save += 1
            if checkpoint_file is not None and processed_since_save >= checkpoint_interval:
                _save_checkpoint(checkpoint_file, result_map)
                processed_since_save = 0

    if checkpoint_file is not None and result_map:
        _save_checkpoint(checkpoint_file, result_map)

    results_df = pd.DataFrame(list(result_map.values()))
    if results_df.empty:
        for column in PREGATE_RESULT_COLUMNS:
            if column not in df_out.columns:
                df_out[column] = pd.Series(dtype="object")
        return df_out.drop(columns=["_pre_periodicity_checkpoint_key"])

    results_df = results_df.drop_duplicates(subset=["pre_periodicity_checkpoint_key"], keep="last")
    df_out = df_out.merge(
        results_df,
        left_on="_pre_periodicity_checkpoint_key",
        right_on="pre_periodicity_checkpoint_key",
        how="left",
    )
    df_out = df_out.drop(columns=["_pre_periodicity_checkpoint_key"])
    return df_out.reset_index(drop=True)
