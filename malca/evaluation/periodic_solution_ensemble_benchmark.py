from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/malca-matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from malca.lightcurve_publication import (
    apply_publication_rcparams,
    FIG_TWO_COL_LC_WIDE,
    figsize_from_legacy,
    figsize_two_col_grid,
    plot_lightcurve_panel,
    plot_phase_panel,
    plot_residual_panel,
)

apply_publication_rcparams(plt)

from malca.baseline import per_camera_gp_baseline_masked, phase_template_baseline
from malca.config import (
    BASELINE_JITTER,
    BASELINE_Q,
    BASELINE_S0,
    BASELINE_W0,
    DEFAULT_OUTPUT_DIR,
    LOGBF_THRESHOLD_DIP,
    LOGBF_THRESHOLD_JUMP,
    MAG_POINTS,
    PARQUET_OUTPUT_COMPRESSION,
    P_POINTS,
    RUN_MAX_GAP_POINTS,
    RUN_MIN_POINTS,
    SIGNIFICANCE_THRESHOLD,
    TRIGGER_MODE,
)
from malca.stv.events import score_lightcurve
from malca.evaluation.periodic_branch_simulation_benchmark import (
    add_metric_bins,
    generate_trial_design,
    simulate_periodic_lightcurve,
)


SOLUTION_SPECS: dict[str, dict[str, object]] = {
    "current_template_true_period": {
        "label": "Current phase template, true period",
        "baseline": "phase_template",
        "period_scale": 1.0,
        "period_error_sigma_frac": 0.0,
        "phase_bins": 64,
        "smooth_window": 5,
    },
    "current_template_1pct_period_error": {
        "label": "Current phase template, 1% period scatter",
        "baseline": "phase_template",
        "period_scale": 1.0,
        "period_error_sigma_frac": 0.01,
        "phase_bins": 64,
        "smooth_window": 5,
    },
    "current_template_5pct_period_error": {
        "label": "Current phase template, 5% period scatter",
        "baseline": "phase_template",
        "period_scale": 1.0,
        "period_error_sigma_frac": 0.05,
        "phase_bins": 64,
        "smooth_window": 5,
    },
    "coarse_smooth_template": {
        "label": "Coarse smoothed phase template",
        "baseline": "phase_template",
        "period_scale": 1.0,
        "period_error_sigma_frac": 0.0,
        "phase_bins": 32,
        "smooth_window": 7,
    },
    "fine_template": {
        "label": "Fine phase template",
        "baseline": "phase_template",
        "period_scale": 1.0,
        "period_error_sigma_frac": 0.0,
        "phase_bins": 96,
        "smooth_window": 3,
    },
    "leave_cycle_template": {
        "label": "Leave-cycle-out phase template",
        "baseline": "leave_cycle_template",
        "period_scale": 1.0,
        "period_error_sigma_frac": 0.0,
        "phase_bins": 64,
    },
    "fourier_3harmonic": {
        "label": "Fourier 3-harmonic template",
        "baseline": "fourier",
        "period_scale": 1.0,
        "period_error_sigma_frac": 0.0,
        "n_harmonics": 3,
    },
    "ensemble_template_fourier": {
        "label": "Median ensemble: template + leave-cycle + Fourier",
        "baseline": "ensemble",
        "period_scale": 1.0,
        "period_error_sigma_frac": 0.0,
    },
    "gp_masked_control": {
        "label": "Masked GP control",
        "baseline": "gp_masked",
        "period_scale": np.nan,
        "period_error_sigma_frac": np.nan,
    },
}

DEFAULT_SOLUTION_MODES: tuple[str, ...] = (
    "current_template_true_period",
    "current_template_1pct_period_error",
    "current_template_5pct_period_error",
    "coarse_smooth_template",
    "fine_template",
    "leave_cycle_template",
    "fourier_3harmonic",
    "ensemble_template_fourier",
    "gp_masked_control",
)


@dataclass
class PeriodicSolutionBenchmarkConfig:
    output_base_dir: Path = DEFAULT_OUTPUT_DIR / "diagnostics" / "periodic_solution_ensemble_benchmark"
    run_tag: str | None = None
    n_trials: int = 24000
    seed: int = 20260514
    workers: int = 8
    show_progress: bool = True
    force: bool = False
    mode_names: tuple[str, ...] = DEFAULT_SOLUTION_MODES

    control_fraction: float = 0.10
    small_dip_fraction: float = 0.50
    medium_dip_fraction: float = 0.32
    broad_dip_fraction: float = 0.08

    trigger_mode: str = TRIGGER_MODE
    logbf_threshold_dip: float = LOGBF_THRESHOLD_DIP
    logbf_threshold_jump: float = LOGBF_THRESHOLD_JUMP
    significance_threshold: float = SIGNIFICANCE_THRESHOLD
    p_points: int = P_POINTS
    mag_points: int = MAG_POINTS
    run_min_points: int = RUN_MIN_POINTS
    run_max_gap_points: int = RUN_MAX_GAP_POINTS
    run_max_gap_days: float | None = None
    run_min_duration_days: float = 0.0
    compute_event_prob: bool = True

    baseline_s0: float = BASELINE_S0
    baseline_w0: float = BASELINE_W0
    baseline_q: float = BASELINE_Q
    baseline_jitter: float = BASELINE_JITTER
    baseline_sigma_floor: float | None = None

    phase_local_bins: int = 64
    phase_local_min_neighbors: int = 6
    phase_local_snr_threshold: float = 3.0


@dataclass
class PeriodicSolutionBenchmarkRun:
    config: PeriodicSolutionBenchmarkConfig
    run_dir: Path
    trial_design: pd.DataFrame
    solution_results: pd.DataFrame
    summary_overall: pd.DataFrame
    summary_slices: dict[str, pd.DataFrame]


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def make_run_dir(config: PeriodicSolutionBenchmarkConfig) -> Path:
    tag = config.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_base_dir).expanduser() / str(tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_config(run_dir: Path, config: PeriodicSolutionBenchmarkConfig) -> None:
    payload = asdict(config)
    payload["output_base_dir"] = str(payload["output_base_dir"])
    payload["mode_names"] = list(payload["mode_names"])
    with (run_dir / "config.json").open("w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _stable_text_token(text: str) -> int:
    value = 0
    for char in str(text):
        value = (value * 131 + ord(char)) & 0xFFFFFFFF
    return int(value)


def _rng_for(seed: int, *tokens: int) -> np.random.Generator:
    value = int(seed) & 0xFFFFFFFF
    for token in tokens:
        value = (value * 1664525 + int(token) + 1013904223) & 0xFFFFFFFF
    return np.random.default_rng(value)


def _robust_sigma(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return np.nan
    med = float(np.median(vals))
    mad = 1.4826 * float(np.median(np.abs(vals - med)))
    if np.isfinite(mad) and mad > 0:
        return mad
    sigma = float(np.nanstd(vals))
    return sigma if np.isfinite(sigma) and sigma > 0 else np.nan


def _phase_distance(a: np.ndarray, b: np.ndarray | float) -> np.ndarray:
    diff = np.abs(np.asarray(a, dtype=float) - b)
    return np.minimum(diff, 1.0 - diff)


def _offsets_by_camera_band(
    df: pd.DataFrame,
    *,
    mag_col: str = "mag",
    cam_col: str = "camera#",
    band_col: str = "v_g_band",
    min_camera_band_points: int = 8,
) -> np.ndarray:
    mag = pd.to_numeric(df[mag_col], errors="coerce")
    finite = mag[np.isfinite(mag)]
    global_median = float(np.median(finite)) if not finite.empty else 0.0
    offsets = np.full(len(df), global_median, dtype=float)
    work = pd.DataFrame(index=df.index)
    work["_mag"] = mag

    if cam_col in df.columns:
        work["_camera"] = df[cam_col]
        cam_median = work.groupby("_camera")["_mag"].median()
        work = work.join(cam_median.rename("_camera_median"), on="_camera")
        vals = pd.to_numeric(work["_camera_median"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(vals)
        offsets[mask] = vals[mask]

    if band_col in df.columns:
        work["_band"] = pd.to_numeric(df[band_col], errors="coerce")
        band_median = work.groupby("_band")["_mag"].median()
        work = work.join(band_median.rename("_band_median"), on="_band")
        vals = pd.to_numeric(work["_band_median"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(vals)
        offsets[mask] = vals[mask]

        if "_camera" in work.columns:
            cam_band = (
                work.groupby(["_band", "_camera"], dropna=False)["_mag"]
                .agg(["median", "size"])
                .rename(columns={"median": "_camera_band_median", "size": "_camera_band_size"})
            )
            work = work.join(cam_band, on=["_band", "_camera"])
            counts = pd.to_numeric(work["_camera_band_size"], errors="coerce").to_numpy(dtype=float)
            vals = pd.to_numeric(work["_camera_band_median"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(counts) & (counts >= int(min_camera_band_points)) & np.isfinite(vals)
            offsets[mask] = vals[mask]

    return offsets


def _finish_residual_baseline(
    df: pd.DataFrame,
    baseline: np.ndarray,
    *,
    source: str,
    mag_col: str = "mag",
    err_col: str = "error",
    cam_col: str = "camera#",
) -> pd.DataFrame:
    out = df.copy()
    mag = pd.to_numeric(out[mag_col], errors="coerce").to_numpy(dtype=float)
    err = pd.to_numeric(out[err_col], errors="coerce").to_numpy(dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    resid = mag - baseline
    out["baseline"] = baseline
    out["resid"] = resid
    out["baseline_source"] = source
    out["sigma_eff"] = np.nan
    out["sigma_resid"] = np.nan

    groups = out.groupby(cam_col, group_keys=False) if cam_col in out.columns else [(None, out)]
    for _, sub in groups:
        idx = sub.index
        resid_here = pd.to_numeric(out.loc[idx, "resid"], errors="coerce").to_numpy(dtype=float)
        err_here = pd.to_numeric(out.loc[idx, err_col], errors="coerce").to_numpy(dtype=float)
        scatter = _robust_sigma(resid_here)
        err_good = err_here[np.isfinite(err_here) & (err_here > 0)]
        err_med = float(np.median(err_good)) if err_good.size else np.nan
        scatter_num = scatter if np.isfinite(scatter) else 0.0
        err_med_num = err_med if np.isfinite(err_med) else 0.0
        err_safe = np.where(np.isfinite(err_here) & (err_here > 0), err_here, err_med_num)
        sigma_eff = np.sqrt(np.maximum(err_safe**2 + scatter_num**2, 1e-12))
        out.loc[idx, "sigma_eff"] = sigma_eff
        out.loc[idx, "sigma_resid"] = resid_here / sigma_eff

    return out


def leave_cycle_phase_template_baseline(
    df: pd.DataFrame,
    *,
    period_days: float | None = None,
    phase_bins: int = 64,
    min_neighbors: int = 6,
    t_col: str = "JD",
    mag_col: str = "mag",
    err_col: str = "error",
    cam_col: str = "camera#",
    band_col: str = "v_g_band",
    **kwargs: object,
) -> pd.DataFrame:
    try:
        period = float(period_days)
    except (TypeError, ValueError):
        period = np.nan
    if not np.isfinite(period) or period <= 0:
        return phase_template_baseline(df, period_days=period_days, **kwargs)

    jd = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
    mag = pd.to_numeric(df[mag_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(jd) & np.isfinite(mag)
    if np.count_nonzero(finite) < max(20, int(min_neighbors) * 4):
        return phase_template_baseline(df, period_days=period, **kwargs)

    jd0 = float(np.nanmin(jd[finite]))
    phase = np.full(len(df), np.nan, dtype=float)
    phase[finite] = np.mod((jd[finite] - jd0) / period, 1.0)
    cycle = np.full(len(df), -1, dtype=int)
    cycle[finite] = np.floor((jd[finite] - jd0) / period).astype(int)
    offsets = _offsets_by_camera_band(df, mag_col=mag_col, cam_col=cam_col, band_col=band_col)
    centered = mag - offsets
    n_bins = max(int(phase_bins), 8)
    bin_idx = np.zeros(len(df), dtype=int)
    phase_finite = np.isfinite(phase)
    bin_idx[phase_finite] = np.floor(phase[phase_finite] * n_bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    global_template = np.full(n_bins, np.nan, dtype=float)
    for bin_id in range(n_bins):
        vals = centered[finite & (bin_idx == bin_id)]
        if vals.size >= max(3, int(min_neighbors) // 2):
            global_template[bin_id] = float(np.median(vals))

    finite_template = np.isfinite(global_template)
    if not finite_template.any():
        return phase_template_baseline(df, period_days=period, **kwargs)
    if finite_template.sum() < n_bins:
        centers = (np.arange(n_bins, dtype=float) + 0.5) / float(n_bins)
        xp = np.concatenate([centers[finite_template] - 1.0, centers[finite_template], centers[finite_template] + 1.0])
        fp = np.concatenate([global_template[finite_template]] * 3)
        global_template = np.interp(centers, xp, fp)

    model = np.full(len(df), np.nan, dtype=float)
    for idx in np.flatnonzero(finite):
        same_bin = finite & (bin_idx == bin_idx[idx]) & (cycle != cycle[idx])
        vals = centered[same_bin]
        if vals.size >= int(min_neighbors):
            model[idx] = float(np.median(vals))
        else:
            model[idx] = float(global_template[bin_idx[idx]])

    missing = ~np.isfinite(model)
    if missing.any():
        model[missing] = global_template[bin_idx[missing]]
    baseline = model + offsets
    return _finish_residual_baseline(df, baseline, source="leave_cycle_phase_template", mag_col=mag_col, err_col=err_col, cam_col=cam_col)


def fourier_phase_baseline(
    df: pd.DataFrame,
    *,
    period_days: float | None = None,
    n_harmonics: int = 3,
    t_col: str = "JD",
    mag_col: str = "mag",
    err_col: str = "error",
    cam_col: str = "camera#",
    band_col: str = "v_g_band",
    **kwargs: object,
) -> pd.DataFrame:
    try:
        period = float(period_days)
    except (TypeError, ValueError):
        period = np.nan
    if not np.isfinite(period) or period <= 0:
        return phase_template_baseline(df, period_days=period_days, **kwargs)

    jd = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
    mag = pd.to_numeric(df[mag_col], errors="coerce").to_numpy(dtype=float)
    err = pd.to_numeric(df[err_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(jd) & np.isfinite(mag)
    if np.count_nonzero(finite) < max(20, 2 * int(n_harmonics) + 8):
        return phase_template_baseline(df, period_days=period, **kwargs)

    jd0 = float(np.nanmin(jd[finite]))
    phase = np.mod((jd - jd0) / period, 1.0)
    offsets = _offsets_by_camera_band(df, mag_col=mag_col, cam_col=cam_col, band_col=band_col)
    centered = mag - offsets

    columns = [np.ones_like(phase)]
    for harmonic in range(1, int(n_harmonics) + 1):
        angle = 2.0 * np.pi * harmonic * phase
        columns.append(np.sin(angle))
        columns.append(np.cos(angle))
    x = np.vstack(columns).T

    good = finite & np.isfinite(centered)
    if err_col in df.columns:
        good &= np.isfinite(err) & (err > 0)
    xg = x[good]
    yg = centered[good]
    if xg.shape[0] <= xg.shape[1]:
        return phase_template_baseline(df, period_days=period, **kwargs)

    if err_col in df.columns:
        wg = 1.0 / np.maximum(err[good], 1e-4)
        xw = xg * wg[:, None]
        yw = yg * wg
    else:
        xw = xg
        yw = yg

    try:
        coef, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    except Exception:
        return phase_template_baseline(df, period_days=period, **kwargs)

    model = x @ coef
    baseline = model + offsets
    return _finish_residual_baseline(df, baseline, source=f"phase_fourier_h{int(n_harmonics)}", mag_col=mag_col, err_col=err_col, cam_col=cam_col)


def ensemble_phase_baseline(
    df: pd.DataFrame,
    *,
    period_days: float | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    baselines: list[np.ndarray] = []
    for func, extra in (
        (phase_template_baseline, {"period_days": period_days}),
        (leave_cycle_phase_template_baseline, {"period_days": period_days}),
        (fourier_phase_baseline, {"period_days": period_days, "n_harmonics": 3}),
    ):
        try:
            result = func(df, **{**kwargs, **extra})
        except Exception:
            continue
        if isinstance(result, pd.DataFrame) and "baseline" in result.columns:
            baseline = pd.to_numeric(result["baseline"], errors="coerce").to_numpy(dtype=float)
            if baseline.size == len(df) and np.isfinite(baseline).any():
                baselines.append(baseline)

    if not baselines:
        return phase_template_baseline(df, period_days=period_days, **kwargs)

    stack = np.vstack(baselines)
    baseline = np.nanmedian(stack, axis=0)
    return _finish_residual_baseline(df, baseline, source="phase_baseline_ensemble_median")


def selected_period_for_mode(row: pd.Series | dict[str, object], mode_name: str, seed: int) -> float:
    spec = SOLUTION_SPECS[mode_name]
    if str(spec["baseline"]) == "gp_masked":
        return np.nan
    base_period = float(row["period_days"])
    period = base_period * float(spec.get("period_scale", 1.0))
    sigma = float(spec.get("period_error_sigma_frac", 0.0) or 0.0)
    if sigma > 0:
        rng = _rng_for(int(row["trial_seed"]), int(seed), _stable_text_token(mode_name))
        period *= 1.0 + float(rng.normal(0.0, sigma))
    return float(max(period, 1e-6))


def baseline_for_mode(
    mode_name: str,
    selected_period: float,
    config: PeriodicSolutionBenchmarkConfig,
) -> tuple[object, dict[str, object]]:
    spec = SOLUTION_SPECS[mode_name]
    baseline_name = str(spec["baseline"])
    base_kwargs: dict[str, object] = {
        "S0": float(config.baseline_s0),
        "w0": float(config.baseline_w0),
        "q": float(config.baseline_q),
        "jitter": float(config.baseline_jitter),
        "sigma_floor": config.baseline_sigma_floor,
        "add_sigma_eff_col": True,
    }

    if baseline_name == "gp_masked":
        return per_camera_gp_baseline_masked, base_kwargs
    if baseline_name == "phase_template":
        base_kwargs.update(
            {
                "period_days": selected_period,
                "phase_bins": int(spec.get("phase_bins", 64)),
                "profile_smooth_window": int(spec.get("smooth_window", 5)),
            }
        )
        return phase_template_baseline, base_kwargs
    if baseline_name == "leave_cycle_template":
        base_kwargs.update(
            {
                "period_days": selected_period,
                "phase_bins": int(spec.get("phase_bins", 64)),
                "min_neighbors": int(config.phase_local_min_neighbors),
            }
        )
        return leave_cycle_phase_template_baseline, base_kwargs
    if baseline_name == "fourier":
        base_kwargs.update(
            {
                "period_days": selected_period,
                "n_harmonics": int(spec.get("n_harmonics", 3)),
            }
        )
        return fourier_phase_baseline, base_kwargs
    if baseline_name == "ensemble":
        base_kwargs.update({"period_days": selected_period})
        return ensemble_phase_baseline, base_kwargs
    raise ValueError(f"Unsupported solution baseline '{baseline_name}' for mode {mode_name}")


def phase_local_outlier_stats(
    clean_df: pd.DataFrame,
    df_base: pd.DataFrame,
    *,
    selected_period: float,
    n_bins: int,
    min_neighbors: int,
    min_points: int,
    snr_threshold: float,
) -> dict[str, object]:
    if not np.isfinite(selected_period) or selected_period <= 0 or clean_df.empty:
        return {
            "phase_local_peak_snr": np.nan,
            "phase_local_truth_peak_snr": np.nan,
            "phase_local_detected": False,
            "phase_local_truth_points_above_threshold": 0,
            "same_phase_truth_contrast_mag": np.nan,
        }

    jd = pd.to_numeric(clean_df["JD"], errors="coerce").to_numpy(dtype=float)
    resid = pd.to_numeric(df_base["resid"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(jd) & np.isfinite(resid)
    if np.count_nonzero(finite) < max(10, int(min_neighbors) * 2):
        return {
            "phase_local_peak_snr": np.nan,
            "phase_local_truth_peak_snr": np.nan,
            "phase_local_detected": False,
            "phase_local_truth_points_above_threshold": 0,
            "same_phase_truth_contrast_mag": np.nan,
        }

    truth_mask = (
        clean_df["truth_dip_mask"].to_numpy(dtype=bool)
        if "truth_dip_mask" in clean_df.columns
        else np.zeros(len(clean_df), dtype=bool)
    )
    jd0 = float(np.nanmin(jd[finite]))
    phase = np.full(len(clean_df), np.nan, dtype=float)
    phase[finite] = np.mod((jd[finite] - jd0) / float(selected_period), 1.0)
    cycle = np.full(len(clean_df), -1, dtype=int)
    cycle[finite] = np.floor((jd[finite] - jd0) / float(selected_period)).astype(int)
    n_bins = max(int(n_bins), 8)
    half_width = 0.75 / float(n_bins)
    z = np.full(len(clean_df), np.nan, dtype=float)
    local_contrast = np.full(len(clean_df), np.nan, dtype=float)

    for idx in np.flatnonzero(finite):
        neighbors = (
            finite
            & (cycle != cycle[idx])
            & (_phase_distance(phase, phase[idx]) <= half_width)
        )
        vals = resid[neighbors]
        vals = vals[np.isfinite(vals)]
        if vals.size < int(min_neighbors):
            neighbors = finite & (cycle != cycle[idx]) & (_phase_distance(phase, phase[idx]) <= 1.5 * half_width)
            vals = resid[neighbors]
            vals = vals[np.isfinite(vals)]
        if vals.size < int(min_neighbors):
            continue
        med = float(np.median(vals))
        sigma = _robust_sigma(vals)
        if np.isfinite(sigma) and sigma > 0:
            local_contrast[idx] = float(resid[idx] - med)
            z[idx] = float((resid[idx] - med) / sigma)

    phase_local_peak_snr = float(np.nanmax(z)) if np.isfinite(z).any() else np.nan
    truth_z = z[truth_mask]
    truth_contrast = local_contrast[truth_mask]
    truth_peak_snr = float(np.nanmax(truth_z)) if truth_z.size and np.isfinite(truth_z).any() else np.nan
    contrast_mag = float(np.nanmax(truth_contrast)) if truth_contrast.size and np.isfinite(truth_contrast).any() else np.nan
    truth_points = int(np.count_nonzero(np.isfinite(truth_z) & (truth_z >= float(snr_threshold))))
    return {
        "phase_local_peak_snr": phase_local_peak_snr,
        "phase_local_truth_peak_snr": truth_peak_snr,
        "phase_local_detected": bool(truth_points >= int(min_points)),
        "phase_local_truth_points_above_threshold": truth_points,
        "same_phase_truth_contrast_mag": contrast_mag,
    }


def _score_detection_for_solution(
    df_lc: pd.DataFrame,
    row: pd.Series,
    mode_name: str,
    config: PeriodicSolutionBenchmarkConfig,
) -> dict[str, object]:
    selected_period = selected_period_for_mode(row, mode_name, config.seed)
    baseline_func, baseline_kwargs = baseline_for_mode(mode_name, selected_period, config)
    scored = score_lightcurve(
        df_lc,
        baseline_func=baseline_func,
        baseline_kwargs=baseline_kwargs,
        filter_residual_bad_cameras_enabled=False,
        p_points=int(config.p_points),
        mag_points=int(config.mag_points),
        trigger_mode=str(config.trigger_mode),
        logbf_threshold_dip=float(config.logbf_threshold_dip),
        logbf_threshold_jump=float(config.logbf_threshold_jump),
        significance_threshold=float(config.significance_threshold),
        run_min_points=int(config.run_min_points),
        max_gap_points=int(config.run_max_gap_points),
        run_max_gap_days=config.run_max_gap_days,
        run_min_duration_days=config.run_min_duration_days,
        compute_event_prob=bool(config.compute_event_prob),
    )
    clean_df = scored["df"].reset_index(drop=True)
    df_base = scored["df_base"].reset_index(drop=True)
    dip = scored["dip"]
    event_indices = np.asarray(dip.get("event_indices", []), dtype=int)
    event_indices = event_indices[(event_indices >= 0) & (event_indices < len(clean_df))]
    event_mask = np.zeros(len(clean_df), dtype=bool)
    event_mask[event_indices] = True
    truth_mask = clean_df["truth_dip_mask"].to_numpy(dtype=bool) if "truth_dip_mask" in clean_df.columns else np.zeros(len(clean_df), dtype=bool)

    has_dip = bool(row["has_dip"])
    truth_observable = bool(clean_df["truth_observable"].iloc[0]) if len(clean_df) and "truth_observable" in clean_df.columns else False
    overlap_points = int(np.count_nonzero(event_mask & truth_mask))
    detected_overlap = bool(overlap_points > 0)
    dip_significant = bool(dip.get("significant", False))
    target_recovered = bool(has_dip and dip_significant and detected_overlap)
    false_positive = bool((not has_dip) and dip_significant)
    off_target_detection = bool(has_dip and dip_significant and not detected_overlap)

    outside = ~truth_mask
    if not outside.any():
        outside = np.ones(len(clean_df), dtype=bool)
    baseline_values = pd.to_numeric(df_base["baseline"], errors="coerce").to_numpy(dtype=float)
    true_baseline = pd.to_numeric(clean_df["true_baseline_mag"], errors="coerce").to_numpy(dtype=float)
    resid = pd.to_numeric(df_base["resid"], errors="coerce").to_numpy(dtype=float)
    truth_signal = pd.to_numeric(clean_df["truth_dip_signal"], errors="coerce").to_numpy(dtype=float)
    baseline_error = baseline_values - true_baseline
    recovered_amp = float(np.nanmax(resid[truth_mask])) if truth_mask.any() and np.isfinite(resid[truth_mask]).any() else np.nan
    true_amp_sampled = float(np.nanmax(truth_signal)) if np.isfinite(truth_signal).any() else np.nan
    phase_stats = phase_local_outlier_stats(
        clean_df,
        df_base,
        selected_period=selected_period,
        n_bins=int(config.phase_local_bins),
        min_neighbors=int(config.phase_local_min_neighbors),
        min_points=int(config.run_min_points),
        snr_threshold=float(config.phase_local_snr_threshold),
    )

    return {
        "mode": mode_name,
        "mode_label": str(SOLUTION_SPECS[mode_name]["label"]),
        "baseline_strategy": str(SOLUTION_SPECS[mode_name]["baseline"]),
        "status": "ok",
        "error": "",
        "selected_period_days": selected_period,
        "period_frac_error": (
            abs(selected_period - float(row["period_days"])) / float(row["period_days"])
            if np.isfinite(selected_period)
            else np.nan
        ),
        "n_points_actual": int(len(clean_df)),
        "n_cameras_actual": int(clean_df["camera#"].nunique()) if "camera#" in clean_df.columns else 0,
        "truth_support_points_actual": int(np.count_nonzero(truth_mask)),
        "truth_peak_snr_actual": float(np.nanmax(truth_signal / clean_df["error"].to_numpy(dtype=float))) if has_dip and len(clean_df) else 0.0,
        "truth_observable_actual": truth_observable,
        "dip_significant": dip_significant,
        "target_recovered": target_recovered,
        "false_positive": false_positive,
        "off_target_detection": off_target_detection,
        "detected_overlap": detected_overlap,
        "overlap_points": overlap_points,
        "event_points": int(event_mask.sum()),
        "dip_run_count": int(dip.get("n_runs", 0)),
        "dip_count": int(len(event_indices)),
        "dip_max_log_bf_local": float(dip.get("max_log_bf_local", np.nan)),
        "dip_bayes_factor": float(dip.get("bayes_factor", np.nan)),
        "dip_trigger_max": float(dip.get("trigger_max", np.nan)),
        "dip_max_run_points": int(dip.get("max_run_points", 0)),
        "dip_max_run_duration": float(dip.get("max_run_duration", np.nan)),
        "baseline_source": str(dip.get("baseline_source", "unknown")),
        "baseline_mae_outside_dip": float(np.nanmedian(np.abs(baseline_error[outside]))),
        "baseline_rmse_outside_dip": float(np.sqrt(np.nanmean(baseline_error[outside] ** 2))),
        "resid_rms_outside_dip": float(np.sqrt(np.nanmean(resid[outside] ** 2))),
        "resid_mad_outside_dip": float(1.4826 * np.nanmedian(np.abs(resid[outside] - np.nanmedian(resid[outside])))),
        "recovered_amp_mag": recovered_amp,
        "true_amp_sampled_mag": true_amp_sampled,
        "amp_recovery_ratio": float(recovered_amp / true_amp_sampled) if np.isfinite(recovered_amp) and true_amp_sampled > 0 else np.nan,
        **phase_stats,
    }


def _evaluate_task(task: tuple[dict[str, object], str, dict[str, object]]) -> dict[str, object]:
    row_dict, mode_name, config_dict = task
    row = pd.Series(row_dict)
    config = PeriodicSolutionBenchmarkConfig(**config_dict)
    try:
        df = simulate_periodic_lightcurve(row)
        out = _score_detection_for_solution(df, row, mode_name, config)
        out["trial_id"] = int(row["trial_id"])
        return out
    except Exception as exc:
        return {
            "trial_id": int(row_dict.get("trial_id", -1)),
            "mode": mode_name,
            "mode_label": str(SOLUTION_SPECS.get(mode_name, {}).get("label", mode_name)),
            "baseline_strategy": str(SOLUTION_SPECS.get(mode_name, {}).get("baseline", "")),
            "status": "error",
            "error": str(exc),
        }


def _config_to_worker_dict(config: PeriodicSolutionBenchmarkConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["output_base_dir"] = str(payload["output_base_dir"])
    payload["mode_names"] = tuple(payload["mode_names"])
    return payload


def evaluate_design(
    design: pd.DataFrame,
    config: PeriodicSolutionBenchmarkConfig,
    *,
    output_path: Path | None = None,
) -> pd.DataFrame:
    tasks: list[tuple[dict[str, object], str, dict[str, object]]] = []
    cfg = _config_to_worker_dict(config)
    for _, row in design.iterrows():
        row_dict = row.to_dict()
        for mode_name in config.mode_names:
            if mode_name not in SOLUTION_SPECS:
                raise ValueError(f"Unknown periodic solution mode: {mode_name}")
            tasks.append((row_dict, str(mode_name), cfg))

    results: list[dict[str, object]] = []
    if config.workers and int(config.workers) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(config.workers)) as executor:
            futures = [executor.submit(_evaluate_task, task) for task in tasks]
            iterator = as_completed(futures)
            if config.show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Periodic solution benchmark")
            for future in iterator:
                results.append(future.result())
    else:
        iterator = tasks
        if config.show_progress:
            iterator = tqdm(tasks, desc="Periodic solution benchmark")
        for task in iterator:
            results.append(_evaluate_task(task))

    out = pd.DataFrame(results)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)
    return out


def merge_design_results(design: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    merged = results.merge(design, on="trial_id", how="left", suffixes=("", "_design"))
    return add_metric_bins(merged)


def _rate(series: pd.Series) -> float:
    if len(series) == 0:
        return np.nan
    return float(series.fillna(False).astype(bool).mean())


def summarize_results(df: pd.DataFrame, group_cols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ok = df[df["status"].eq("ok")].copy()
    grouped = ok.groupby(list(group_cols), dropna=False) if group_cols else [((), ok)]
    for key, sub in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {col: val for col, val in zip(group_cols, key_tuple)}
        has_dip = sub["has_dip"].fillna(False).astype(bool)
        observable = has_dip & sub["truth_observable_actual"].fillna(False).astype(bool)
        controls = ~has_dip
        detected = sub["dip_significant"].fillna(False).astype(bool)
        recovered = sub["target_recovered"].fillna(False).astype(bool)
        local_detected = sub["phase_local_detected"].fillna(False).astype(bool)
        true_positive_detected = detected & has_dip & sub["detected_overlap"].fillna(False).astype(bool)

        row.update(
            {
                "n": int(len(sub)),
                "n_dip": int(has_dip.sum()),
                "n_observable_dip": int(observable.sum()),
                "n_control": int(controls.sum()),
                "dip_significant_rate": _rate(detected),
                "observable_recall": float(recovered[observable].mean()) if observable.any() else np.nan,
                "all_dip_recall": float(recovered[has_dip].mean()) if has_dip.any() else np.nan,
                "phase_local_observable_recall": float(local_detected[observable].mean()) if observable.any() else np.nan,
                "control_false_positive_rate": float(detected[controls].mean()) if controls.any() else np.nan,
                "off_target_detection_rate": _rate(sub.loc[has_dip, "off_target_detection"]) if has_dip.any() else np.nan,
                "precision_by_trial": float(true_positive_detected.sum() / detected.sum()) if detected.any() else np.nan,
                "median_baseline_mae_outside_dip": float(sub["baseline_mae_outside_dip"].median()),
                "median_baseline_rmse_outside_dip": float(sub["baseline_rmse_outside_dip"].median()),
                "median_resid_rms_outside_dip": float(sub["resid_rms_outside_dip"].median()),
                "median_resid_mad_outside_dip": float(sub["resid_mad_outside_dip"].median()),
                "median_amp_recovery_ratio": float(sub.loc[has_dip, "amp_recovery_ratio"].median()) if has_dip.any() else np.nan,
                "median_phase_local_truth_peak_snr": float(sub.loc[has_dip, "phase_local_truth_peak_snr"].median()) if has_dip.any() else np.nan,
                "median_same_phase_truth_contrast_mag": float(sub.loc[has_dip, "same_phase_truth_contrast_mag"].median()) if has_dip.any() else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary_slices(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "overall_by_mode": summarize_results(merged, ["mode", "mode_label", "baseline_strategy"]),
        "by_mode_and_amp": summarize_results(merged, ["mode", "dip_amp_bin"]),
        "by_mode_and_period": summarize_results(merged, ["mode", "period_bin"]),
        "by_mode_and_width": summarize_results(merged, ["mode", "width_bin"]),
        "by_mode_and_points": summarize_results(merged, ["mode", "points_bin_actual"]),
        "by_mode_and_waveform": summarize_results(merged, ["mode", "waveform_kind"]),
    }


def run_periodic_solution_benchmark(config: PeriodicSolutionBenchmarkConfig) -> PeriodicSolutionBenchmarkRun:
    run_dir = make_run_dir(config)
    write_config(run_dir, config)

    design_path = run_dir / "trial_design.parquet"
    results_path = run_dir / "solution_results.parquet"
    merged_path = run_dir / "solution_results_with_design.parquet"

    if design_path.exists() and not config.force:
        design = pd.read_parquet(design_path)
    else:
        design = generate_trial_design(config)  # type: ignore[arg-type]
        design_path.parent.mkdir(parents=True, exist_ok=True)
        design.to_parquet(design_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)

    if results_path.exists() and not config.force:
        results = pd.read_parquet(results_path)
    else:
        results = evaluate_design(design, config, output_path=results_path)

    merged = merge_design_results(design, results)
    merged.to_parquet(merged_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)
    slices = build_summary_slices(merged)
    summary_dir = run_dir / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for name, table in slices.items():
        table.to_parquet(summary_dir / f"{name}.parquet", index=False, compression=PARQUET_OUTPUT_COMPRESSION)
        table.to_csv(summary_dir / f"{name}.csv", index=False)

    return PeriodicSolutionBenchmarkRun(
        config=config,
        run_dir=run_dir,
        trial_design=design,
        solution_results=merged,
        summary_overall=slices["overall_by_mode"],
        summary_slices=slices,
    )


def load_periodic_solution_benchmark(run_dir: Path | str) -> PeriodicSolutionBenchmarkRun:
    run_dir = Path(run_dir).expanduser()
    with (run_dir / "config.json").open("r", encoding="ascii") as handle:
        payload = json.load(handle)
    payload["output_base_dir"] = Path(payload["output_base_dir"])
    payload["mode_names"] = tuple(payload["mode_names"])
    config = PeriodicSolutionBenchmarkConfig(**payload)
    design = pd.read_parquet(run_dir / "trial_design.parquet")
    merged = pd.read_parquet(run_dir / "solution_results_with_design.parquet")
    slices = build_summary_slices(merged)
    return PeriodicSolutionBenchmarkRun(
        config=config,
        run_dir=run_dir,
        trial_design=design,
        solution_results=merged,
        summary_overall=slices["overall_by_mode"],
        summary_slices=slices,
    )


def plot_solution_metrics(summary: pd.DataFrame, *, ax: plt.Axes | None = None) -> plt.Axes:
    metrics = [
        "observable_recall",
        "phase_local_observable_recall",
        "precision_by_trial",
        "control_false_positive_rate",
        "off_target_detection_rate",
    ]
    plot_df = summary.set_index("mode")[metrics]
    if ax is None:
        _, ax = plt.subplots(figsize=FIG_TWO_COL_LC_WIDE)
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("Periodic solution recovery and contamination")
    ax.legend(loc="best", fontsize=8)
    ax.tick_params(axis="x", rotation=35)
    return ax


def plot_solution_heatmap(
    df: pd.DataFrame,
    *,
    metric: str,
    mode: str,
    row: str = "dip_amp_bin",
    col: str = "period_bin",
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    ok = df[(df["status"].eq("ok")) & (df["mode"].eq(mode))].copy()
    if metric in ok.columns and pd.api.types.is_bool_dtype(ok[metric]):
        values = ok.groupby([row, col], dropna=False)[metric].mean().unstack(col)
    else:
        values = ok.groupby([row, col], dropna=False)[metric].median().unstack(col)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize_from_legacy(9, 5))
    im = ax.imshow(values.to_numpy(dtype=float), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(values.shape[1]), labels=[str(x) for x in values.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(values.shape[0]), labels=[str(x) for x in values.index])
    ax.set_xlabel(col)
    ax.set_ylabel(row)
    ax.set_title(title or f"{metric}: {mode}")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(metric)
    return ax


def recompute_solution_trial(
    run: PeriodicSolutionBenchmarkRun,
    trial_id: int,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    row = run.trial_design.loc[run.trial_design["trial_id"] == int(trial_id)]
    if row.empty:
        raise KeyError(f"trial_id {trial_id} not found")
    row_series = row.iloc[0]
    df = simulate_periodic_lightcurve(row_series)
    selected_period = selected_period_for_mode(row_series, mode, run.config.seed)
    baseline_func, baseline_kwargs = baseline_for_mode(mode, selected_period, run.config)
    scored = score_lightcurve(
        df,
        baseline_func=baseline_func,
        baseline_kwargs=baseline_kwargs,
        filter_residual_bad_cameras_enabled=False,
        p_points=run.config.p_points,
        mag_points=run.config.mag_points,
        trigger_mode=run.config.trigger_mode,
        logbf_threshold_dip=run.config.logbf_threshold_dip,
        logbf_threshold_jump=run.config.logbf_threshold_jump,
        significance_threshold=run.config.significance_threshold,
        run_min_points=run.config.run_min_points,
        max_gap_points=run.config.run_max_gap_points,
        run_max_gap_days=run.config.run_max_gap_days,
        run_min_duration_days=run.config.run_min_duration_days,
        compute_event_prob=run.config.compute_event_prob,
    )
    return (
        scored["df"].reset_index(drop=True),
        scored["df_base"].reset_index(drop=True),
        {
            "mode": mode,
            "mode_label": SOLUTION_SPECS[mode]["label"],
            "selected_period_days": selected_period,
            "dip": scored["dip"],
            "jump": scored["jump"],
        },
    )


def _resolve_modes(
    run: PeriodicSolutionBenchmarkRun,
    modes: Sequence[str] | None,
) -> tuple[str, ...]:
    resolved = tuple(str(mode) for mode in (modes if modes is not None else run.config.mode_names))
    for mode in resolved:
        if mode not in SOLUTION_SPECS:
            raise ValueError(f"Unknown periodic solution mode: {mode}")
    return resolved


def recompute_solution_trial_baseline_modes(
    run: PeriodicSolutionBenchmarkRun,
    trial_id: int,
    *,
    modes: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Recompute one simulated trial with several baseline modes."""
    row = run.trial_design.loc[run.trial_design["trial_id"] == int(trial_id)]
    if row.empty:
        raise KeyError(f"trial_id {trial_id} not found")
    row_series = row.iloc[0]
    raw_df = simulate_periodic_lightcurve(row_series)
    clean_df: pd.DataFrame | None = None
    outputs: dict[str, dict[str, object]] = {}

    for mode in _resolve_modes(run, modes):
        selected_period = selected_period_for_mode(row_series, mode, run.config.seed)
        baseline_func, baseline_kwargs = baseline_for_mode(mode, selected_period, run.config)
        try:
            scored = score_lightcurve(
                raw_df,
                baseline_func=baseline_func,
                baseline_kwargs=baseline_kwargs,
                filter_residual_bad_cameras_enabled=False,
                p_points=run.config.p_points,
                mag_points=run.config.mag_points,
                trigger_mode=run.config.trigger_mode,
                logbf_threshold_dip=run.config.logbf_threshold_dip,
                logbf_threshold_jump=run.config.logbf_threshold_jump,
                significance_threshold=run.config.significance_threshold,
                run_min_points=run.config.run_min_points,
                max_gap_points=run.config.run_max_gap_points,
                run_max_gap_days=run.config.run_max_gap_days,
                run_min_duration_days=run.config.run_min_duration_days,
                compute_event_prob=run.config.compute_event_prob,
            )
            scored_df = scored["df"].reset_index(drop=True)
            if clean_df is None:
                clean_df = scored_df
            outputs[mode] = {
                "mode": mode,
                "mode_label": str(SOLUTION_SPECS[mode]["label"]),
                "baseline_strategy": str(SOLUTION_SPECS[mode]["baseline"]),
                "selected_period_days": selected_period,
                "df_base": scored["df_base"].reset_index(drop=True),
                "dip": scored["dip"],
                "jump": scored["jump"],
                "status": "ok",
                "error": "",
            }
        except Exception as exc:
            outputs[mode] = {
                "mode": mode,
                "mode_label": str(SOLUTION_SPECS[mode]["label"]),
                "baseline_strategy": str(SOLUTION_SPECS[mode]["baseline"]),
                "selected_period_days": selected_period,
                "status": "error",
                "error": str(exc),
            }

    if clean_df is None:
        details = "; ".join(f"{mode}: {info.get('error', '')}" for mode, info in outputs.items())
        raise RuntimeError(f"All baseline modes failed for trial_id {trial_id}: {details}")
    return clean_df, outputs


def _safe_metric_label(results: pd.DataFrame, trial_id: int, mode: str) -> str:
    if results.empty or not {"trial_id", "mode"}.issubset(results.columns):
        return ""
    row = results[(results["trial_id"].eq(int(trial_id))) & (results["mode"].eq(mode))]
    if row.empty:
        return ""
    item = row.iloc[0]
    parts: list[str] = []
    for column, label, fmt in (
        ("baseline_mae_outside_dip", "MAE", ".3f"),
        ("resid_mad_outside_dip", "resid MAD", ".3f"),
        ("amp_recovery_ratio", "amp rec", ".2f"),
    ):
        value = item.get(column, np.nan)
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = np.nan
        if np.isfinite(number):
            parts.append(f"{label}={number:{fmt}}")
    return " | ".join(parts)


def plot_solution_baseline_mode_grid(
    run: PeriodicSolutionBenchmarkRun,
    trial_id: int,
    *,
    modes: Sequence[str] | None = None,
    ax: np.ndarray | None = None,
    mark_truth: bool = True,
    mark_detected: bool = True,
    show_true_baseline: bool = True,
) -> np.ndarray:
    """Plot observed baseline fits and baseline-subtracted residuals by mode."""
    mode_names = _resolve_modes(run, modes)
    n_modes = len(mode_names)
    if n_modes == 0:
        raise ValueError("At least one mode is required")

    if ax is None:
        _, ax = plt.subplots(
            n_modes,
            2,
            figsize=figsize_two_col_grid(2, n_modes, row_height=max(3.0, 5.5 / n_modes)),
            sharex="col",
            squeeze=False,
        )
    axes = np.asarray(ax, dtype=object)
    if axes.ndim == 1:
        axes = axes.reshape(n_modes, 2)
    if axes.shape[0] < n_modes or axes.shape[1] < 2:
        raise ValueError(f"Expected axes with shape at least ({n_modes}, 2)")

    df, mode_outputs = recompute_solution_trial_baseline_modes(run, trial_id, modes=mode_names)
    jd = pd.to_numeric(df["JD"], errors="coerce").to_numpy(dtype=float)
    truth_mask = (
        df["truth_dip_mask"].to_numpy(dtype=bool)
        if "truth_dip_mask" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    true_baseline = (
        pd.to_numeric(df["true_baseline_mag"], errors="coerce").to_numpy(dtype=float)
        if "true_baseline_mag" in df.columns
        else np.full(len(df), np.nan, dtype=float)
    )

    for row_idx, mode in enumerate(mode_names):
        ax_lc = axes[row_idx, 0]
        ax_resid = axes[row_idx, 1]
        info = mode_outputs[mode]
        mode_label = str(info.get("mode_label", mode))
        metric_label = _safe_metric_label(run.solution_results, trial_id, mode)
        selected_period = info.get("selected_period_days", np.nan)
        try:
            period_label = f"P={float(selected_period):.4g} d" if np.isfinite(float(selected_period)) else "P=n/a"
        except (TypeError, ValueError):
            period_label = "P=n/a"

        if info.get("status") != "ok":
            message = f"{mode_label}\nfailed: {info.get('error', '')}"
            for axis in (ax_lc, ax_resid):
                axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center", va="center", fontsize=9)
                axis.set_axis_off()
            continue

        df_base = info["df_base"]
        if not isinstance(df_base, pd.DataFrame):
            raise TypeError(f"Mode {mode} did not return a DataFrame baseline")
        baseline = pd.to_numeric(df_base["baseline"], errors="coerce").to_numpy(dtype=float)
        resid = pd.to_numeric(df_base["resid"], errors="coerce").to_numpy(dtype=float)
        if len(baseline) != len(df) or len(resid) != len(df):
            raise ValueError(f"Mode {mode} returned a baseline with length {len(baseline)} for {len(df)} points")

        baseline_overlay = pd.DataFrame(
            {
                "JD": jd,
                "baseline": baseline,
                "camera": df["camera#"].astype(str) if "camera#" in df.columns else "all",
            }
        )
        plot_lightcurve_panel(
            ax_lc,
            df,
            group_by="camera",
            camera_col="camera#",
            show_errorbars=False,
            marker_size=2.7,
            legend="none",
            time_offset="none",
            xlabel="JD" if row_idx == n_modes - 1 else "",
            ylabel="mag",
            baseline=baseline_overlay,
            baseline_col="baseline",
            baseline_time_col="JD",
            baseline_group_col="camera",
            baseline_label="fit baseline",
            baseline_style={"color": "crimson", "linewidth": 1.15, "alpha": 0.9},
        )
        if show_true_baseline and np.isfinite(true_baseline).any():
            order = np.argsort(jd)
            ax_lc.plot(jd[order], true_baseline[order], color="black", lw=0.95, alpha=0.8, label="true baseline")

        residual_plot = df.copy()
        residual_plot.loc[:, "resid"] = resid
        plot_residual_panel(
            ax_resid,
            residual_plot,
            group_by="camera",
            camera_col="camera#",
            show_errorbars=False,
            marker_size=2.7,
            legend="none",
            time_offset="none",
            xlabel="JD" if row_idx == n_modes - 1 else "",
            ylabel="mag - baseline",
            invert_y=False,
        )
        ax_resid.axhline(0.0, color="0.2", linestyle="--", linewidth=0.8, alpha=0.65)

        dip = info.get("dip", {})
        event_idx = np.asarray(dip.get("event_indices", []) if isinstance(dip, dict) else [], dtype=int)
        event_idx = event_idx[(event_idx >= 0) & (event_idx < len(df))]
        if mark_truth and truth_mask.any():
            ax_lc.scatter(
                jd[truth_mask],
                df.loc[truth_mask, "mag"],
                s=26,
                facecolors="none",
                edgecolors="limegreen",
                linewidths=0.9,
                label="truth dip support",
                zorder=8,
            )
            ax_resid.scatter(
                jd[truth_mask],
                resid[truth_mask],
                s=26,
                facecolors="none",
                edgecolors="limegreen",
                linewidths=0.9,
                zorder=8,
            )
        if mark_detected and event_idx.size:
            ax_lc.scatter(
                jd[event_idx],
                df.loc[event_idx, "mag"],
                marker="x",
                s=36,
                color="gold",
                linewidths=1.0,
                label="detected event points",
                zorder=9,
            )
            ax_resid.scatter(
                jd[event_idx],
                resid[event_idx],
                marker="x",
                s=36,
                color="gold",
                linewidths=1.0,
                zorder=9,
            )

        subtitle = f"{mode_label} | {period_label}"
        if metric_label:
            subtitle = f"{subtitle} | {metric_label}"
        ax_lc.set_title(f"{subtitle}\nobserved light curve + baseline", fontsize=9)
        ax_resid.set_title("baseline-subtracted residuals", fontsize=9)
        if row_idx != n_modes - 1:
            ax_lc.set_xlabel("")
            ax_resid.set_xlabel("")
        if row_idx == 0:
            handles, labels = ax_lc.get_legend_handles_labels()
            if handles:
                ax_lc.legend(handles, labels, ncol=min(4, len(labels)), fontsize=7, frameon=False, loc="best")

    return axes


def select_baseline_gallery_trials(
    df: pd.DataFrame,
    *,
    reference_mode: str | None = None,
    n_trials: int = 24,
    seed: int = 0,
) -> list[int]:
    """Select a diverse set of trial IDs for baseline-gallery figures."""
    ok = df[df["status"].eq("ok")].copy()
    if ok.empty:
        return []
    if reference_mode is None:
        reference_mode = str(ok["mode"].iloc[0])
    sub = ok[ok["mode"].eq(reference_mode)].copy()
    if sub.empty:
        sub = ok.drop_duplicates("trial_id").copy()

    n_trials = max(int(n_trials), 0)
    if n_trials == 0:
        return []
    selected: list[int] = []

    def add_rows(rows: pd.DataFrame, *, sort_col: str | None = None, ascending: bool = False, limit: int = 3) -> None:
        added = 0
        if rows.empty or len(selected) >= n_trials:
            return
        pool = rows.copy()
        if sort_col is not None and sort_col in pool.columns:
            pool = pool.sort_values(sort_col, ascending=ascending)
        for trial_id in pool["trial_id"].astype(int):
            if trial_id not in selected:
                selected.append(int(trial_id))
                added += 1
            if len(selected) >= n_trials or added >= limit:
                break

    per_bucket = max(1, int(np.ceil(n_trials / 8.0)))
    has_dip = sub["has_dip"].fillna(False).astype(bool)
    observable = sub["truth_observable_actual"].fillna(False).astype(bool) if "truth_observable_actual" in sub.columns else has_dip
    recovered = sub["target_recovered"].fillna(False).astype(bool) if "target_recovered" in sub.columns else pd.Series(False, index=sub.index)
    off_target = sub["off_target_detection"].fillna(False).astype(bool) if "off_target_detection" in sub.columns else pd.Series(False, index=sub.index)
    false_positive = sub["false_positive"].fillna(False).astype(bool) if "false_positive" in sub.columns else pd.Series(False, index=sub.index)
    phase_local = sub["phase_local_detected"].fillna(False).astype(bool) if "phase_local_detected" in sub.columns else pd.Series(False, index=sub.index)

    add_rows(sub[has_dip & observable & recovered], sort_col="dip_bayes_factor", ascending=False, limit=per_bucket)
    add_rows(sub[has_dip & observable & ~recovered], sort_col="dip_amp_mag", ascending=False, limit=per_bucket)
    add_rows(sub[off_target], sort_col="dip_bayes_factor", ascending=False, limit=per_bucket)
    add_rows(sub[~has_dip & false_positive], sort_col="dip_bayes_factor", ascending=False, limit=per_bucket)
    add_rows(sub[phase_local & ~recovered], sort_col="phase_local_truth_peak_snr", ascending=False, limit=per_bucket)

    if len(selected) < n_trials:
        rng = np.random.default_rng(int(seed))
        group_cols = [col for col in ("waveform_kind", "dip_amp_bin", "period_bin") if col in sub.columns]
        if group_cols:
            shuffled = sub.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
            for _, group in shuffled.groupby(group_cols, dropna=False):
                if len(selected) >= n_trials:
                    break
                trial_id = int(group["trial_id"].iloc[0])
                if trial_id not in selected:
                    selected.append(trial_id)

    if len(selected) < n_trials:
        remaining = sub[~sub["trial_id"].astype(int).isin(selected)]
        if not remaining.empty:
            remaining = remaining.sample(
                n=min(len(remaining), n_trials - len(selected)),
                random_state=int(seed),
            )
            selected.extend([int(x) for x in remaining["trial_id"]])
    return selected[:n_trials]


def write_solution_baseline_gallery(
    run: PeriodicSolutionBenchmarkRun,
    *,
    trial_ids: Sequence[int] | None = None,
    modes: Sequence[str] | None = None,
    n_trials: int = 24,
    output_dir: Path | str | None = None,
    dpi: int = 150,
    close: bool = True,
    show_progress: bool | None = None,
) -> list[Path]:
    """Write many baseline/residual comparison figures for one benchmark run."""
    mode_names = _resolve_modes(run, modes)
    if not mode_names:
        raise ValueError("At least one mode is required")
    if trial_ids is None:
        trial_ids = select_baseline_gallery_trials(
            run.solution_results,
            reference_mode=mode_names[0] if mode_names else None,
            n_trials=n_trials,
            seed=run.config.seed,
        )
    out_dir = Path(output_dir) if output_dir is not None else run.run_dir / "plots" / "baseline_mode_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    iterator: Any = [int(trial_id) for trial_id in trial_ids]
    should_show_progress = bool(show_progress) if show_progress is not None else bool(run.config.show_progress)
    if should_show_progress:
        iterator = tqdm(iterator, desc="Baseline gallery figures")
    for trial_id in iterator:
        fig, axes = plt.subplots(
            len(mode_names),
            2,
            figsize=figsize_two_col_grid(2, len(mode_names), row_height=max(3.0, 5.5 / len(mode_names))),
            sharex="col",
            squeeze=False,
        )
        plot_solution_baseline_mode_grid(run, int(trial_id), modes=mode_names, ax=axes)
        fig.suptitle(f"Trial {int(trial_id)} baseline solutions by mode", y=0.995)
        fig.tight_layout()
        path = out_dir / f"trial_{int(trial_id):05d}_baseline_modes.png"
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        if close:
            plt.close(fig)
        paths.append(path)
    return paths


def plot_solution_trial_diagnostic(
    run: PeriodicSolutionBenchmarkRun,
    trial_id: int,
    *,
    mode: str,
    ax: np.ndarray | None = None,
) -> np.ndarray:
    df, df_base, info = recompute_solution_trial(run, trial_id, mode)
    if ax is None:
        _, ax = plt.subplots(3, 1, figsize=figsize_from_legacy(13, 10), sharex=False)
    jd = df["JD"].to_numpy(dtype=float)
    truth_mask = df["truth_dip_mask"].to_numpy(dtype=bool)
    event_idx = np.asarray(info["dip"].get("event_indices", []), dtype=int)
    event_idx = event_idx[(event_idx >= 0) & (event_idx < len(df))]

    baseline_overlay = pd.DataFrame({"JD": jd, "baseline": df_base["baseline"].to_numpy(dtype=float)})
    plot_lightcurve_panel(
        ax[0],
        df,
        group_by="camera",
        camera_col="camera#",
        show_errorbars=False,
        marker_size=3.4,
        legend="none",
        time_offset="none",
        xlabel="JD",
        ylabel="mag",
        baseline=baseline_overlay,
        baseline_col="baseline",
        baseline_time_col="JD",
        baseline_label=str(info["mode_label"]),
        baseline_style={"color": "crimson", "linewidth": 1.1},
    )
    ax[0].plot(jd, df["true_baseline_mag"], color="black", lw=1.0, label="true baseline")
    if truth_mask.any():
        ax[0].scatter(jd[truth_mask], df.loc[truth_mask, "mag"], s=28, facecolors="none", edgecolors="limegreen", label="truth dip support")
    if event_idx.size:
        ax[0].scatter(jd[event_idx], df.loc[event_idx, "mag"], marker="x", s=42, color="gold", label="detected event points")
    ax[0].set_title(f"Trial {trial_id}: observed light curve and baseline")
    ax[0].legend(ncol=4, fontsize=8)

    residual_plot = pd.DataFrame({"JD": jd, "resid": df_base["resid"].to_numpy(dtype=float)})
    plot_residual_panel(
        ax[1],
        residual_plot,
        group_by="none",
        show_errorbars=False,
        marker_size=3.4,
        legend="none",
        time_offset="none",
        xlabel="JD",
        ylabel="mag - baseline",
        invert_y=False,
    )
    if truth_mask.any():
        ax[1].scatter(jd[truth_mask], df_base.loc[truth_mask, "resid"], s=28, facecolors="none", edgecolors="limegreen")
    if event_idx.size:
        ax[1].scatter(jd[event_idx], df_base.loc[event_idx, "resid"], marker="x", s=42, color="gold")
    ax[1].set_title("Residuals used by event scoring")

    period = float(df["period_days"].iloc[0])
    phase = np.mod((jd - np.nanmin(jd)) / period, 1.0)
    plot_phase_panel(
        ax[2],
        df,
        period_days=period,
        epoch_jd=float(np.nanmin(jd)),
        group_by="none",
        show_errorbars=False,
        marker_size=3.4,
        legend="none",
    )
    order = np.argsort(phase)
    baseline = df_base["baseline"].to_numpy(dtype=float)
    ax[2].plot(phase[order], baseline[order], color="crimson", lw=1.0)
    ax[2].plot(phase[order] + 1.0, baseline[order], color="crimson", lw=1.0, alpha=0.7)
    ax[2].set_title("Folded view")
    return ax


def select_example_trials(df: pd.DataFrame, mode: str) -> dict[str, int | None]:
    sub = df[(df["mode"] == mode) & df["status"].eq("ok")].copy()
    examples: dict[str, int | None] = {}
    hit = sub[sub["target_recovered"].fillna(False).astype(bool)]
    miss = sub[
        sub["has_dip"].fillna(False).astype(bool)
        & sub["truth_observable_actual"].fillna(False).astype(bool)
        & ~sub["target_recovered"].fillna(False).astype(bool)
    ]
    off_target = sub[sub["off_target_detection"].fillna(False).astype(bool)]
    control_fp = sub[sub["false_positive"].fillna(False).astype(bool)]
    local_only = sub[
        sub["phase_local_detected"].fillna(False).astype(bool)
        & ~sub["target_recovered"].fillna(False).astype(bool)
    ]

    examples["strong_hit"] = int(hit.sort_values("dip_bayes_factor", ascending=False)["trial_id"].iloc[0]) if not hit.empty else None
    examples["observable_miss"] = int(miss.sort_values("dip_amp_mag", ascending=False)["trial_id"].iloc[0]) if not miss.empty else None
    examples["off_target_detection"] = int(off_target.sort_values("dip_bayes_factor", ascending=False)["trial_id"].iloc[0]) if not off_target.empty else None
    examples["control_false_positive"] = int(control_fp.sort_values("dip_bayes_factor", ascending=False)["trial_id"].iloc[0]) if not control_fp.empty else None
    examples["phase_local_only"] = int(local_only.sort_values("phase_local_truth_peak_snr", ascending=False)["trial_id"].iloc[0]) if not local_only.empty else None
    return examples
