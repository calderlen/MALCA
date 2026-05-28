from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from itertools import product
import os
from pathlib import Path
from typing import Any

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

import malca.stv.periodicity_gate as pregate
from malca.baseline import (
    global_median_baseline,
    per_camera_gp_baseline,
    per_camera_gp_baseline_masked,
    per_camera_median_baseline,
    phase_template_baseline,
)
from malca.config import (
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    BASELINE_FUNC,
    BASELINE_JITTER,
    BASELINE_Q,
    BASELINE_S0,
    BASELINE_W0,
    CLEAN_LC_MAX_ERROR_ABSOLUTE,
    CLEAN_LC_MAX_ERROR_SIGMA,
    LIGHT_CURVE_FILE_EXTENSION,
    LOGBF_THRESHOLD_DIP,
    LOGBF_THRESHOLD_JUMP,
    MAG_POINTS,
    MIN_MAG_OFFSET,
    PARQUET_OUTPUT_COMPRESSION,
    P_POINTS,
    PRE_PERIODICITY_CE_SNR_RESCUE_MARGIN,
    PRE_PERIODICITY_CE_SNR_THRESHOLD,
    PRE_PERIODICITY_MAX_PERIOD,
    PRE_PERIODICITY_MIN_POINTS,
    PRE_PERIODICITY_MIN_PERIOD,
    PRE_PERIODICITY_N_PERIODS,
    PRE_PERIODICITY_PHASE_PEAK_REGION_MAX,
    PRE_PERIODICITY_PHASE_PEAK_SNR_MIN,
    PRE_PERIODICITY_PHASE_PEAK_WIDTH_MAX,
    PRE_PERIODICITY_SCATTER_RATIO_MAX,
    PRE_PERIODICITY_SCATTER_RATIO_RESCUE_MARGIN,
    RUN_MAX_GAP_POINTS,
    RUN_MIN_POINTS,
    SIGNIFICANCE_THRESHOLD,
    TRIGGER_MODE,
)
from malca.stv.events import score_lightcurve
from malca.stv.filter import apply_filters
from malca.lightcurve_io import load_lightcurve_df
from malca.lightcurve_publication import (
    plot_lightcurve_panel,
    plot_phase_panel,
    plot_residual_panel,
    publication_style_context,
    style_publication_axis,
)
from malca.phase import phase_fold_dataframe
from malca.stv.score import compute_event_score
from malca.stats import compute_ce_stats
from malca.stv.triggering import posterior_probability_threshold
from malca.utils import clean_lc, compute_n_cameras, filter_bad_cameras


BENCHMARK_CLASS_ORDER: tuple[str, ...] = (
    "no_injection_control",
    "shuffled_magnitude_control",
    "synthetic_gaussian_noise_control",
    "single_non_recurrent_dip",
    "non_periodic_recurrent_dips",
    "semi_periodic_dips",
    "periodic_fixed_depth_dips",
    "periodic_varying_depth_dips",
    "broad_long_dips",
    "smooth_periodic_contaminant",
)

TARGET_DIP_BY_CLASS: dict[str, bool] = {
    "no_injection_control": False,
    "shuffled_magnitude_control": False,
    "synthetic_gaussian_noise_control": False,
    "single_non_recurrent_dip": True,
    "non_periodic_recurrent_dips": True,
    "semi_periodic_dips": True,
    "periodic_fixed_depth_dips": True,
    "periodic_varying_depth_dips": True,
    "broad_long_dips": True,
    "smooth_periodic_contaminant": False,
}

TARGET_GATE_LABEL_BY_CLASS: dict[str, str] = {
    "no_injection_control": "non_periodic",
    "shuffled_magnitude_control": "non_periodic",
    "synthetic_gaussian_noise_control": "non_periodic",
    "single_non_recurrent_dip": "non_periodic",
    "non_periodic_recurrent_dips": "non_periodic",
    "semi_periodic_dips": "ambiguous",
    "periodic_fixed_depth_dips": "periodic",
    "periodic_varying_depth_dips": "periodic",
    "broad_long_dips": "non_periodic",
    "smooth_periodic_contaminant": "periodic",
}

THRESHOLD_CE_SNR_VALUES: tuple[float, ...] = (6.0, 8.0, 10.0, 12.0, 14.0)
THRESHOLD_SCATTER_RATIO_VALUES: tuple[float, ...] = (0.65, 0.75, 0.8, 0.9, 1.0)
THRESHOLD_PHASE_PEAK_SNR_VALUES: tuple[float, ...] = (2.0, 2.5, 3.0)

PERIOD_SEARCH_N_PERIODS_VALUES: tuple[int, ...] = (2000, 5000)
PERIOD_SEARCH_MIN_PERIOD_VALUES: tuple[float, ...] = (0.2, 0.5)
PERIOD_SEARCH_MAX_PERIOD_VALUES: tuple[float, ...] = (50.0, 100.0, 200.0)

PIPELINE_DETECTED_COLUMNS: dict[str, str] = {
    "standard_only": "standard_detected",
    "phase_folded_only": "phase_folded_detected",
    "bifurcated_gate": "bifurcated_detected",
}

POST_FILTER_FIELD_COLUMNS: tuple[str, ...] = (
    "dip_significant",
    "jump_significant",
    "dip_count",
    "jump_count",
    "dip_run_count",
    "jump_run_count",
    "dip_max_run_points",
    "jump_max_run_points",
    "dip_max_run_cameras",
    "jump_max_run_cameras",
    "dip_max_log_bf_local",
    "jump_max_log_bf_local",
    "dip_bayes_factor",
    "jump_bayes_factor",
    "baseline_mag",
    "dip_best_mag_event",
    "jump_best_mag_event",
    "dip_best_morph",
    "jump_best_morph",
    "dip_best_delta_bic",
    "jump_best_delta_bic",
    "dipper_score",
    "jumper_score",
)

PERIOD_MATCH_HARMONIC_FACTORS: tuple[float, ...] = (
    1.0,
    0.5,
    2.0,
    1.0 / 3.0,
    3.0,
    0.25,
    4.0,
)


@dataclass
class BenchmarkConfig:
    bundle_lc_dir: Path = Path("output/runs/runs_march18_bundle_all/bundle_assets/lightcurves")
    lightcurve_file_ext: str | None = None
    output_base_dir: Path = Path("output/diagnostics/periodicity_gate_injection_benchmark")
    run_tag: str | None = None
    smoke_mode: bool = False
    total_trials: int = 6000
    smoke_total_trials: int = 512
    control_sample_size: int = 205
    smoke_control_sample_size: int = 96
    seed: int = 20260513
    cache_generated_lightcurves: bool = True
    baseline_func: str = BASELINE_FUNC
    min_mag_offset: float = MIN_MAG_OFFSET
    p_points: int = P_POINTS
    mag_points: int = MAG_POINTS
    trigger_mode: str = TRIGGER_MODE
    logbf_threshold_dip: float = LOGBF_THRESHOLD_DIP
    logbf_threshold_jump: float = LOGBF_THRESHOLD_JUMP
    significance_threshold: float = SIGNIFICANCE_THRESHOLD
    run_min_points: int = RUN_MIN_POINTS
    run_max_gap_points: int = RUN_MAX_GAP_POINTS
    run_max_gap_days: float | None = None
    run_min_duration_days: float = 0.0
    compute_event_prob: bool = True
    gate_min_period: float = PRE_PERIODICITY_MIN_PERIOD
    gate_max_period: float = PRE_PERIODICITY_MAX_PERIOD
    # The production pregate default is PRE_PERIODICITY_N_PERIODS=5000. That is
    # too expensive for a multi-thousand-trial injection grid, so the benchmark defaults to a
    # coarse scan and reserves 5000-period comparisons for the subset sweep.
    gate_n_periods: int = 500
    smoke_gate_n_periods: int = 800
    gate_ce_snr_threshold: float = PRE_PERIODICITY_CE_SNR_THRESHOLD
    gate_min_points: int = PRE_PERIODICITY_MIN_POINTS
    gate_scatter_ratio_max: float = PRE_PERIODICITY_SCATTER_RATIO_MAX
    period_search_subset_size: int = 100
    smoke_period_search_subset_size: int = 64
    checkpoint_every: int = 1000
    workers: int = 10
    trial_task_size: int = 1
    show_progress: bool = True
    post_filter_apply_periodic_catalog_validation: bool = False
    post_filter_apply_gaia_ruwe_validation: bool = False
    post_filter_apply_gaia_pm_validation: bool = False
    post_filter_apply_periodicity_validation: bool = False

    @property
    def effective_total_trials(self) -> int:
        return int(self.smoke_total_trials if self.smoke_mode else self.total_trials)

    @property
    def effective_control_sample_size(self) -> int:
        return int(self.smoke_control_sample_size if self.smoke_mode else self.control_sample_size)

    @property
    def effective_gate_n_periods(self) -> int:
        return int(self.smoke_gate_n_periods if self.smoke_mode else self.gate_n_periods)

    @property
    def effective_period_search_subset_size(self) -> int:
        return int(self.smoke_period_search_subset_size if self.smoke_mode else self.period_search_subset_size)


@dataclass
class BenchmarkRun:
    config: BenchmarkConfig
    run_dir: Path
    control_table: pd.DataFrame
    trial_design: pd.DataFrame
    trial_results: pd.DataFrame
    gate_threshold_sweep: pd.DataFrame
    period_search_subset_sweep: pd.DataFrame
    summary_metrics: pd.DataFrame
    post_filter_results: pd.DataFrame
    post_filter_rejection_summary: pd.DataFrame
    generated_lightcurves: dict[int, pd.DataFrame]


def make_run_dir(config: BenchmarkConfig) -> Path:
    tag = config.run_tag
    if not tag:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"{timestamp}_smoke" if config.smoke_mode else timestamp
    run_dir = Path(config.output_base_dir).expanduser() / str(tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _control_file_extensions(file_ext: str | None = None) -> tuple[str, ...]:
    if file_ext:
        return (str(file_ext).lstrip("."),)

    extensions: list[str] = []
    for ext in (LIGHT_CURVE_FILE_EXTENSION, "dat3", "dat2", "dat"):
        ext_normalized = str(ext).lstrip(".")
        if ext_normalized and ext_normalized not in extensions:
            extensions.append(ext_normalized)
    return tuple(extensions)


def discover_control_paths(
    bundle_lc_dir: Path | str,
    *,
    file_ext: str | None = None,
    limit: int | None = None,
) -> list[Path]:
    root = Path(bundle_lc_dir).expanduser()
    paths: list[Path] = []
    for ext in _control_file_extensions(file_ext):
        paths = sorted(root.glob(f"*.{ext}"))
        if paths:
            break
    if limit is not None:
        paths = paths[: int(limit)]
    return paths


def load_control_lightcurves(
    bundle_lc_dir: Path | str,
    *,
    sample_size: int,
    seed: int,
    min_points: int = PRE_PERIODICITY_MIN_POINTS,
    mag_lo: float = 12.0,
    mag_hi: float = 15.0,
    file_ext: str | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    paths = discover_control_paths(bundle_lc_dir, file_ext=file_ext)
    if not paths:
        extensions = ", ".join(f"*.{ext}" for ext in _control_file_extensions(file_ext))
        raise FileNotFoundError(f"No light curves ({extensions}) found in {bundle_lc_dir}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    rows: list[dict[str, Any]] = []
    controls: dict[str, pd.DataFrame] = {}
    iterator = tqdm(order, desc="Loading controls", disable=not show_progress)
    for idx in iterator:
        path = paths[int(idx)]
        try:
            df = load_lightcurve_df(path)
        except Exception:
            continue
        if df.empty or len(df) < int(min_points):
            continue
        mag = pd.to_numeric(df["mag"], errors="coerce").to_numpy(dtype=float)
        finite_mag = mag[np.isfinite(mag)]
        if finite_mag.size < int(min_points):
            continue
        median_mag = float(np.median(finite_mag))
        if median_mag < float(mag_lo) or median_mag > float(mag_hi):
            continue
        jd = pd.to_numeric(df["JD"], errors="coerce").to_numpy(dtype=float)
        jd = jd[np.isfinite(jd)]
        if jd.size < 2:
            continue
        source_id = path.stem
        path_text = str(path)
        controls[path_text] = df.reset_index(drop=True)
        rows.append(
            {
                "source_id": source_id,
                "source_path": path_text,
                "n_points": int(len(df)),
                "median_mag": median_mag,
                "jd_min": float(np.min(jd)),
                "jd_max": float(np.max(jd)),
                "jd_span": float(np.max(jd) - np.min(jd)),
                "n_cameras": int(compute_n_cameras(df)) if "camera#" in df.columns else np.nan,
            }
        )
        if len(rows) >= int(sample_size):
            break

    if not rows:
        raise RuntimeError("Could not load any usable control light curves")
    return pd.DataFrame(rows), controls


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(10.0 ** rng.uniform(np.log10(float(lo)), np.log10(float(hi))))


def _balanced_classes(total_trials: int) -> list[str]:
    total = int(total_trials)
    if total < len(BENCHMARK_CLASS_ORDER):
        raise ValueError(f"Need at least {len(BENCHMARK_CLASS_ORDER)} trials for balanced classes")
    base = total // len(BENCHMARK_CLASS_ORDER)
    rem = total % len(BENCHMARK_CLASS_ORDER)
    classes: list[str] = []
    for idx, class_name in enumerate(BENCHMARK_CLASS_ORDER):
        classes.extend([class_name] * (base + (1 if idx < rem else 0)))
    return classes


def build_trial_design(control_table: pd.DataFrame, config: BenchmarkConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    classes = _balanced_classes(config.effective_total_trials)
    rng.shuffle(classes)
    control_idx = rng.integers(0, len(control_table), size=len(classes))

    rows: list[dict[str, Any]] = []
    for trial_id, (class_name, cidx) in enumerate(zip(classes, control_idx, strict=True)):
        control = control_table.iloc[int(cidx)]
        jd_min = float(control["jd_min"])
        jd_max = float(control["jd_max"])
        span = max(float(control["jd_span"]), 1.0)
        trial_seed = int(config.seed + 1009 * (trial_id + 1))
        trng = np.random.default_rng(trial_seed)

        amplitude = 0.0
        duration = 0.0
        period_days = np.nan
        period_jitter_frac = 0.0
        depth_scatter = 0.0
        event_count = 0
        phase0 = float(trng.uniform(0.0, 1.0))
        t0 = float(trng.uniform(jd_min + 0.1 * span, jd_max - 0.1 * span))

        if class_name == "single_non_recurrent_dip":
            amplitude = float(trng.uniform(0.08, 0.9))
            duration = min(_log_uniform(trng, 1.5, 30.0), 0.25 * span)
            event_count = 1
        elif class_name == "non_periodic_recurrent_dips":
            amplitude = float(trng.uniform(0.06, 0.8))
            duration = min(_log_uniform(trng, 1.5, 25.0), 0.20 * span)
            event_count = int(trng.integers(2, 7))
        elif class_name == "semi_periodic_dips":
            period_days = min(_log_uniform(trng, 3.0, 80.0), 0.45 * span)
            amplitude = float(trng.uniform(0.06, 0.8))
            duration = min(_log_uniform(trng, 1.5, 20.0), 0.25 * period_days)
            period_jitter_frac = float(trng.uniform(0.08, 0.25))
            depth_scatter = float(trng.uniform(0.10, 0.45))
            event_count = -1
        elif class_name == "periodic_fixed_depth_dips":
            period_days = min(_log_uniform(trng, 2.0, 80.0), 0.45 * span)
            amplitude = float(trng.uniform(0.05, 0.7))
            duration = min(_log_uniform(trng, 1.0, 18.0), 0.25 * period_days)
            event_count = -1
        elif class_name == "periodic_varying_depth_dips":
            period_days = min(_log_uniform(trng, 2.0, 80.0), 0.45 * span)
            amplitude = float(trng.uniform(0.05, 0.75))
            duration = min(_log_uniform(trng, 1.0, 18.0), 0.25 * period_days)
            depth_scatter = float(trng.uniform(0.20, 0.70))
            event_count = -1
        elif class_name == "broad_long_dips":
            amplitude = float(trng.uniform(0.08, 1.2))
            duration = min(_log_uniform(trng, 40.0, 220.0), 0.35 * span)
            event_count = 1
        elif class_name == "smooth_periodic_contaminant":
            period_days = min(_log_uniform(trng, 2.0, 80.0), 0.45 * span)
            amplitude = float(trng.uniform(0.03, 0.35))
            event_count = 0

        rows.append(
            {
                "trial_id": int(trial_id),
                "class_name": class_name,
                "source_id": str(control["source_id"]),
                "source_path": str(control["source_path"]),
                "trial_seed": trial_seed,
                "target_dip": bool(TARGET_DIP_BY_CLASS[class_name]),
                "target_gate_label": TARGET_GATE_LABEL_BY_CLASS[class_name],
                "amplitude": float(amplitude),
                "duration": float(duration),
                "period_days": float(period_days),
                "period_jitter_frac": float(period_jitter_frac),
                "depth_scatter": float(depth_scatter),
                "requested_event_count": int(event_count),
                "phase0": float(phase0),
                "t0": float(t0),
                "control_median_mag": float(control["median_mag"]),
                "control_n_points": int(control["n_points"]),
                "control_jd_span": float(control["jd_span"]),
            }
        )
    return pd.DataFrame(rows)


def _gaussian_dip_profile(t: np.ndarray, center: float, duration: float, depth: float) -> np.ndarray:
    sigma = max(float(duration) / 2.355, 0.05)
    return float(depth) * np.exp(-0.5 * ((t - float(center)) / sigma) ** 2)


def _periodic_centers(
    jd: np.ndarray,
    *,
    period_days: float,
    phase0: float,
    duration: float,
    jitter_frac: float,
    rng: np.random.Generator,
) -> list[float]:
    finite_jd = jd[np.isfinite(jd)]
    if finite_jd.size == 0 or not np.isfinite(period_days) or period_days <= 0:
        return []
    jd_min = float(np.min(finite_jd))
    jd_max = float(np.max(finite_jd))
    period = float(period_days)
    center = jd_min + float(phase0) * period
    while center > jd_min - period:
        center -= period
    centers: list[float] = []
    while center <= jd_max + period:
        jitter = float(rng.normal(0.0, float(jitter_frac) * period)) if jitter_frac else 0.0
        value = center + jitter
        if jd_min - duration <= value <= jd_max + duration:
            centers.append(float(value))
        center += period
    return centers


def _shuffle_magnitudes_within_camera(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    mag = pd.to_numeric(out["mag"], errors="coerce").to_numpy(dtype=float)
    shuffled = mag.copy()
    if "camera#" in out.columns:
        camera_values = out["camera#"].fillna("__missing__").astype(str).to_numpy()
        for camera in np.unique(camera_values):
            idx = np.flatnonzero(camera_values == camera)
            finite_idx = idx[np.isfinite(mag[idx])]
            if finite_idx.size > 1:
                shuffled[finite_idx] = mag[rng.permutation(finite_idx)]
    else:
        finite_idx = np.flatnonzero(np.isfinite(mag))
        if finite_idx.size > 1:
            shuffled[finite_idx] = mag[rng.permutation(finite_idx)]
    out["mag"] = shuffled
    return out


def _replace_with_gaussian_noise_control(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    mag = pd.to_numeric(out["mag"], errors="coerce")
    err = pd.to_numeric(out["error"], errors="coerce").to_numpy(dtype=float)
    finite_err = err[np.isfinite(err) & (err > 0)]
    fallback_err = float(np.nanmedian(finite_err)) if finite_err.size else 0.05
    err = np.where(np.isfinite(err) & (err > 0), err, fallback_err)

    if "camera#" in out.columns:
        camera_values = out["camera#"].fillna("__missing__").astype(str)
        centers = mag.groupby(camera_values).transform("median")
        global_center = float(np.nanmedian(mag.to_numpy(dtype=float)))
        centers = centers.fillna(global_center)
    else:
        centers = pd.Series(float(np.nanmedian(mag.to_numpy(dtype=float))), index=out.index)

    out["mag"] = centers.to_numpy(dtype=float) + rng.normal(0.0, err, size=len(out))
    return out


def generate_trial_lightcurve(base_df: pd.DataFrame, trial: pd.Series | dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    row = dict(trial)
    df = base_df.copy()
    if df.empty:
        return df, {"injected_event_count": 0, "injected_max_delta_mag": np.nan}

    rng = np.random.default_rng(int(row["trial_seed"]))
    class_name = str(row["class_name"])
    if class_name == "shuffled_magnitude_control":
        df = _shuffle_magnitudes_within_camera(df, rng)
    elif class_name == "synthetic_gaussian_noise_control":
        df = _replace_with_gaussian_noise_control(df, rng)

    t = pd.to_numeric(df["JD"], errors="coerce").to_numpy(dtype=float)
    signal = np.zeros(len(df), dtype=float)
    amplitude = float(row.get("amplitude", 0.0) or 0.0)
    duration = float(row.get("duration", 0.0) or 0.0)
    injected_event_count = 0

    if class_name == "smooth_periodic_contaminant":
        period = float(row.get("period_days", np.nan))
        if np.isfinite(period) and period > 0:
            signal = amplitude * np.sin(2.0 * np.pi * (t - float(row.get("t0", 0.0))) / period)
    elif class_name != "no_injection_control" and amplitude > 0 and duration > 0:
        if class_name in {"periodic_fixed_depth_dips", "periodic_varying_depth_dips", "semi_periodic_dips"}:
            centers = _periodic_centers(
                t,
                period_days=float(row.get("period_days", np.nan)),
                phase0=float(row.get("phase0", 0.0)),
                duration=duration,
                jitter_frac=float(row.get("period_jitter_frac", 0.0)),
                rng=rng,
            )
        elif class_name == "non_periodic_recurrent_dips":
            finite = t[np.isfinite(t)]
            if finite.size:
                lo = float(np.min(finite) + duration)
                hi = float(np.max(finite) - duration)
                n_events = max(1, int(row.get("requested_event_count", 3)))
                centers = sorted(float(x) for x in rng.uniform(lo, hi, size=n_events)) if hi > lo else []
            else:
                centers = []
        else:
            centers = [float(row.get("t0", np.nan))]

        for center in centers:
            if not np.isfinite(center):
                continue
            scatter = float(row.get("depth_scatter", 0.0) or 0.0)
            depth = amplitude
            if scatter > 0:
                depth *= float(np.clip(rng.lognormal(mean=-0.5 * scatter**2, sigma=scatter), 0.2, 2.5))
            signal += _gaussian_dip_profile(t, center, duration, depth)
            injected_event_count += 1

    out = df.copy()
    out["mag"] = pd.to_numeric(out["mag"], errors="coerce").to_numpy(dtype=float) + signal
    out["benchmark_trial_id"] = int(row["trial_id"])
    out["benchmark_class"] = class_name
    return out, {
        "injected_event_count": int(injected_event_count),
        "injected_max_delta_mag": float(np.nanmax(signal)) if signal.size else np.nan,
    }


def _phase_complexity_ok(phase_peak_regions: Any, *, max_regions: float = PRE_PERIODICITY_PHASE_PEAK_REGION_MAX) -> bool:
    try:
        regions = float(phase_peak_regions)
    except (TypeError, ValueError):
        return True
    return (not np.isfinite(regions)) or regions <= float(max_regions)


def label_gate_from_features(
    features: pd.Series | dict[str, Any],
    *,
    ce_snr_threshold: float,
    scatter_ratio_max: float,
    phase_peak_snr_min: float = PRE_PERIODICITY_PHASE_PEAK_SNR_MIN,
    phase_peak_width_max: float = PRE_PERIODICITY_PHASE_PEAK_WIDTH_MAX,
    phase_peak_region_max: float = PRE_PERIODICITY_PHASE_PEAK_REGION_MAX,
    ce_snr_rescue_margin: float = PRE_PERIODICITY_CE_SNR_RESCUE_MARGIN,
    scatter_ratio_rescue_margin: float = PRE_PERIODICITY_SCATTER_RATIO_RESCUE_MARGIN,
) -> tuple[str, bool, str, bool]:
    row = dict(features)
    selected_snr = float(row.get("pre_periodicity_score", row.get("pre_ce_snr", np.nan)))
    scatter_ratio = float(row.get("pre_periodicity_scatter_ratio", np.nan))
    phase_peak_snr = float(row.get("pre_periodicity_phase_peak_snr", np.nan))
    phase_peak_width = float(row.get("pre_periodicity_phase_peak_width", np.nan))
    phase_peak_regions = float(row.get("pre_periodicity_phase_peak_regions", np.nan))
    alias_flag = bool(row.get("pre_periodicity_alias_flag", False))

    ce_support = bool(np.isfinite(selected_snr) and selected_snr >= float(ce_snr_threshold))
    scatter_ok = bool(np.isfinite(scatter_ratio) and scatter_ratio <= float(scatter_ratio_max))
    phase_peak_ok = bool(
        np.isfinite(phase_peak_snr)
        and phase_peak_snr >= float(phase_peak_snr_min)
        and np.isfinite(phase_peak_width)
        and phase_peak_width <= float(phase_peak_width_max)
    )
    phase_complexity_ok = _phase_complexity_ok(phase_peak_regions, max_regions=phase_peak_region_max)
    scatter_rescue_ok = bool(
        np.isfinite(scatter_ratio)
        and scatter_ratio <= (float(scatter_ratio_max) + float(scatter_ratio_rescue_margin))
    )
    ce_near_support = bool(
        np.isfinite(selected_snr)
        and selected_snr >= (float(ce_snr_threshold) - float(ce_snr_rescue_margin))
    )

    periodic_by_base = bool(ce_support and scatter_ok and phase_complexity_ok)
    periodic_by_scatter_rescue = bool(
        ce_support
        and (not scatter_ok)
        and phase_peak_ok
        and phase_complexity_ok
        and scatter_rescue_ok
    )
    periodic_by_ce_rescue = bool(
        (not ce_support)
        and ce_near_support
        and phase_peak_ok
        and phase_complexity_ok
        and scatter_rescue_ok
    )
    periodic = bool(periodic_by_base or periodic_by_scatter_rescue or periodic_by_ce_rescue)
    label = "periodic" if periodic else "non_periodic"

    if periodic_by_base:
        reasons = ["ce", "folded_scatter"]
    elif periodic_by_scatter_rescue:
        reasons = ["ce", "phase_peak", "scatter_rescue"]
    elif periodic_by_ce_rescue:
        reasons = ["ce_near_threshold"]
        if scatter_ok:
            reasons.extend(["folded_scatter", "phase_peak"])
        else:
            reasons.extend(["phase_peak", "scatter_rescue"])
    else:
        reasons = []
        if ce_support:
            reasons.append("ce")
        elif ce_near_support and phase_peak_ok:
            reasons.append("ce_near_threshold")
        else:
            reasons.append("ce_below_threshold")
        if scatter_ok:
            reasons.append("folded_scatter")
        elif np.isfinite(scatter_ratio):
            reasons.append("scatter_too_high")
        else:
            reasons.append("no_folded_scatter")
        if phase_peak_ok:
            reasons.append("phase_peak")
        if np.isfinite(phase_peak_regions) and not phase_complexity_ok:
            reasons.append("phase_complexity")
        if alias_flag:
            reasons.append("alias")
    return label, periodic, ",".join(reasons), phase_peak_ok


def evaluate_gate_on_dataframe(
    df_lc: pd.DataFrame,
    *,
    min_period: float,
    max_period: float,
    n_periods: int,
    ce_snr_threshold: float,
    min_points: int,
    scatter_ratio_max: float,
    bad_camera_scatter_ratio: float = BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    clean_max_error_absolute: float = CLEAN_LC_MAX_ERROR_ABSOLUTE,
    clean_max_error_sigma: float = CLEAN_LC_MAX_ERROR_SIGMA,
) -> dict[str, Any]:
    try:
        df = df_lc.copy()
        if df.empty:
            return _empty_gate_result("empty_lc")
        if "camera#" in df.columns:
            df, _ = filter_bad_cameras(
                df,
                filter_scatter=False,
                filter_offset=False,
                filter_catastrophic=True,
                scatter_ratio_threshold=float(bad_camera_scatter_ratio),
            )
        df = clean_lc(
            df,
            max_error_absolute=float(clean_max_error_absolute),
            max_error_sigma=float(clean_max_error_sigma),
        )
        if df.empty:
            return _empty_gate_result("empty_after_clean")

        n_points = int(len(df))
        n_cameras = int(compute_n_cameras(df)) if "camera#" in df.columns else 0
        if n_points < int(min_points):
            result = _empty_gate_result("too_few_points")
            result["pre_n_points"] = n_points
            result["pre_n_cameras"] = n_cameras
            return result

        df_aligned, v_minus_g_median_offset = pregate.align_v_to_g_magnitude(df)
        jd = df_aligned["JD"].to_numpy(dtype=float)
        mag = df_aligned["mag"].to_numpy(dtype=float)
        err = df_aligned["error"].to_numpy(dtype=float)
        ce_result = compute_ce_stats(
            jd,
            mag,
            err,
            min_period=float(min_period),
            max_period=float(max_period),
            n_periods=int(n_periods),
            n_bootstrap=0,
            refine=True,
        )
        ce_support = bool(
            np.isfinite(ce_result.get("ce_snr", np.nan))
            and float(ce_result["ce_snr"]) >= float(ce_snr_threshold)
        )
        band_resid = pregate._build_band_residuals(df_aligned)
        ce_candidate = pregate._harmonically_correct_period_candidate(
            "ce",
            float(ce_result.get("ce_period", np.nan)),
            snr=float(ce_result.get("ce_snr", np.nan)),
            supported=ce_support,
            band_resid=band_resid,
            min_period=float(min_period),
            max_period=float(max_period),
        )
        selected_candidate = pregate._select_period_candidate([ce_candidate])
        if selected_candidate is None:
            selected_candidate = {}

        features = {
            "pre_n_points": n_points,
            "pre_n_cameras": n_cameras,
            "pre_ce_period": float(ce_result.get("ce_period", np.nan)),
            "pre_ce_corrected_period": float(ce_candidate.get("corrected_period", np.nan)),
            "pre_ce_harmonic_factor": float(ce_candidate.get("harmonic_factor", np.nan)),
            "pre_ce_entropy": float(ce_result.get("ce_min_entropy", np.nan)),
            "pre_ce_snr": float(ce_result.get("ce_snr", np.nan)),
            "pre_ce_support": ce_support,
            "pre_periodicity_router_mode": pregate.PREGATE_ROUTER_MODE,
            "pre_periodicity_v_minus_g_median_offset": float(v_minus_g_median_offset),
            "pre_periodicity_method": selected_candidate.get("method"),
            "pre_periodicity_base_period": float(selected_candidate.get("raw_period", np.nan)),
            "pre_periodicity_selected_period": float(selected_candidate.get("corrected_period", np.nan)),
            "pre_periodicity_harmonic_factor": float(selected_candidate.get("harmonic_factor", np.nan)),
            "pre_periodicity_selection_objective": float(selected_candidate.get("selection_objective", np.nan)),
            "pre_periodicity_support_count": int(bool(selected_candidate.get("supported", False))),
            "pre_periodicity_score": float(selected_candidate.get("snr", np.nan)),
            "pre_periodicity_scatter_ratio": float(selected_candidate.get("scatter_ratio", np.nan)),
            "pre_periodicity_phase_peak_snr": float(selected_candidate.get("phase_peak_snr", np.nan)),
            "pre_periodicity_phase_peak_width": float(selected_candidate.get("phase_peak_width", np.nan)),
            "pre_periodicity_phase_peak_regions": float(selected_candidate.get("phase_peak_regions", np.nan)),
            "pre_periodicity_phase_lag_g_v_cycles": float(selected_candidate.get("phase_lag_g_v_cycles", np.nan)),
            "pre_periodicity_phase_lag_g_v_abs_cycles": float(selected_candidate.get("phase_lag_g_v_abs_cycles", np.nan)),
            "pre_periodicity_alias_flag": bool(selected_candidate.get("alias_flag", False)),
            "pre_periodicity_error": None,
        }
        label, periodic, reason, phase_peak_ok = label_gate_from_features(
            features,
            ce_snr_threshold=ce_snr_threshold,
            scatter_ratio_max=scatter_ratio_max,
        )
        features.update(
            {
                "pre_periodicity_phase_peak_flag": bool(phase_peak_ok),
                "pre_periodicity_label": label,
                "pre_periodic_flag": bool(periodic),
                "pre_periodicity_reason": reason,
            }
        )
        return features
    except Exception as exc:
        result = _empty_gate_result("error")
        result["pre_periodicity_error"] = str(exc)
        return result


def _empty_gate_result(reason: str) -> dict[str, Any]:
    return {
        "pre_n_points": 0,
        "pre_n_cameras": 0,
        "pre_ce_period": np.nan,
        "pre_ce_corrected_period": np.nan,
        "pre_ce_harmonic_factor": np.nan,
        "pre_ce_entropy": np.nan,
        "pre_ce_snr": np.nan,
        "pre_ce_support": False,
        "pre_periodicity_router_mode": pregate.PREGATE_ROUTER_MODE,
        "pre_periodicity_v_minus_g_median_offset": np.nan,
        "pre_periodicity_method": None,
        "pre_periodicity_base_period": np.nan,
        "pre_periodicity_selected_period": np.nan,
        "pre_periodicity_harmonic_factor": np.nan,
        "pre_periodicity_selection_objective": np.nan,
        "pre_periodicity_support_count": 0,
        "pre_periodicity_score": np.nan,
        "pre_periodicity_scatter_ratio": np.nan,
        "pre_periodicity_phase_peak_snr": np.nan,
        "pre_periodicity_phase_peak_width": np.nan,
        "pre_periodicity_phase_peak_regions": np.nan,
        "pre_periodicity_phase_peak_flag": False,
        "pre_periodicity_phase_lag_g_v_cycles": np.nan,
        "pre_periodicity_phase_lag_g_v_abs_cycles": np.nan,
        "pre_periodicity_alias_flag": False,
        "pre_periodicity_label": "non_periodic",
        "pre_periodic_flag": False,
        "pre_periodicity_reason": reason,
        "pre_periodicity_error": None,
    }


def baseline_func_from_name(name: str):
    return {
        "gp": per_camera_gp_baseline,
        "gp_masked": per_camera_gp_baseline_masked,
        "global_median": global_median_baseline,
        "per_camera_median": per_camera_median_baseline,
        "phase_template": phase_template_baseline,
    }.get(str(name), per_camera_gp_baseline_masked)


def _detection_options(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "trigger_mode": config.trigger_mode,
        "logbf_threshold_dip": config.logbf_threshold_dip,
        "logbf_threshold_jump": config.logbf_threshold_jump,
        "significance_threshold": config.significance_threshold,
        "p_points": int(config.p_points),
        "mag_points": int(config.mag_points),
        "run_min_points": int(config.run_min_points),
        "max_gap_points": int(config.run_max_gap_points),
        "run_max_gap_days": config.run_max_gap_days,
        "run_min_duration_days": config.run_min_duration_days,
        "compute_event_prob": bool(config.compute_event_prob),
    }


def _baseline_kwargs(config: BenchmarkConfig, *, period_days: float | None = None) -> dict[str, Any]:
    out = {
        "S0": BASELINE_S0,
        "w0": BASELINE_W0,
        "q": BASELINE_Q,
        "jitter": BASELINE_JITTER,
        "sigma_floor": None,
        "add_sigma_eff_col": True,
    }
    if period_days is not None:
        out["period_days"] = period_days
    return out


def _best_morph_info(run_list: list[dict[str, Any]] | None) -> dict[str, float | str]:
    if not run_list:
        return {"morph": "none", "delta_bic": 0.0}
    best_run = max(run_list, key=lambda x: x.get("run_max", -np.inf))
    return {
        "morph": str(best_run.get("morphology", "none")),
        "delta_bic": float(best_run.get("delta_bic_null", 0.0) or 0.0),
    }


def _score_filter_fields(res: dict[str, Any]) -> dict[str, Any]:
    dip = res["dip"]
    jump = res["jump"]
    df = res.get("df")
    df_base = res.get("df_base")
    baseline_mags = None
    if isinstance(df_base, pd.DataFrame) and "baseline" in df_base.columns:
        baseline_mags = df_base["baseline"].to_numpy()

    dipper_score = 0.0
    jumper_score = 0.0
    if isinstance(df, pd.DataFrame) and bool(dip.get("significant", False)):
        dipper_score = float(compute_event_score(df, event_type="dip", baseline_mags=baseline_mags)[0])
    if isinstance(df, pd.DataFrame) and bool(jump.get("significant", False)):
        jumper_score = float(compute_event_score(df, event_type="jump", baseline_mags=baseline_mags)[0])

    dip_morph = _best_morph_info(dip.get("run_summaries", []))
    jump_morph = _best_morph_info(jump.get("run_summaries", []))
    return {
        "dip_count": int(len(dip.get("event_indices", []))),
        "jump_count": int(len(jump.get("event_indices", []))),
        "dip_run_count": int(dip.get("n_runs", 0) or 0),
        "jump_run_count": int(jump.get("n_runs", 0) or 0),
        "dip_max_run_points": int(dip.get("max_run_points", 0) or 0),
        "jump_max_run_points": int(jump.get("max_run_points", 0) or 0),
        "dip_max_run_cameras": int(dip.get("max_run_cameras", 0) or 0),
        "jump_max_run_cameras": int(jump.get("max_run_cameras", 0) or 0),
        "dip_best_morph": str(dip_morph["morph"]),
        "jump_best_morph": str(jump_morph["morph"]),
        "dip_best_delta_bic": float(dip_morph["delta_bic"]),
        "jump_best_delta_bic": float(jump_morph["delta_bic"]),
        "dipper_score": dipper_score,
        "jumper_score": jumper_score,
    }


def _empty_score_result(baseline_name: str, error: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "detected": False,
        "dip_significant": False,
        "jump_significant": False,
        "dip_bayes_factor": np.nan,
        "jump_bayes_factor": np.nan,
        "dip_best_p": np.nan,
        "jump_best_p": np.nan,
        "dip_max_log_bf_local": np.nan,
        "jump_max_log_bf_local": np.nan,
        "baseline_mag": np.nan,
        "dip_best_mag_event": np.nan,
        "jump_best_mag_event": np.nan,
        "baseline_source": str(baseline_name),
        "error": error,
    }
    out.update(
        {
            "dip_count": 0,
            "jump_count": 0,
            "dip_run_count": 0,
            "jump_run_count": 0,
            "dip_max_run_points": 0,
            "jump_max_run_points": 0,
            "dip_max_run_cameras": 0,
            "jump_max_run_cameras": 0,
            "dip_best_morph": "none",
            "jump_best_morph": "none",
            "dip_best_delta_bic": 0.0,
            "jump_best_delta_bic": 0.0,
            "dipper_score": 0.0,
            "jumper_score": 0.0,
        }
    )
    return out


def score_detection(
    df_lc: pd.DataFrame,
    config: BenchmarkConfig,
    *,
    baseline_name: str,
    period_days: float | None = None,
) -> dict[str, Any]:
    baseline_func = baseline_func_from_name(baseline_name)
    try:
        res = score_lightcurve(
            df_lc,
            baseline_func=baseline_func,
            baseline_kwargs=_baseline_kwargs(config, period_days=period_days),
            filter_residual_bad_cameras_enabled=False,
            bad_camera_scatter_ratio=BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
            **_detection_options(config),
        )
        dip = res["dip"]
        jump = res["jump"]
        baseline_mag = float(dip.get("baseline_mag", jump.get("baseline_mag", np.nan)))
        dip_best_mag_event = float(dip.get("best_mag_event", np.nan))
        jump_best_mag_event = float(jump.get("best_mag_event", np.nan))
        dip_significant = bool(dip.get("significant", False))
        jump_significant = bool(jump.get("significant", False))

        if config.min_mag_offset > 0 and np.isfinite(baseline_mag):
            dip_diff = abs(dip_best_mag_event - baseline_mag) if np.isfinite(dip_best_mag_event) else 0.0
            jump_diff = abs(jump_best_mag_event - baseline_mag) if np.isfinite(jump_best_mag_event) else 0.0
            if dip_diff <= float(config.min_mag_offset):
                dip_significant = False
            if jump_diff <= float(config.min_mag_offset):
                jump_significant = False

        df_base = res.get("df_base")
        if isinstance(df_base, pd.DataFrame) and "baseline_source" in df_base.columns:
            baseline_source = ",".join(sorted(set(df_base["baseline_source"].dropna().astype(str)))) or str(baseline_name)
        else:
            baseline_source = str(baseline_name)

        out = {
            "detected": bool(dip_significant),
            "dip_significant": bool(dip_significant),
            "jump_significant": bool(jump_significant),
            "dip_bayes_factor": float(dip.get("bayes_factor", np.nan)),
            "jump_bayes_factor": float(jump.get("bayes_factor", np.nan)),
            "dip_best_p": float(dip.get("best_p", np.nan)),
            "jump_best_p": float(jump.get("best_p", np.nan)),
            "dip_max_log_bf_local": float(dip.get("max_log_bf_local", np.nan)),
            "jump_max_log_bf_local": float(jump.get("max_log_bf_local", np.nan)),
            "baseline_mag": baseline_mag,
            "dip_best_mag_event": dip_best_mag_event,
            "jump_best_mag_event": jump_best_mag_event,
            "baseline_source": baseline_source,
            "error": None,
        }
        out.update(_score_filter_fields(res))
        return out
    except Exception as exc:
        return _empty_score_result(baseline_name, error=str(exc))


def _prefix_dict(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def period_match_quality(selected_period: float, true_period: float, *, rel_tol: float = 0.05) -> tuple[bool, float, float]:
    if not np.isfinite(selected_period) or not np.isfinite(true_period) or selected_period <= 0 or true_period <= 0:
        return False, np.nan, np.nan
    errors = [
        abs(float(selected_period) - float(true_period) * factor) / max(abs(float(true_period) * factor), 1e-9)
        for factor in PERIOD_MATCH_HARMONIC_FACTORS
    ]
    best_idx = int(np.argmin(errors))
    return bool(errors[best_idx] <= float(rel_tol)), float(errors[best_idx]), float(PERIOD_MATCH_HARMONIC_FACTORS[best_idx])


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)


_TRIAL_WORKER_CONTEXT: dict[str, Any] = {}


def _init_trial_worker(controls: dict[str, pd.DataFrame], config: BenchmarkConfig) -> None:
    _TRIAL_WORKER_CONTEXT["controls"] = controls
    _TRIAL_WORKER_CONTEXT["config"] = config


def _process_trial_record_worker(trial_dict: dict[str, Any]) -> dict[str, Any]:
    controls = _TRIAL_WORKER_CONTEXT["controls"]
    config = _TRIAL_WORKER_CONTEXT["config"]
    row, _ = _evaluate_trial_record(trial_dict, controls, config, return_lightcurve=False)
    return row


def _evaluate_trial_record(
    trial_dict: dict[str, Any],
    controls: dict[str, pd.DataFrame],
    config: BenchmarkConfig,
    *,
    return_lightcurve: bool,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    base = controls[str(trial_dict["source_path"])]
    df_trial, injection_meta = generate_trial_lightcurve(base, trial_dict)

    gate = evaluate_gate_on_dataframe(
        df_trial,
        min_period=config.gate_min_period,
        max_period=config.gate_max_period,
        n_periods=config.effective_gate_n_periods,
        ce_snr_threshold=config.gate_ce_snr_threshold,
        min_points=config.gate_min_points,
        scatter_ratio_max=config.gate_scatter_ratio_max,
    )
    selected_period = float(gate.get("pre_periodicity_selected_period", np.nan))
    standard = score_detection(df_trial, config, baseline_name=config.baseline_func)
    phase = score_detection(df_trial, config, baseline_name="phase_template", period_days=selected_period)
    bifurcated_detected = bool(phase["detected"] if bool(gate.get("pre_periodic_flag", False)) else standard["detected"])

    period_usable, period_rel_error, period_harmonic_factor = period_match_quality(
        selected_period,
        float(trial_dict.get("period_days", np.nan)),
    )
    row = {
        **trial_dict,
        **injection_meta,
        **gate,
        **_prefix_dict("standard", standard),
        **_prefix_dict("phase_folded", phase),
        "bifurcated_detected": bifurcated_detected,
        "period_usable": bool(period_usable),
        "period_rel_error": period_rel_error,
        "period_match_harmonic_factor": period_harmonic_factor,
    }
    return row, df_trial if return_lightcurve else None


def _build_representative_lightcurve_cache(
    results: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
    *,
    max_examples: int = 6,
) -> dict[int, pd.DataFrame]:
    examples = _select_representative_rows(results, n=max_examples)
    out: dict[int, pd.DataFrame] = {}
    for _, row in examples.iterrows():
        trial_id = int(row["trial_id"])
        try:
            df_trial, _ = generate_trial_lightcurve(controls[str(row["source_path"])], row)
        except Exception:
            continue
        out[trial_id] = df_trial
    return out


def run_benchmark(config: BenchmarkConfig) -> BenchmarkRun:
    run_dir = make_run_dir(config)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    control_table, controls = load_control_lightcurves(
        config.bundle_lc_dir,
        sample_size=config.effective_control_sample_size,
        seed=config.seed,
        file_ext=config.lightcurve_file_ext,
        show_progress=config.show_progress,
    )
    trial_design = build_trial_design(control_table, config)
    _save_parquet(trial_design, run_dir / "trial_design.parquet")

    rows: list[dict[str, Any]] = []
    generated_lightcurves: dict[int, pd.DataFrame] = {}
    trial_records = trial_design.to_dict(orient="records")
    workers = max(1, int(config.workers))
    if workers > 1 and len(trial_records) > 1:
        max_workers = min(workers, len(trial_records))
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_trial_worker,
            initargs=(controls, config),
        ) as executor:
            iterator = executor.map(
                _process_trial_record_worker,
                trial_records,
                chunksize=max(1, int(config.trial_task_size)),
            )
            rows = list(
                tqdm(
                    iterator,
                    total=len(trial_records),
                    desc=f"Benchmark trials ({max_workers} workers)",
                    disable=not config.show_progress,
                )
            )
    else:
        iterator = tqdm(
            trial_records,
            total=len(trial_records),
            desc="Benchmark trials",
            disable=not config.show_progress,
        )
        for trial_dict in iterator:
            row, df_trial = _evaluate_trial_record(
                trial_dict,
                controls,
                config,
                return_lightcurve=bool(config.cache_generated_lightcurves),
            )
            rows.append(row)
            if df_trial is not None:
                generated_lightcurves[int(trial_dict["trial_id"])] = df_trial

    trial_results = pd.DataFrame(rows)
    _save_parquet(trial_results, run_dir / "trial_results.parquet")

    post_filter_results, post_filter_rejection_summary = run_post_filter_analysis(trial_results, config)
    _save_parquet(post_filter_results, run_dir / "post_filter_results.parquet")
    _save_parquet(post_filter_rejection_summary, run_dir / "post_filter_rejection_summary.parquet")

    if not generated_lightcurves and bool(config.cache_generated_lightcurves):
        generated_lightcurves = _build_representative_lightcurve_cache(trial_results, controls)

    gate_threshold_sweep = run_gate_threshold_sweep(trial_results)
    _save_parquet(gate_threshold_sweep, run_dir / "gate_threshold_sweep.parquet")

    period_search_subset_sweep = run_period_search_subset_sweep(
        trial_design,
        controls,
        config,
        generated_lightcurves=generated_lightcurves,
    )
    _save_parquet(period_search_subset_sweep, run_dir / "period_search_subset_sweep.parquet")

    summary_metrics = summarize_results(trial_results, post_filter_results)
    _save_parquet(summary_metrics, run_dir / "summary_metrics.parquet")

    save_benchmark_plots(
        trial_results,
        gate_threshold_sweep,
        run_dir=run_dir,
        generated_lightcurves=generated_lightcurves,
        controls=controls,
        post_filter_results=post_filter_results,
    )
    return BenchmarkRun(
        config=config,
        run_dir=run_dir,
        control_table=control_table,
        trial_design=trial_design,
        trial_results=trial_results,
        gate_threshold_sweep=gate_threshold_sweep,
        period_search_subset_sweep=period_search_subset_sweep,
        summary_metrics=summary_metrics,
        post_filter_results=post_filter_results,
        post_filter_rejection_summary=post_filter_rejection_summary,
        generated_lightcurves=generated_lightcurves,
    )


def _mean_bool(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float(series.fillna(False).astype(bool).mean())


def _route_confusion_metrics(df: pd.DataFrame, route_periodic: pd.Series) -> dict[str, Any]:
    hard = df["target_gate_label"].isin(["periodic", "non_periodic"])
    if not hard.any():
        return {
            "gate_hard_n": 0,
            "gate_periodic_recall": np.nan,
            "gate_non_periodic_specificity": np.nan,
            "gate_false_periodic_route_rate": np.nan,
            "gate_accuracy": np.nan,
        }
    truth_periodic = df.loc[hard, "target_gate_label"].eq("periodic")
    pred_periodic = route_periodic.loc[hard].fillna(False).astype(bool)
    tp = int((truth_periodic & pred_periodic).sum())
    fn = int((truth_periodic & ~pred_periodic).sum())
    tn = int((~truth_periodic & ~pred_periodic).sum())
    fp = int((~truth_periodic & pred_periodic).sum())
    return {
        "gate_hard_n": int(hard.sum()),
        "gate_periodic_recall": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "gate_non_periodic_specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "gate_false_periodic_route_rate": float(fp / (fp + tn)) if (fp + tn) else np.nan,
        "gate_accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
    }


def _default_post_filter_value(column: str) -> Any:
    if column.endswith("_significant"):
        return False
    if column.endswith("_morph"):
        return "none"
    if column.endswith("_score"):
        return 0.0
    if column.endswith("_count") or column.endswith("_points") or column.endswith("_cameras"):
        return 0
    if column.endswith("_delta_bic"):
        return 0.0
    return np.nan


def _pipeline_post_filter_input(results: pd.DataFrame, pipeline: str) -> pd.DataFrame:
    if pipeline not in PIPELINE_DETECTED_COLUMNS:
        raise ValueError(f"Unknown pipeline: {pipeline}")

    route_phase = results["pre_periodic_flag"].fillna(False).astype(bool)
    out = pd.DataFrame(
        {
            "path": [f"synthetic://{pipeline}/trial_{int(tid)}" for tid in results["trial_id"]],
            "trial_id": results["trial_id"].astype(int).to_numpy(),
            "class_name": results["class_name"].astype(str).to_numpy(),
            "source_id": results["source_id"].astype(str).to_numpy(),
            "source_path": results["source_path"].astype(str).to_numpy(),
            "target_dip": results["target_dip"].fillna(False).astype(bool).to_numpy(),
            "target_gate_label": results["target_gate_label"].astype(str).to_numpy(),
            "raw_detected": results[PIPELINE_DETECTED_COLUMNS[pipeline]].fillna(False).astype(bool).to_numpy(),
            "gaia_id": "",
            "pmra": np.nan,
            "pmdec": np.nan,
            "ra": np.nan,
            "dec": np.nan,
        }
    )

    if pipeline == "bifurcated_gate":
        for column in POST_FILTER_FIELD_COLUMNS:
            std_col = f"standard_{column}"
            phase_col = f"phase_folded_{column}"
            std_values = results[std_col] if std_col in results.columns else _default_post_filter_value(column)
            phase_values = results[phase_col] if phase_col in results.columns else _default_post_filter_value(column)
            out[column] = np.where(route_phase.to_numpy(), phase_values, std_values)
    else:
        prefix = "standard" if pipeline == "standard_only" else "phase_folded"
        for column in POST_FILTER_FIELD_COLUMNS:
            src = f"{prefix}_{column}"
            out[column] = results[src].to_numpy() if src in results.columns else _default_post_filter_value(column)

    return out


def _summarize_post_filter_rejections(post_filter_results: pd.DataFrame) -> pd.DataFrame:
    failed_cols = [
        col
        for col in post_filter_results.columns
        if col.startswith("failed_") and col != "failed_any"
    ]
    rows: list[dict[str, Any]] = []
    for pipeline, pipe_group in post_filter_results.groupby("pipeline", sort=False):
        groups: list[tuple[str, str, pd.DataFrame]] = [("all", "ALL", pipe_group)]
        groups.extend(
            ("class", str(class_name), group)
            for class_name, group in pipe_group.groupby("class_name", sort=False)
        )
        for scope, class_name, group in groups:
            raw = group["raw_detected"].fillna(False).astype(bool)
            post_pass = group["post_filter_passed"].fillna(False).astype(bool)
            failed_any = group.get("failed_any", pd.Series(False, index=group.index)).fillna(False).astype(bool)
            rows.append(
                {
                    "scope": scope,
                    "class_name": class_name,
                    "pipeline": pipeline,
                    "filter": "failed_any",
                    "n": int(len(group)),
                    "raw_detected_n": int(raw.sum()),
                    "post_filter_passed_n": int(post_pass.sum()),
                    "rejected_raw_detected_n": int((raw & failed_any).sum()),
                    "reject_fraction_raw_detected": float((raw & failed_any).sum() / raw.sum()) if int(raw.sum()) else np.nan,
                }
            )
            for failed_col in failed_cols:
                failed = group[failed_col].fillna(False).astype(bool)
                rows.append(
                    {
                        "scope": scope,
                        "class_name": class_name,
                        "pipeline": pipeline,
                        "filter": failed_col.removeprefix("failed_"),
                        "n": int(len(group)),
                        "raw_detected_n": int(raw.sum()),
                        "post_filter_passed_n": int(post_pass.sum()),
                        "rejected_raw_detected_n": int((raw & failed).sum()),
                        "reject_fraction_raw_detected": float((raw & failed).sum() / raw.sum()) if int(raw.sum()) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def run_post_filter_analysis(
    results: pd.DataFrame,
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for pipeline in PIPELINE_DETECTED_COLUMNS:
        filter_input = _pipeline_post_filter_input(results, pipeline)
        filtered = apply_filters(
            filter_input,
            apply_periodic_catalog_validation=bool(config.post_filter_apply_periodic_catalog_validation),
            apply_gaia_ruwe_validation=bool(config.post_filter_apply_gaia_ruwe_validation),
            apply_gaia_pm_validation=bool(config.post_filter_apply_gaia_pm_validation),
            apply_periodicity_validation=bool(config.post_filter_apply_periodicity_validation),
            show_tqdm=bool(config.show_progress),
            verbose=False,
        )
        raw = filtered["raw_detected"].fillna(False).astype(bool)
        failed_any = filtered.get("failed_any", pd.Series(False, index=filtered.index)).fillna(False).astype(bool)
        filtered["pipeline"] = pipeline
        filtered["post_filter_passed"] = raw & ~failed_any
        frames.append(filtered)

    post_filter_results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return post_filter_results, _summarize_post_filter_rejections(post_filter_results)


def _post_filter_metrics_for_trials(
    post_filter_results: pd.DataFrame | None,
    *,
    pipeline: str,
    trial_ids: pd.Series,
) -> dict[str, Any]:
    empty = {
        "post_filter_passed_n": np.nan,
        "post_filter_pass_rate": np.nan,
        "post_filter_target_recovery": np.nan,
        "final_false_positive_rate": np.nan,
        "post_filter_detection_rate": np.nan,
    }
    if post_filter_results is None or post_filter_results.empty:
        return empty
    ids = set(pd.to_numeric(trial_ids, errors="coerce").dropna().astype(int))
    group = post_filter_results[
        post_filter_results["pipeline"].eq(pipeline)
        & post_filter_results["trial_id"].astype(int).isin(ids)
    ]
    if group.empty:
        return empty
    target = group["target_dip"].fillna(False).astype(bool)
    passed = group["post_filter_passed"].fillna(False).astype(bool)
    return {
        "post_filter_passed_n": int(passed.sum()),
        "post_filter_pass_rate": _mean_bool(passed),
        "post_filter_target_recovery": _mean_bool(passed[target]) if bool(target.any()) else np.nan,
        "final_false_positive_rate": _mean_bool(passed[~target]) if bool((~target).any()) else np.nan,
        "post_filter_detection_rate": _mean_bool(passed),
    }


def summarize_results(
    results: pd.DataFrame,
    post_filter_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pipelines = {
        pipeline: results[column].fillna(False).astype(bool)
        for pipeline, column in PIPELINE_DETECTED_COLUMNS.items()
    }
    for pipeline, detected in pipelines.items():
        target = results["target_dip"].fillna(False).astype(bool)
        rows.append(
            {
                "scope": "all",
                "class_name": "ALL",
                "pipeline": pipeline,
                "n": int(len(results)),
                "raw_detected_n": int(detected.sum()),
                "target_recovery": _mean_bool(detected[target]),
                "false_positive_rate": _mean_bool(detected[~target]),
                "detection_rate": _mean_bool(detected),
                **_post_filter_metrics_for_trials(
                    post_filter_results,
                    pipeline=pipeline,
                    trial_ids=results["trial_id"],
                ),
                "gate_periodic_fraction": _mean_bool(results["pre_periodic_flag"]),
                "period_usable_rate": _mean_bool(results.loc[results["target_gate_label"].isin(["periodic", "ambiguous"]), "period_usable"]),
                **_route_confusion_metrics(results, results["pre_periodic_flag"].fillna(False).astype(bool)),
            }
        )
        for class_name, group in results.groupby("class_name", sort=False):
            gdet = detected.loc[group.index]
            gtarget = group["target_dip"].fillna(False).astype(bool)
            rows.append(
                {
                    "scope": "class",
                    "class_name": str(class_name),
                    "pipeline": pipeline,
                    "n": int(len(group)),
                    "raw_detected_n": int(gdet.sum()),
                    "target_recovery": _mean_bool(gdet[gtarget]) if bool(gtarget.any()) else np.nan,
                    "false_positive_rate": _mean_bool(gdet[~gtarget]) if bool((~gtarget).any()) else np.nan,
                    "detection_rate": _mean_bool(gdet),
                    **_post_filter_metrics_for_trials(
                        post_filter_results,
                        pipeline=pipeline,
                        trial_ids=group["trial_id"],
                    ),
                    "gate_periodic_fraction": _mean_bool(group["pre_periodic_flag"]),
                    "period_usable_rate": _mean_bool(group.loc[group["target_gate_label"].isin(["periodic", "ambiguous"]), "period_usable"]),
                    **_route_confusion_metrics(group, group["pre_periodic_flag"].fillna(False).astype(bool)),
                }
            )
    return pd.DataFrame(rows)


def run_gate_threshold_sweep(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ce_snr_threshold, scatter_ratio_max, phase_peak_snr_min in product(
        THRESHOLD_CE_SNR_VALUES,
        THRESHOLD_SCATTER_RATIO_VALUES,
        THRESHOLD_PHASE_PEAK_SNR_VALUES,
    ):
        labels: list[str] = []
        flags: list[bool] = []
        reasons: list[str] = []
        for _, row in results.iterrows():
            label, flag, reason, _ = label_gate_from_features(
                row,
                ce_snr_threshold=float(ce_snr_threshold),
                scatter_ratio_max=float(scatter_ratio_max),
                phase_peak_snr_min=float(phase_peak_snr_min),
            )
            labels.append(label)
            flags.append(flag)
            reasons.append(reason)
        route_periodic = pd.Series(flags, index=results.index, dtype=bool)
        bifurcated_detected = pd.Series(
            np.where(
                route_periodic,
                results["phase_folded_detected"].fillna(False).astype(bool),
                results["standard_detected"].fillna(False).astype(bool),
            ),
            index=results.index,
        )
        target = results["target_dip"].fillna(False).astype(bool)
        common = {
            "ce_snr_threshold": float(ce_snr_threshold),
            "scatter_ratio_max": float(scatter_ratio_max),
            "phase_peak_snr_min": float(phase_peak_snr_min),
            "periodic_branch_load": _mean_bool(route_periodic),
            "target_recovery": _mean_bool(bifurcated_detected[target]),
            "false_positive_rate": _mean_bool(bifurcated_detected[~target]),
            "detection_rate": _mean_bool(bifurcated_detected),
            **_route_confusion_metrics(results, route_periodic),
        }
        rows.append({"scope": "all", "class_name": "ALL", **common})
        for class_name, group in results.groupby("class_name", sort=False):
            idx = group.index
            gtarget = group["target_dip"].fillna(False).astype(bool)
            rows.append(
                {
                    "scope": "class",
                    "class_name": str(class_name),
                    "ce_snr_threshold": float(ce_snr_threshold),
                    "scatter_ratio_max": float(scatter_ratio_max),
                    "phase_peak_snr_min": float(phase_peak_snr_min),
                    "periodic_branch_load": _mean_bool(route_periodic.loc[idx]),
                    "target_recovery": _mean_bool(bifurcated_detected.loc[idx][gtarget]) if bool(gtarget.any()) else np.nan,
                    "false_positive_rate": _mean_bool(bifurcated_detected.loc[idx][~gtarget]) if bool((~gtarget).any()) else np.nan,
                    "detection_rate": _mean_bool(bifurcated_detected.loc[idx]),
                    **_route_confusion_metrics(group, route_periodic.loc[idx]),
                }
            )
    return pd.DataFrame(rows)


def run_period_search_subset_sweep(
    trial_design: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
    config: BenchmarkConfig,
    *,
    generated_lightcurves: dict[int, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    subset = trial_design.head(min(config.effective_period_search_subset_size, len(trial_design))).copy()
    rows: list[dict[str, Any]] = []
    generated_lightcurves = generated_lightcurves or {}
    for n_periods, min_period, max_period in product(
        PERIOD_SEARCH_N_PERIODS_VALUES,
        PERIOD_SEARCH_MIN_PERIOD_VALUES,
        PERIOD_SEARCH_MAX_PERIOD_VALUES,
    ):
        gate_rows: list[dict[str, Any]] = []
        iterator = tqdm(
            subset.iterrows(),
            total=len(subset),
            desc=f"Period sweep n={n_periods} min={min_period} max={max_period}",
            disable=not config.show_progress,
            leave=False,
        )
        for _, trial in iterator:
            trial_id = int(trial["trial_id"])
            df_trial = generated_lightcurves.get(trial_id)
            if df_trial is None:
                df_trial, _ = generate_trial_lightcurve(controls[str(trial["source_path"])], trial)
            gate = evaluate_gate_on_dataframe(
                df_trial,
                min_period=float(min_period),
                max_period=float(max_period),
                n_periods=int(n_periods),
                ce_snr_threshold=config.gate_ce_snr_threshold,
                min_points=config.gate_min_points,
                scatter_ratio_max=config.gate_scatter_ratio_max,
            )
            usable, rel_error, harmonic = period_match_quality(
                float(gate.get("pre_periodicity_selected_period", np.nan)),
                float(trial.get("period_days", np.nan)),
            )
            gate_rows.append(
                {
                    "trial_id": trial_id,
                    "pre_periodic_flag": bool(gate.get("pre_periodic_flag", False)),
                    "pre_periodicity_label": gate.get("pre_periodicity_label"),
                    "period_usable": bool(usable),
                    "period_rel_error": rel_error,
                    "period_match_harmonic_factor": harmonic,
                }
            )
        gate_df = pd.DataFrame(gate_rows).merge(
            subset[["trial_id", "class_name", "target_gate_label"]],
            on="trial_id",
            how="left",
        )
        route = gate_df["pre_periodic_flag"].fillna(False).astype(bool)
        rows.append(
            {
                "n_periods": int(n_periods),
                "min_period": float(min_period),
                "max_period": float(max_period),
                "n": int(len(gate_df)),
                "periodic_branch_load": _mean_bool(route),
                "period_usable_rate": _mean_bool(gate_df.loc[gate_df["target_gate_label"].isin(["periodic", "ambiguous"]), "period_usable"]),
                **_route_confusion_metrics(gate_df, route),
            }
        )
    return pd.DataFrame(rows)


def _finite_metric(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _fmt_metric(value: object, fmt: str = ".3g") -> str:
    number = _finite_metric(value)
    return format(number, fmt) if np.isfinite(number) else "nan"


def _as_bool(value: object, *, default: bool = False) -> bool:
    if pd.isna(value):
        return bool(default)
    return bool(value)


def _take_example_rows(
    df: pd.DataFrame,
    *,
    mask: pd.Series,
    label: str,
    n: int,
    sort_by: tuple[str, ...] = ("pre_periodicity_score", "injected_max_delta_mag"),
) -> pd.DataFrame:
    sub = df.loc[mask].copy()
    if sub.empty:
        return sub
    sort_cols = [col for col in sort_by if col in sub.columns]
    if sort_cols:
        sub = sub.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return sub.head(int(n)).assign(example_kind=label)


def select_gate_processing_examples(
    results: pd.DataFrame,
    *,
    per_bucket: int = 2,
    max_examples: int = 14,
) -> pd.DataFrame:
    """Select deterministic light-curve examples for gate processing diagnostics."""
    if results.empty:
        return results.copy()

    route_periodic = results["pre_periodic_flag"].fillna(False).astype(bool)
    target_periodic = results["target_gate_label"].eq("periodic")
    target_nonperiodic = results["target_gate_label"].eq("non_periodic")
    target_dip = results["target_dip"].fillna(False).astype(bool)
    recovered = results["bifurcated_detected"].fillna(False).astype(bool)

    frames: list[pd.DataFrame] = [
        _take_example_rows(
            results,
            mask=target_periodic & route_periodic,
            label="periodic target routed periodic",
            n=per_bucket,
        ),
        _take_example_rows(
            results,
            mask=target_periodic & ~route_periodic,
            label="periodic target routed non-periodic",
            n=per_bucket,
        ),
        _take_example_rows(
            results,
            mask=target_nonperiodic & ~route_periodic & target_dip,
            label="non-periodic dip routed non-periodic",
            n=per_bucket,
        ),
        _take_example_rows(
            results,
            mask=target_nonperiodic & route_periodic,
            label="non-periodic target routed periodic",
            n=per_bucket,
        ),
        _take_example_rows(
            results,
            mask=results["class_name"].eq("semi_periodic_dips") & route_periodic,
            label="semi-periodic routed periodic",
            n=per_bucket,
        ),
        _take_example_rows(
            results,
            mask=results["class_name"].eq("semi_periodic_dips") & ~route_periodic,
            label="semi-periodic routed non-periodic",
            n=per_bucket,
        ),
        _take_example_rows(
            results,
            mask=target_dip & recovered & ~results["standard_detected"].fillna(False).astype(bool),
            label="bifurcated rescue",
            n=per_bucket,
            sort_by=("injected_max_delta_mag", "pre_periodicity_score"),
        ),
        _take_example_rows(
            results,
            mask=target_dip & ~recovered,
            label="bifurcated miss",
            n=per_bucket,
            sort_by=("injected_max_delta_mag", "pre_periodicity_score"),
        ),
        _take_example_rows(
            results,
            mask=results["class_name"].eq("no_injection_control")
            & results["standard_detected"].fillna(False).astype(bool),
            label="standard false positive control",
            n=per_bucket,
            sort_by=("standard_dip_bayes_factor", "standard_dip_max_log_bf_local"),
        ),
    ]

    selected = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if selected.empty:
        selected = results.head(int(max_examples)).copy().assign(example_kind="fallback")

    selected = selected.drop_duplicates(subset=["trial_id"], keep="first").head(int(max_examples)).copy()
    route = selected["pre_periodic_flag"].fillna(False).astype(bool)
    selected["routed_branch"] = np.where(route, "periodic", "non_periodic")
    return selected.reset_index(drop=True)


def _load_trial_lightcurve_for_row(
    row: pd.Series | dict[str, Any],
    *,
    controls: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    row_dict = dict(row)
    source_path = str(row_dict["source_path"])
    if controls is not None and source_path in controls:
        base = controls[source_path]
    else:
        base = load_lightcurve_df(source_path)
    df_trial, _ = generate_trial_lightcurve(base, row_dict)
    return df_trial


def _prepare_gate_processing_frames(
    df_lc: pd.DataFrame,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    raw = df_lc.copy().reset_index(drop=True)
    raw["_pgib_original_index"] = np.arange(len(raw), dtype=int)

    if raw.empty:
        empty = raw.copy()
        return {
            "raw": raw,
            "camera_filtered": empty,
            "cleaned": empty,
            "gp_base": empty,
            "bad_cameras": set(),
            "bad_camera_rejected": empty,
            "bad_point_rejected": empty,
        }

    camera_filtered, bad_cameras = filter_bad_cameras(
        raw,
        filter_scatter=False,
        filter_offset=False,
        filter_catastrophic=True,
        scatter_ratio_threshold=BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    )
    cleaned = clean_lc(
        camera_filtered,
        max_error_absolute=CLEAN_LC_MAX_ERROR_ABSOLUTE,
        max_error_sigma=CLEAN_LC_MAX_ERROR_SIGMA,
    )

    camera_index = camera_filtered.get("_pgib_original_index", pd.Series(dtype=float))
    clean_index = cleaned.get("_pgib_original_index", pd.Series(dtype=float))
    camera_ids = set(pd.to_numeric(camera_index, errors="coerce").dropna().astype(int))
    clean_ids = set(pd.to_numeric(clean_index, errors="coerce").dropna().astype(int))
    raw_ids = pd.to_numeric(raw["_pgib_original_index"], errors="coerce").astype(int)
    bad_camera_rejected = raw.loc[~raw_ids.isin(camera_ids)].copy()
    bad_point_rejected = raw.loc[raw_ids.isin(camera_ids) & ~raw_ids.isin(clean_ids)].copy()

    gp_base = pd.DataFrame()
    if not cleaned.empty:
        gp_base = per_camera_gp_baseline_masked(cleaned, **_baseline_kwargs(config))

    return {
        "raw": raw,
        "camera_filtered": camera_filtered.reset_index(drop=True),
        "cleaned": cleaned.reset_index(drop=True),
        "gp_base": gp_base.reset_index(drop=True) if not gp_base.empty else gp_base,
        "bad_cameras": set(bad_cameras),
        "bad_camera_rejected": bad_camera_rejected.reset_index(drop=True),
        "bad_point_rejected": bad_point_rejected.reset_index(drop=True),
    }


def _score_branch_for_processing_plot(
    cleaned: pd.DataFrame,
    row: pd.Series | dict[str, Any],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    if cleaned.empty:
        return {"df": cleaned, "df_base": cleaned, "dip": {}, "jump": {}, "branch_baseline": "empty"}

    route_periodic = _as_bool(dict(row).get("pre_periodic_flag", False))
    selected_period = _finite_metric(dict(row).get("pre_periodicity_selected_period", np.nan))
    use_phase_template = bool(route_periodic and np.isfinite(selected_period) and selected_period > 0)
    if use_phase_template:
        baseline_func = phase_template_baseline
        baseline_name = "phase_template"
        baseline_kwargs = _baseline_kwargs(config, period_days=selected_period)
    else:
        baseline_func = baseline_func_from_name(config.baseline_func)
        baseline_name = str(config.baseline_func)
        baseline_kwargs = _baseline_kwargs(config)

    try:
        scored = score_lightcurve(
            cleaned,
            baseline_func=baseline_func,
            baseline_kwargs=baseline_kwargs,
            filter_residual_bad_cameras_enabled=False,
            bad_camera_scatter_ratio=BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
            **_detection_options(config),
        )
        scored["branch_baseline"] = baseline_name
        return scored
    except Exception as exc:
        fallback = baseline_func(cleaned, **baseline_kwargs)
        return {
            "df": cleaned,
            "df_base": fallback,
            "dip": {"event_indices": [], "error": str(exc)},
            "jump": {"event_indices": [], "error": str(exc)},
            "branch_baseline": baseline_name,
        }


def _empty_branch_score(cleaned: pd.DataFrame, label: str, reason: str) -> dict[str, Any]:
    return {
        "df": cleaned,
        "df_base": pd.DataFrame(),
        "dip": {"event_indices": [], "log_bf_local": np.array([]), "event_probability": None, "error": reason},
        "jump": {"event_indices": [], "log_bf_local": np.array([]), "event_probability": None, "error": reason},
        "branch_baseline": label,
        "score_error": reason,
    }


def _score_named_branch_for_processing_plot(
    cleaned: pd.DataFrame,
    config: BenchmarkConfig,
    *,
    baseline_name: str,
    period_days: float | None = None,
) -> dict[str, Any]:
    if cleaned.empty:
        return _empty_branch_score(cleaned, baseline_name, "empty_cleaned_lightcurve")

    baseline_func = baseline_func_from_name(baseline_name)
    baseline_kwargs = _baseline_kwargs(config, period_days=period_days)
    try:
        scored = score_lightcurve(
            cleaned,
            baseline_func=baseline_func,
            baseline_kwargs=baseline_kwargs,
            filter_residual_bad_cameras_enabled=False,
            bad_camera_scatter_ratio=BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
            **_detection_options(config),
        )
        scored["branch_baseline"] = baseline_name
        return scored
    except Exception as exc:
        try:
            fallback = baseline_func(cleaned, **baseline_kwargs)
        except Exception:
            fallback = pd.DataFrame()
        return {
            "df": cleaned,
            "df_base": fallback,
            "dip": {"event_indices": [], "log_bf_local": np.array([]), "event_probability": None, "error": str(exc)},
            "jump": {"event_indices": [], "log_bf_local": np.array([]), "event_probability": None, "error": str(exc)},
            "branch_baseline": baseline_name,
            "score_error": str(exc),
        }


def _score_processing_branches(
    cleaned: pd.DataFrame,
    row: pd.Series | dict[str, Any],
    config: BenchmarkConfig,
) -> dict[str, dict[str, Any]]:
    selected_period = _finite_metric(dict(row).get("pre_periodicity_selected_period", np.nan))
    stochastic = _score_named_branch_for_processing_plot(
        cleaned,
        config,
        baseline_name=str(config.baseline_func),
    )
    if np.isfinite(selected_period) and selected_period > 0:
        phase = _score_named_branch_for_processing_plot(
            cleaned,
            config,
            baseline_name="phase_template",
            period_days=selected_period,
        )
    else:
        phase = _empty_branch_score(cleaned, "phase_template", "missing_gate_selected_period")
    return {"stochastic": stochastic, "phase": phase}


def _branch_frames(score: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(score.get("df", pd.DataFrame())).reset_index(drop=True)
    df_base = pd.DataFrame(score.get("df_base", pd.DataFrame())).reset_index(drop=True)
    return df, df_base


def _plot_missing_panel(ax, message: str, *, title: str | None = None) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def _plot_branch_baseline_panel(
    ax,
    score: dict[str, Any],
    *,
    title: str,
    color: str,
) -> None:
    df, df_base = _branch_frames(score)
    if df.empty or df_base.empty or "baseline" not in df_base.columns:
        _plot_missing_panel(ax, str(score.get("score_error", "missing branch baseline")), title=title)
        return
    group_by = "camera" if "camera#" in df.columns else "none"
    plot_lightcurve_panel(
        ax,
        df,
        group_by=group_by,
        camera_col="camera#" if "camera#" in df.columns else None,
        show_errorbars=False,
        marker_size=2.5,
        legend="none",
        time_offset="none",
        xlabel="JD",
        ylabel="mag",
        baseline=df_base,
        baseline_col="baseline",
        baseline_time_col="JD",
        baseline_group_col="camera#" if "camera#" in df_base.columns else None,
        baseline_label=str(score.get("branch_baseline", "baseline")),
        baseline_style={"color": color, "linewidth": 1.15, "alpha": 0.9},
    )
    ax.set_title(title, fontsize=10)


def _plot_branch_residual_time_panel(
    ax,
    score: dict[str, Any],
    *,
    title: str,
) -> None:
    df, df_base = _branch_frames(score)
    if df.empty or df_base.empty or "resid" not in df_base.columns:
        _plot_missing_panel(ax, str(score.get("score_error", "missing branch residuals")), title=title)
        return
    plot_residual_panel(
        ax,
        df_base,
        residual_col="resid",
        group_by="camera" if "camera#" in df_base.columns else "none",
        camera_col="camera#" if "camera#" in df_base.columns else None,
        show_errorbars=False,
        marker_size=2.6,
        legend="none",
        time_offset="none",
        xlabel="JD",
        ylabel="mag - baseline",
        invert_y=False,
    )
    dip_idx = np.asarray(score.get("dip", {}).get("event_indices", []), dtype=int)
    _highlight_event_indices(ax, df, df_base["resid"], dip_idx, label="dip event points")
    if dip_idx.size:
        ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_title(title, fontsize=10)


def _highlight_phase_event_indices(
    ax,
    df: pd.DataFrame,
    y_values: pd.Series | np.ndarray,
    indices: Sequence[int],
    *,
    period_days: float,
    label: str,
) -> None:
    period = _finite_metric(period_days)
    if not np.isfinite(period) or period <= 0:
        return
    idx = np.asarray(indices, dtype=int)
    y_array = np.asarray(y_values, dtype=float)
    idx = idx[(idx >= 0) & (idx < len(df)) & (idx < y_array.size)]
    if idx.size == 0 or "JD" not in df.columns:
        return
    jd_all = pd.to_numeric(df["JD"], errors="coerce").to_numpy(dtype=float)
    finite_jd = jd_all[np.isfinite(jd_all)]
    if finite_jd.size == 0:
        return
    epoch = float(np.min(finite_jd))
    phase = np.mod((jd_all[idx] - epoch) / period, 1.0)
    y = y_array[idx]
    mask = np.isfinite(phase) & np.isfinite(y)
    if not mask.any():
        return
    phase = phase[mask]
    y = y[mask]
    ax.scatter(
        phase,
        y,
        marker="x",
        s=46,
        color="goldenrod",
        linewidths=1.2,
        label=label,
        zorder=9,
    )
    ax.scatter(
        phase + 1.0,
        y,
        marker="x",
        s=46,
        color="goldenrod",
        linewidths=1.2,
        zorder=9,
    )


def _plot_branch_residual_phase_panel(
    ax,
    score: dict[str, Any],
    *,
    period_days: float,
    title: str,
) -> None:
    df, df_base = _branch_frames(score)
    if df.empty or df_base.empty or "resid" not in df_base.columns:
        _plot_missing_panel(ax, str(score.get("score_error", "missing phase residuals")), title=title)
        return
    period = _finite_metric(period_days)
    if not np.isfinite(period) or period <= 0:
        _plot_missing_panel(ax, "No finite gate-selected period", title=title)
        return
    try:
        plot_phase_panel(
            ax,
            df_base,
            period_days=period,
            value_mode="resid",
            residual_col="resid",
            group_by="band" if "v_g_band" in df_base.columns else "none",
            show_errorbars=False,
            marker_size=2.6,
            legend="none",
            ylabel="mag - phase baseline",
        )
        ax.axhline(0.0, color="0.2", linestyle="--", linewidth=0.8, alpha=0.65)
        dip_idx = np.asarray(score.get("dip", {}).get("event_indices", []), dtype=int)
        _highlight_phase_event_indices(
            ax,
            df,
            df_base["resid"],
            dip_idx,
            period_days=period,
            label="dip event points",
        )
        if dip_idx.size:
            ax.legend(frameon=False, fontsize=8, loc="best")
        ax.set_title(title, fontsize=10)
    except Exception as exc:
        _plot_missing_panel(ax, f"Phase residual plot failed: {exc}", title=title)


def _dip_evidence_frame(score: dict[str, Any], metric: str) -> tuple[pd.DataFrame, np.ndarray]:
    df, df_base = _branch_frames(score)
    dip = score.get("dip", {}) or {}
    values = dip.get(metric)
    if values is None:
        return pd.DataFrame(), np.asarray(dip.get("event_indices", []), dtype=int)
    values_arr = np.asarray(values, dtype=float)
    if values_arr.size == 0:
        return pd.DataFrame(), np.asarray(dip.get("event_indices", []), dtype=int)

    source = df_base if len(df_base) == values_arr.size else df
    if source.empty or "JD" not in source.columns or len(source) != values_arr.size:
        return pd.DataFrame(), np.asarray(dip.get("event_indices", []), dtype=int)

    out = source.reset_index(drop=True).copy()
    out["evidence_value"] = values_arr
    return out, np.asarray(dip.get("event_indices", []), dtype=int)


def _plot_evidence_time_panel(
    ax,
    score: dict[str, Any],
    *,
    metric: str,
    title: str,
    ylabel: str,
    threshold: float | None = None,
    color: str = "0.25",
) -> None:
    frame, event_idx = _dip_evidence_frame(score, metric)
    if frame.empty:
        _plot_missing_panel(ax, f"No {ylabel} values", title=title)
        return
    jd = pd.to_numeric(frame["JD"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(frame["evidence_value"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(jd) & np.isfinite(values)
    if not mask.any():
        _plot_missing_panel(ax, f"No finite {ylabel} values", title=title)
        return
    ax.scatter(jd[mask], values[mask], s=11, color=color, alpha=0.72, linewidths=0)
    idx = event_idx[(event_idx >= 0) & (event_idx < len(frame))]
    if idx.size:
        ev_jd = jd[idx]
        ev_values = values[idx]
        ev_mask = np.isfinite(ev_jd) & np.isfinite(ev_values)
        ax.scatter(
            ev_jd[ev_mask],
            ev_values[ev_mask],
            marker="x",
            s=42,
            color="goldenrod",
            linewidths=1.2,
            label="kept dip points",
            zorder=8,
        )
        ax.legend(frameon=False, fontsize=8, loc="best")
    if threshold is not None and np.isfinite(float(threshold)):
        ax.axhline(float(threshold), color="crimson", linestyle="--", linewidth=0.9, alpha=0.75)
    ax.set_xlabel("JD")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    style_publication_axis(ax)


def _plot_evidence_phase_panel(
    ax,
    score: dict[str, Any],
    *,
    metric: str,
    period_days: float,
    title: str,
    ylabel: str,
    threshold: float | None = None,
    color: str = "0.25",
) -> None:
    period = _finite_metric(period_days)
    if not np.isfinite(period) or period <= 0:
        _plot_missing_panel(ax, "No finite gate-selected period", title=title)
        return
    frame, event_idx = _dip_evidence_frame(score, metric)
    if frame.empty:
        _plot_missing_panel(ax, f"No {ylabel} values", title=title)
        return
    jd = pd.to_numeric(frame["JD"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(frame["evidence_value"], errors="coerce").to_numpy(dtype=float)
    finite_jd = jd[np.isfinite(jd)]
    if finite_jd.size == 0:
        _plot_missing_panel(ax, "No finite phase times", title=title)
        return
    epoch = float(np.min(finite_jd))
    phase = np.mod((jd - epoch) / period, 1.0)
    mask = np.isfinite(phase) & np.isfinite(values)
    if not mask.any():
        _plot_missing_panel(ax, f"No finite {ylabel} values", title=title)
        return
    ax.scatter(phase[mask], values[mask], s=11, color=color, alpha=0.72, linewidths=0)
    ax.scatter(phase[mask] + 1.0, values[mask], s=11, color=color, alpha=0.45, linewidths=0)
    idx = event_idx[(event_idx >= 0) & (event_idx < len(frame))]
    if idx.size:
        ev_phase = phase[idx]
        ev_values = values[idx]
        ev_mask = np.isfinite(ev_phase) & np.isfinite(ev_values)
        ax.scatter(
            ev_phase[ev_mask],
            ev_values[ev_mask],
            marker="x",
            s=42,
            color="goldenrod",
            linewidths=1.2,
            label="kept dip points",
            zorder=8,
        )
        ax.scatter(
            ev_phase[ev_mask] + 1.0,
            ev_values[ev_mask],
            marker="x",
            s=42,
            color="goldenrod",
            linewidths=1.2,
            zorder=8,
        )
        ax.legend(frameon=False, fontsize=8, loc="best")
    if threshold is not None and np.isfinite(float(threshold)):
        ax.axhline(float(threshold), color="crimson", linestyle="--", linewidth=0.9, alpha=0.75)
    for x in (0.0, 1.0, 2.0):
        ax.axvline(x, color="0.45", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.set_xlim(0.0, 2.0)
    ax.set_xlabel("Phase")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    style_publication_axis(ax)


def _scatter_rejected_points(ax, rejected: pd.DataFrame, *, label: str, color: str, marker: str) -> None:
    if rejected.empty or "JD" not in rejected.columns or "mag" not in rejected.columns:
        return
    jd = pd.to_numeric(rejected["JD"], errors="coerce")
    mag = pd.to_numeric(rejected["mag"], errors="coerce")
    mask = np.isfinite(jd) & np.isfinite(mag)
    if not mask.any():
        return
    ax.scatter(
        jd.loc[mask],
        mag.loc[mask],
        marker=marker,
        s=32,
        color=color,
        linewidths=0.9,
        alpha=0.9,
        label=label,
        zorder=8,
    )


def _highlight_event_indices(
    ax,
    df: pd.DataFrame,
    y_values: pd.Series | np.ndarray,
    indices: Sequence[int],
    *,
    label: str,
) -> None:
    idx = np.asarray(indices, dtype=int)
    y_array = np.asarray(y_values, dtype=float)
    idx = idx[(idx >= 0) & (idx < len(df)) & (idx < y_array.size)]
    if idx.size == 0 or "JD" not in df.columns:
        return
    jd = pd.to_numeric(df.iloc[idx]["JD"], errors="coerce").to_numpy(dtype=float)
    y = y_array[idx]
    mask = np.isfinite(jd) & np.isfinite(y)
    if not mask.any():
        return
    ax.scatter(
        jd[mask],
        y[mask],
        marker="x",
        s=48,
        color="goldenrod",
        linewidths=1.2,
        label=label,
        zorder=9,
    )


def plot_gate_processing_trial_diagnostic(
    run: BenchmarkRun,
    trial_id: int,
    *,
    controls: dict[str, pd.DataFrame] | None = None,
    example_kind: str | None = None,
    ax: np.ndarray | None = None,
) -> np.ndarray:
    """Plot one injected light curve through rejection, GP subtraction, and gate routing."""
    rows = run.trial_results[run.trial_results["trial_id"].astype(int).eq(int(trial_id))]
    if rows.empty:
        raise KeyError(f"trial_id {trial_id} not found")
    row = rows.iloc[0]
    df_trial = _load_trial_lightcurve_for_row(row, controls=controls)
    frames = _prepare_gate_processing_frames(df_trial, run.config)
    branch_scores = _score_processing_branches(frames["cleaned"], row, run.config)
    stochastic_score = branch_scores["stochastic"]
    phase_score = branch_scores["phase"]
    selected_period = _finite_metric(row.get("pre_periodicity_selected_period", np.nan))

    if ax is None:
        _, ax = plt.subplots(5, 2, figsize=(15.2, 16.8), squeeze=False)
    axes = np.asarray(ax)
    if axes.size < 10:
        raise ValueError("plot_gate_processing_trial_diagnostic requires at least 10 axes")
    axes = axes.reshape(-1)[:10].reshape(5, 2)
    (
        ax_raw,
        ax_gp,
        ax_stoch_base,
        ax_phase_base,
        ax_stoch_resid,
        ax_phase_resid,
        ax_stoch_prob,
        ax_phase_prob,
        ax_stoch_logbf,
        ax_phase_logbf,
    ) = axes.flat

    group_by = "camera" if "camera#" in frames["raw"].columns else "none"
    plot_lightcurve_panel(
        ax_raw,
        frames["raw"],
        group_by=group_by,
        camera_col="camera#" if "camera#" in frames["raw"].columns else None,
        show_errorbars=False,
        marker_size=2.6,
        legend="none",
        time_offset="none",
        xlabel="JD",
        ylabel="mag",
    )
    _scatter_rejected_points(
        ax_raw,
        frames["bad_camera_rejected"],
        label="rejected camera",
        color="crimson",
        marker="x",
    )
    _scatter_rejected_points(
        ax_raw,
        frames["bad_point_rejected"],
        label="rejected point",
        color="black",
        marker="+",
    )
    if frames["bad_camera_rejected"].empty and frames["bad_point_rejected"].empty:
        ax_raw.set_title("Raw injected light curve; no points rejected", fontsize=10)
    else:
        ax_raw.set_title(
            f"Raw injected light curve; rejected {len(frames['raw']) - len(frames['cleaned'])}/{len(frames['raw'])}",
            fontsize=10,
        )
        ax_raw.legend(frameon=False, fontsize=8, loc="best")

    if frames["cleaned"].empty or frames["gp_base"].empty:
        _plot_missing_panel(ax_gp, "No cleaned points for GP baseline", title="Cleaned pregate curve")
    else:
        plot_lightcurve_panel(
            ax_gp,
            frames["cleaned"],
            group_by=group_by,
            camera_col="camera#" if "camera#" in frames["cleaned"].columns else None,
            show_errorbars=False,
            marker_size=2.7,
            legend="none",
            time_offset="none",
            xlabel="JD",
            ylabel="mag",
            baseline=frames["gp_base"],
            baseline_col="baseline",
            baseline_time_col="JD",
            baseline_group_col="camera#" if "camera#" in frames["gp_base"].columns else None,
            baseline_label="masked GP baseline",
            baseline_style={"linewidth": 1.15, "alpha": 0.9},
        )
        ax_gp.set_title("Cleaned curve with masked per-camera GP baseline", fontsize=10)

    _plot_branch_baseline_panel(
        ax_stoch_base,
        stochastic_score,
        title="Stochastic/non-periodic branch baseline",
        color="crimson",
    )
    _plot_branch_baseline_panel(
        ax_phase_base,
        phase_score,
        title="Phase-template branch baseline",
        color="seagreen",
    )
    _plot_branch_residual_time_panel(
        ax_stoch_resid,
        stochastic_score,
        title="Stochastic residuals vs time",
    )
    _plot_branch_residual_phase_panel(
        ax_phase_resid,
        phase_score,
        period_days=selected_period,
        title=f"Phase-template residuals vs phase (P={_fmt_metric(selected_period)} d)",
    )

    posterior_threshold = posterior_probability_threshold(run.config.significance_threshold)
    _plot_evidence_time_panel(
        ax_stoch_prob,
        stochastic_score,
        metric="event_probability",
        title="Stochastic LOO posterior vs time",
        ylabel="LOO P(dip)",
        threshold=posterior_threshold,
        color="#4c78a8",
    )
    _plot_evidence_phase_panel(
        ax_phase_prob,
        phase_score,
        metric="event_probability",
        period_days=selected_period,
        title="Phase-template LOO posterior vs phase",
        ylabel="LOO P(dip)",
        threshold=posterior_threshold,
        color="#4c78a8",
    )
    _plot_evidence_time_panel(
        ax_stoch_logbf,
        stochastic_score,
        metric="log_bf_local",
        title="Stochastic local log BF vs time",
        ylabel="local log BF",
        threshold=run.config.logbf_threshold_dip,
        color="#7f3c8d",
    )
    _plot_evidence_phase_panel(
        ax_phase_logbf,
        phase_score,
        metric="log_bf_local",
        period_days=selected_period,
        title="Phase-template local log BF vs phase",
        ylabel="local log BF",
        threshold=run.config.logbf_threshold_dip,
        color="#7f3c8d",
    )

    bad_cameras = sorted(str(cam) for cam in frames["bad_cameras"])
    bad_camera_text = ",".join(bad_cameras) if bad_cameras else "none"
    branch = "periodic" if _as_bool(row.get("pre_periodic_flag", False)) else "non-periodic"
    example_text = example_kind if example_kind is not None else str(row.get("example_kind", ""))
    title = (
        f"trial {int(row['trial_id'])} | {row['class_name']} | {example_text}\n"
        f"gate target={row['target_gate_label']} -> {row['pre_periodicity_label']} ({branch} branch), "
        f"reason={row.get('pre_periodicity_reason', '')}; "
        f"CE S/N={_fmt_metric(row.get('pre_periodicity_score'))}, "
        f"scatter ratio={_fmt_metric(row.get('pre_periodicity_scatter_ratio'))}, "
        f"phase peak S/N={_fmt_metric(row.get('pre_periodicity_phase_peak_snr'))}; "
        f"bad cameras={bad_camera_text}"
    )
    axes.flat[0].figure.suptitle(title, fontsize=11, y=0.995)
    axes.flat[0].figure.tight_layout(rect=(0, 0, 1, 0.95))
    return axes


def save_gate_processing_visualizations(
    run: BenchmarkRun,
    *,
    output_dir: Path | None = None,
    per_bucket: int = 2,
    max_examples: int = 14,
    controls: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Save publication-style per-trial diagnostics for selected gate examples."""
    output_root = (
        Path(output_dir)
        if output_dir is not None
        else run.run_dir / "plots" / "gate_processing_examples"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    examples = select_gate_processing_examples(
        run.trial_results,
        per_bucket=per_bucket,
        max_examples=max_examples,
    )
    rows: list[dict[str, Any]] = []
    with publication_style_context():
        for _, row in examples.iterrows():
            trial_id = int(row["trial_id"])
            fig, axes = plt.subplots(5, 2, figsize=(15.2, 16.8), squeeze=False)
            plot_gate_processing_trial_diagnostic(
                run,
                trial_id,
                controls=controls,
                example_kind=str(row.get("example_kind", "")),
                ax=axes,
            )
            safe_kind = str(row.get("example_kind", "example")).replace(" ", "_").replace("/", "_")
            out_path = output_root / f"trial_{trial_id:05d}_{safe_kind}.png"
            fig.savefig(out_path, dpi=220)
            plt.close(fig)
            record = {
                "trial_id": trial_id,
                "example_kind": row.get("example_kind", ""),
                "class_name": row.get("class_name", ""),
                "target_gate_label": row.get("target_gate_label", ""),
                "pre_periodicity_label": row.get("pre_periodicity_label", ""),
                "routed_branch": row.get("routed_branch", ""),
                "selected_period_days": row.get("pre_periodicity_selected_period", np.nan),
                "pre_periodicity_score": row.get("pre_periodicity_score", np.nan),
                "pre_periodicity_scatter_ratio": row.get("pre_periodicity_scatter_ratio", np.nan),
                "standard_detected": row.get("standard_detected", False),
                "phase_folded_detected": row.get("phase_folded_detected", False),
                "bifurcated_detected": row.get("bifurcated_detected", False),
                "figure_path": str(out_path),
            }
            rows.append(record)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "manifest.csv", index=False)
    return summary


def save_benchmark_plots(
    results: pd.DataFrame,
    threshold_sweep: pd.DataFrame,
    *,
    run_dir: Path,
    generated_lightcurves: dict[int, pd.DataFrame] | None = None,
    controls: dict[str, pd.DataFrame] | None = None,
    post_filter_results: pd.DataFrame | None = None,
) -> None:
    plots_dir = Path(run_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_recovery_by_class(results, plots_dir / "recovery_by_class_pipeline.png")
    _plot_gate_confusion(results, plots_dir / "gate_routing_confusion.png")
    _plot_recovery_vs_parameters(results, plots_dir / "recovery_vs_injection_parameters.png")
    _plot_threshold_heatmaps(threshold_sweep, plots_dir)
    _plot_pareto(threshold_sweep, plots_dir / "gate_sweep_pareto.png")
    if generated_lightcurves:
        _plot_representative_examples(results, generated_lightcurves, plots_dir / "representative_gate_examples.png")
    if controls:
        _plot_standard_no_injection_false_positives(
            results,
            controls,
            plots_dir / "standard_no_injection_false_positives.png",
            post_filter_results=post_filter_results,
        )


def _plot_recovery_by_class(results: pd.DataFrame, output_path: Path) -> None:
    pipelines = ["standard_detected", "phase_folded_detected", "bifurcated_detected"]
    plot_rows = []
    for class_name, group in results.groupby("class_name", sort=False):
        for pipeline in pipelines:
            plot_rows.append(
                {
                    "class_name": class_name,
                    "pipeline": pipeline.replace("_detected", ""),
                    "rate": _mean_bool(group[pipeline]),
                }
            )
    df = pd.DataFrame(plot_rows)
    pivot = df.pivot(index="class_name", columns="pipeline", values="rate").reindex(BENCHMARK_CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(13, 6))
    pivot.plot.bar(ax=ax)
    ax.set_ylabel("Detection fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Detection by Injected Class and Pipeline Configuration")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_gate_confusion(results: pd.DataFrame, output_path: Path) -> None:
    hard = results[results["target_gate_label"].isin(["periodic", "non_periodic"])].copy()
    fig, ax = plt.subplots(figsize=(6, 5))
    if hard.empty:
        ax.text(0.5, 0.5, "No hard-labeled gate rows", ha="center", va="center")
    else:
        tab = pd.crosstab(hard["target_gate_label"], hard["pre_periodicity_label"])
        im = ax.imshow(tab.to_numpy(dtype=float), cmap="Blues")
        ax.set_xticks(range(len(tab.columns)), labels=tab.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(tab.index)), labels=tab.index)
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                ax.text(j, i, str(int(tab.iloc[i, j])), ha="center", va="center")
        fig.colorbar(im, ax=ax, label="count")
    ax.set_title("Pre-periodicity Gate Routing Confusion")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _bin_rate(df: pd.DataFrame, column: str, detected_col: str, bins: int = 8, log: bool = False) -> pd.DataFrame:
    values = pd.to_numeric(df[column], errors="coerce")
    mask = np.isfinite(values)
    if log:
        mask &= values > 0
    work = df.loc[mask, [column, detected_col]].copy()
    if work.empty:
        return pd.DataFrame(columns=["center", "rate"])
    vals = pd.to_numeric(work[column], errors="coerce")
    if log:
        edges = np.logspace(np.log10(vals.min()), np.log10(vals.max()), bins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        edges = np.linspace(vals.min(), vals.max(), bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
    rates = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        bin_mask = (vals >= lo) & (vals < hi)
        rates.append(_mean_bool(work.loc[bin_mask, detected_col]))
    return pd.DataFrame({"center": centers, "rate": rates})


def _plot_recovery_vs_parameters(results: pd.DataFrame, output_path: Path) -> None:
    target = results[results["target_dip"].fillna(False)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [
        ("amplitude", False, "Amplitude (mag)"),
        ("duration", True, "Duration (days)"),
        ("period_days", True, "Injected period (days)"),
        ("control_median_mag", False, "Median magnitude"),
    ]
    for ax, (column, log, xlabel) in zip(axes.flat, specs, strict=True):
        binned = _bin_rate(target, column, "bifurcated_detected", bins=8, log=log)
        if not binned.empty:
            ax.plot(binned["center"], binned["rate"], marker="o")
        if log:
            ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Bifurcated recovery")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_threshold_heatmaps(threshold_sweep: pd.DataFrame, plots_dir: Path) -> None:
    all_rows = threshold_sweep[
        (threshold_sweep["scope"] == "all")
        & np.isclose(threshold_sweep["phase_peak_snr_min"], PRE_PERIODICITY_PHASE_PEAK_SNR_MIN)
    ].copy()
    for metric, label in [
        ("target_recovery", "Target recovery"),
        ("gate_false_periodic_route_rate", "False periodic route rate"),
        ("periodic_branch_load", "Periodic branch load"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        if all_rows.empty:
            ax.text(0.5, 0.5, "No threshold rows", ha="center", va="center")
        else:
            pivot = all_rows.pivot(index="ce_snr_threshold", columns="scatter_ratio_max", values=metric)
            im = ax.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(pivot.columns)), labels=[f"{x:g}" for x in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), labels=[f"{x:g}" for x in pivot.index])
            fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("scatter_ratio_max")
        ax.set_ylabel("ce_snr_threshold")
        ax.set_title(f"Gate Threshold Sweep: {label}")
        fig.tight_layout()
        fig.savefig(plots_dir / f"threshold_heatmap_{metric}.png", dpi=180)
        plt.close(fig)


def _plot_pareto(threshold_sweep: pd.DataFrame, output_path: Path) -> None:
    all_rows = threshold_sweep[threshold_sweep["scope"] == "all"].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    if all_rows.empty:
        ax.text(0.5, 0.5, "No threshold rows", ha="center", va="center")
    else:
        scatter = ax.scatter(
            all_rows["periodic_branch_load"],
            all_rows["target_recovery"],
            c=all_rows["false_positive_rate"],
            s=40,
            cmap="magma",
            alpha=0.85,
        )
        fig.colorbar(scatter, ax=ax, label="False positive rate")
    ax.set_xlabel("Periodic branch load")
    ax.set_ylabel("Target recovery")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)
    ax.set_title("Gate Sweep Pareto: Recovery vs Branch Load")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _select_representative_rows(results: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    periodic_miss = results[
        results["target_gate_label"].eq("periodic")
        & ~results["pre_periodic_flag"].fillna(False).astype(bool)
    ].head(2)
    if not periodic_miss.empty:
        frames.append(periodic_miss.assign(example_kind="periodic routed stochastic"))
    nonperiodic_false = results[
        results["target_gate_label"].eq("non_periodic")
        & results["pre_periodic_flag"].fillna(False).astype(bool)
    ].head(2)
    if not nonperiodic_false.empty:
        frames.append(nonperiodic_false.assign(example_kind="non-periodic routed periodic"))
    target_miss = results[
        results["target_dip"].fillna(False).astype(bool)
        & ~results["bifurcated_detected"].fillna(False).astype(bool)
    ].head(2)
    if not target_miss.empty:
        frames.append(target_miss.assign(example_kind="target not recovered"))
    if not frames:
        frames.append(results.head(n).assign(example_kind="representative"))
    return pd.concat(frames, ignore_index=True).head(n)


def _plot_representative_examples(
    results: pd.DataFrame,
    generated_lightcurves: dict[int, pd.DataFrame],
    output_path: Path,
) -> None:
    examples = _select_representative_rows(results, n=6)
    fig, axes = plt.subplots(len(examples), 2, figsize=(13, 3.2 * len(examples)), squeeze=False)
    for row_idx, (_, row) in enumerate(examples.iterrows()):
        trial_id = int(row["trial_id"])
        df = generated_lightcurves.get(trial_id)
        if df is None or df.empty:
            continue
        ax_raw, ax_phase = axes[row_idx]
        plot_lightcurve_panel(
            ax_raw,
            df,
            group_by="band" if "v_g_band" in df.columns else "none",
            show_errorbars=False,
            marker_size=2.8,
            legend="none",
            time_offset="none",
            xlabel="JD",
            ylabel="mag",
        )
        ax_raw.set_title(
            f"{row.get('example_kind', '')}: {row['class_name']} | gate={row['pre_periodicity_label']}",
            loc="left",
            fontsize=10,
        )
        period = float(row.get("pre_periodicity_selected_period", np.nan))
        if np.isfinite(period) and period > 0:
            try:
                plot_phase_panel(
                    ax_phase,
                    df,
                    period_days=period,
                    align_v_to_g=True,
                    group_by="band" if "v_g_band" in df.columns else "none",
                    show_errorbars=False,
                    marker_size=2.8,
                    legend="none",
                    ylabel="aligned mag",
                )
                ax_phase.set_title(f"Selected period = {period:.4g} d", fontsize=10)
            except Exception as exc:
                ax_phase.text(0.5, 0.5, f"phase plot failed: {exc}", ha="center", va="center")
        else:
            ax_phase.text(0.5, 0.5, "No selected period", ha="center", va="center")
            ax_phase.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_standard_no_injection_false_positives(
    results: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
    output_path: Path,
    *,
    post_filter_results: pd.DataFrame | None = None,
    max_examples: int = 12,
) -> None:
    fp = results[
        results["class_name"].eq("no_injection_control")
        & results["standard_detected"].fillna(False).astype(bool)
    ].copy()
    if fp.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No standard_only no-injection false positives", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    examples = fp.sample(n=min(int(max_examples), len(fp)), random_state=20260513)
    n = len(examples)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.4 * nrows), squeeze=False)

    post_lookup: dict[int, bool] = {}
    if post_filter_results is not None and not post_filter_results.empty:
        sub = post_filter_results[post_filter_results["pipeline"].eq("standard_only")]
        post_lookup = dict(
            zip(
                sub["trial_id"].astype(int),
                sub["post_filter_passed"].fillna(False).astype(bool),
                strict=False,
            )
        )

    for ax, (_, row) in zip(axes.flat, examples.iterrows(), strict=False):
        trial_id = int(row["trial_id"])
        base = controls.get(str(row["source_path"]))
        if base is None:
            ax.text(0.5, 0.5, f"Missing light curve for trial {trial_id}", ha="center", va="center")
            ax.set_axis_off()
            continue
        df, _ = generate_trial_lightcurve(base, row)
        plot_lightcurve_panel(
            ax,
            df,
            group_by="band" if "v_g_band" in df.columns else "none",
            show_errorbars=False,
            marker_size=2.6,
            legend="none",
            time_offset="none",
            xlabel="JD",
            ylabel="mag",
        )
        post_text = "post-pass" if bool(post_lookup.get(trial_id, False)) else "post-reject"
        ax.set_title(
            f"trial {trial_id} | BF={row.get('standard_dip_bayes_factor', np.nan):.3g} | {post_text}",
            fontsize=9,
            loc="left",
        )

    for ax in axes.flat[n:]:
        ax.set_axis_off()

    fig.suptitle("Random standard_only false positives on no-injection controls", y=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
