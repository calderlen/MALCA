"""
Injection-recovery testing for dipper detection pipeline.

Implements approach similar to ZTF paper Section 3.5:
1. Select control sample of clean light curves
2. Inject synthetic dips using skew-normal model
3. Run through detection pipeline
4. Measure detection efficiency vs amplitude/duration

This validates the completeness and contamination of the pipeline
and characterizes sensitivity to different dip morphologies.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json

from scipy.stats import norm, skewnorm, skew
from scipy.optimize import minimize_scalar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from malca.plotting.lightcurve_publication import (
    apply_publication_rcparams,
    finalize_publication_figure,
    save_publication_figure,
    FIG_SINGLE_COL_HEATMAP,
    FIG_TWO_COL_STANDARD,
    scaled_publication_text_sizes,
)

apply_publication_rcparams(plt)
import plotly.graph_objects as go

from malca.core.baseline import (
    global_median_baseline,
    per_camera_median_baseline,
    per_camera_gp_baseline,
    per_camera_gp_baseline_masked,
)
from malca.cli_config import add_config_args, namespace_keys, parse_args_with_config
from malca.config import INJECTION_CHUNK_SIZE
from malca.config import (
    WORKERS,
    TRIGGER_MODE,
    LOGBF_THRESHOLD_DIP,
    LOGBF_THRESHOLD_JUMP,
    SIGNIFICANCE_THRESHOLD,
    P_POINTS,
    MAG_POINTS,
    MIN_MAG_OFFSET,
    RUN_MIN_POINTS,
    RUN_MAX_GAP_POINTS,
    BASELINE_FUNC,
    BASELINE_S0,
    BASELINE_W0,
    BASELINE_Q,
    BASELINE_JITTER,
    INJECTION_MAG_LO,
    INJECTION_MAG_HI,
    INJECTION_N_SAMPLE,
    INJECTION_TOTAL_TRIALS,
    INJECTION_MIN_POINTS,
    INJECTION_SEED,
    INJECTION_MAX_ATTEMPTS,
    DEFAULT_OUTPUT_DIR,
    LIGHT_CURVE_FILE_EXTENSION,
)
from malca.stv.events import score_lightcurve
from malca.core.utils import read_lc_dat2


INJECTION_CONFIG_DEFAULTS = {
    "trigger_mode": TRIGGER_MODE,
    "logbf_threshold_dip": LOGBF_THRESHOLD_DIP,
    "logbf_threshold_jump": LOGBF_THRESHOLD_JUMP,
    "significance_threshold": SIGNIFICANCE_THRESHOLD,
    "p_points": P_POINTS,
    "mag_points": MAG_POINTS,
    "p_min_dip": None,
    "p_max_dip": None,
    "p_min_jump": None,
    "p_max_jump": None,
    "run_min_points": RUN_MIN_POINTS,
    "run_max_gap_points": RUN_MAX_GAP_POINTS,
    "run_max_gap_days": None,
    "run_min_duration_days": 0.0,
    "baseline_func": BASELINE_FUNC,
    "baseline_s0": BASELINE_S0,
    "baseline_w0": BASELINE_W0,
    "baseline_q": BASELINE_Q,
    "baseline_jitter": BASELINE_JITTER,
    "baseline_sigma_floor": None,
    "mag_min_dip": None,
    "mag_max_dip": None,
    "mag_min_jump": None,
    "mag_max_jump": None,
    "no_event_prob": False,
    "min_mag_offset": MIN_MAG_OFFSET,
    "measure_pre_injection": True,
}








_GLOBAL: dict[str, object] = {}

BASELINE_CHOICES = ("gp", "gp_masked", "global_median", "per_camera_median")
MIN_EFFICIENCY_BIN_TRIALS = 5


def _skewnormal_mode_standardized(skewness: float) -> float:
    result = minimize_scalar(
        lambda x: -float(skewnorm.pdf(x, a=float(skewness))),
        bounds=(-10.0, 10.0),
        method="bounded",
    )
    if not result.success:
        raise RuntimeError("Could not determine skew-normal mode")
    return float(result.x)


def skewnormal_dip(
    t: np.ndarray,
    t_center: float,
    duration: float,
    amplitude: float,
    skewness: float = 0.0,
    offset: float = 0.0,
) -> np.ndarray:
    """
    Generate skew-normal dip profile.

    Parameters
    ----------
    t : np.ndarray
        Time array
    t_center : float
        Center time of dip (mean mu)
    duration : float
        Duration of dip (related to sigma)
    amplitude : float
        Depth of dip in magnitudes
    skewness : float
        Skewness parameter alpha (default 0 = symmetric Gaussian)
        Positive = tail to right, Negative = tail to left
    offset : float
        Baseline offset C0

    Returns
    -------
    np.ndarray
        Magnitude perturbation (positive = fainter)
    """
    if not np.isfinite(duration) or float(duration) <= 0:
        raise ValueError("duration must be positive and finite")
    if not np.isfinite(amplitude) or float(amplitude) < 0:
        raise ValueError("amplitude must be non-negative and finite")
    sigma = float(duration) / 2.355
    # Normalize against the continuous skew-normal mode, never against the
    # maximum sampled data point.  Sample-dependent normalization would force
    # a full-depth point even when cadence misses the peak and erase the window
    # function this experiment is intended to measure.
    mode = _skewnormal_mode_standardized(float(skewness))
    peak_pdf = float(skewnorm.pdf(mode, a=float(skewness)))
    standardized = (np.asarray(t, dtype=float) - float(t_center)) / sigma
    profile = float(amplitude) * skewnorm.pdf(standardized, a=float(skewness)) / peak_pdf
    return profile + float(offset)


def estimate_magnitude_error_polynomial(
    lc_sample: list[pd.DataFrame],
    order: int = 5,
    mag_col: str = "mag",
    err_col: str = "error",
) -> np.poly1d:
    """
    Fit polynomial to approximate mag-dependent errors.
    """
    all_mags = []
    all_errs = []

    for df in lc_sample:
        if df.empty:
            continue
        mag = df[mag_col].values
        err = df[err_col].values
        mask = np.isfinite(mag) & np.isfinite(err) & (err > 0)
        all_mags.extend(mag[mask])
        all_errs.extend(err[mask])

    if len(all_mags) < 10:
        return np.poly1d([0.1])

    coeffs = np.polyfit(all_mags, all_errs, order)
    return np.poly1d(coeffs)


def inject_dip(
    df_lc: pd.DataFrame,
    t_center: float,
    duration: float,
    amplitude: float,
    skewness: float = 0.0,
    mag_err_poly: np.poly1d | None = None,
    rng: np.random.Generator | None = None,
    mag_col: str = "mag",
    time_col: str = "JD",
    err_col: str = "error",
) -> pd.DataFrame:
    """
    Inject synthetic dip into light curve.

    Adds a dip to the *observed* magnitudes, preserving the actual cadence,
    systematics, and measurement-noise realization already present in the
    light curve.

    ``mag_err_poly`` and ``rng`` are retained for API compatibility, but no
    additional random noise is drawn.  Adding another error draw to observed
    photometry would broaden the noise by roughly ``sqrt(2)`` and bias the
    recovery efficiency low.
    """
    df_out = df_lc.copy()
    if df_out.empty:
        return df_out

    t = df_out[time_col].values
    mag_old = df_out[mag_col].values.astype(float)
    dip_profile = skewnormal_dip(t, t_center, duration, amplitude, skewness)
    df_out[mag_col] = mag_old + dip_profile
    return df_out


def deterministic_trial_seed(seed: int, trial_index: int) -> int:
    """Return a stable per-trial seed independent of scheduling and resume order."""
    if int(trial_index) < 0:
        raise ValueError("trial_index must be non-negative")
    # SeedSequence avoids the simple ``seed + trial_index`` collision pattern
    # and defines the random stream solely from immutable trial identity.
    seed_u64 = int(seed) & ((1 << 64) - 1)
    sequence = np.random.SeedSequence(
        [seed_u64 & 0xFFFFFFFF, seed_u64 >> 32, int(trial_index)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def trial_rng(seed: int, trial_index: int) -> np.random.Generator:
    return np.random.default_rng(deterministic_trial_seed(seed, trial_index))


def _jsonable_experiment_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _jsonable_experiment_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable_experiment_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable_experiment_value(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def experiment_fingerprint(config: dict) -> str:
    canonical = json.dumps(
        _jsonable_experiment_value(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _assert_resume_fingerprint(output_path: Path | None, expected: str) -> None:
    if output_path is None or not output_path.exists() or output_path.stat().st_size == 0:
        return
    try:
        existing = pd.read_parquet(output_path, columns=["experiment_fingerprint"])
    except Exception as exc:
        raise ValueError(
            "Existing injection output has no experiment fingerprint and cannot be safely resumed"
        ) from exc
    values = set(existing["experiment_fingerprint"].dropna().astype(str))
    if existing["experiment_fingerprint"].isna().any() or values != {str(expected)}:
        raise ValueError(
            "Existing injection output was produced by a different experiment configuration"
        )


def binomial_confidence_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion.

    Empty samples return ``(None, None)`` rather than a fabricated zero or a
    value interpolated from neighboring bins.
    """
    k = int(successes)
    n = int(trials)
    if n < 0 or k < 0 or k > n:
        raise ValueError("Expected 0 <= successes <= trials")
    if n == 0:
        return None, None
    if not (0.0 < float(confidence) < 1.0):
        raise ValueError("confidence must be between 0 and 1")
    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _efficiency_record(successes: int, trials: int, confidence: float) -> dict:
    low, high = binomial_confidence_interval(successes, trials, confidence=confidence)
    return {
        "successes": int(successes),
        "trials": int(trials),
        "efficiency": (float(successes / trials) if trials else None),
        "ci_low": low,
        "ci_high": high,
        "confidence": float(confidence),
        "ci_method": "wilson_score",
    }


def summarize_injection_efficiency(
    results_df: pd.DataFrame,
    *,
    detected_col: str = "detected",
    confidence: float = 0.95,
) -> dict:
    """Report operational and science efficiency without conflating errors.

    ``end_to_end`` uses every designed trial as its denominator. ``completed``
    excludes processing errors. ``conditional_observable`` additionally
    requires the injected signal to have the recorded minimum cadence support.
    """
    total = int(len(results_df))
    status = results_df.get(
        "trial_status", pd.Series("completed", index=results_df.index, dtype="object")
    ).astype("string").fillna("completed")
    processing_error = status.eq("processing_error").fillna(False)
    if "processing_error" in results_df.columns:
        processing_error |= results_df["processing_error"].fillna(False).astype(bool)
    elif "error" in results_df.columns:
        processing_error |= results_df["error"].notna()

    detected = results_df.get(
        detected_col, pd.Series(False, index=results_df.index, dtype="boolean")
    ).astype("boolean")
    completed_mask = ~processing_error & detected.notna()
    observable = results_df.get(
        "observable", pd.Series(True, index=results_df.index, dtype="boolean")
    ).fillna(False).astype(bool)
    observable_mask = completed_mask & observable

    n_recovered_all = int(detected.fillna(False).sum())
    n_recovered_completed = int(detected.loc[completed_mask].fillna(False).sum())
    n_recovered_observable = int(detected.loc[observable_mask].fillna(False).sum())
    paired_evaluated = results_df.get(
        "paired_control_evaluated", pd.Series(False, index=results_df.index)
    )
    paired_evaluated = int(
        pd.Series(paired_evaluated, index=results_df.index).fillna(False).astype(bool).sum()
    )
    return {
        "total_designed_trials": total,
        "processing_error_trials": int(processing_error.sum()),
        "completed_trials": int(completed_mask.sum()),
        "observable_trials": int(observable_mask.sum()),
        "unobservable_completed_trials": int((completed_mask & ~observable).sum()),
        "paired_control_evaluated_trials": paired_evaluated,
        "end_to_end": _efficiency_record(n_recovered_all, total, confidence),
        "completed": _efficiency_record(
            n_recovered_completed, int(completed_mask.sum()), confidence
        ),
        "conditional_observable": _efficiency_record(
            n_recovered_observable, int(observable_mask.sum()), confidence
        ),
    }


def _baseline_func_from_name(name: str):
    baseline_map = {
        "gp": per_camera_gp_baseline,
        "gp_masked": per_camera_gp_baseline_masked,
        "global_median": global_median_baseline,
        "per_camera_median": per_camera_median_baseline,
    }
    try:
        return baseline_map[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported baseline_func {name!r}; expected one of {', '.join(BASELINE_CHOICES)}"
        ) from exc


def _get_id_col(df: pd.DataFrame) -> str:
    for col in ("asas_sn_id", "source_id", "id"):
        if col in df.columns:
            return col
    raise KeyError("Manifest is missing a usable ID column (expected asas_sn_id/source_id/id).")


def _resolve_lc_path(row: pd.Series) -> Path | None:
    if "dat_path" in row and pd.notna(row["dat_path"]):
        return Path(str(row["dat_path"]))
    if "path" in row and pd.notna(row["path"]):
        return Path(str(row["path"]))
    if "lc_dir" in row and pd.notna(row["lc_dir"]):
        # Fallback to directory only if we must
        return Path(str(row["lc_dir"]))
    return None



def _load_lc(asas_sn_id: str, lc_dir: Path, *, file_ext: str | None = None) -> pd.DataFrame:
    df_g, df_v = read_lc_dat2(asas_sn_id, str(lc_dir), file_ext=file_ext)
    if df_g.empty and df_v.empty:
        return pd.DataFrame()
    return pd.concat([df_g, df_v], ignore_index=True)


def select_control_sample(
    manifest_df: pd.DataFrame,
    n_sample: int = INJECTION_N_SAMPLE,
    reject_candidates: pd.DataFrame | None = None,
    min_points: int = INJECTION_MIN_POINTS,
    seed: int = INJECTION_SEED,
) -> pd.DataFrame:
    """
    Select clean light curves for injection testing.
    """
    df = manifest_df.copy()

    if reject_candidates is not None and "asas_sn_id" in reject_candidates.columns:
        exclude_ids = set(reject_candidates["asas_sn_id"].astype(str))
        df = df[~df["asas_sn_id"].astype(str).isin(exclude_ids)]

    if "n_points" in df.columns:
        df = df[df["n_points"] >= min_points]

    rng = np.random.default_rng(seed)
    if len(df) < n_sample:
        return df.reset_index(drop=True)

    indices = rng.choice(len(df), size=n_sample, replace=False)
    return df.iloc[indices].reset_index(drop=True)


def _build_detection_kwargs(args: argparse.Namespace) -> dict:
    # Build baseline_kwargs from CLI args
    baseline_kwargs = dict(
        S0=args.baseline_s0,
        w0=args.baseline_w0,
        q=args.baseline_q,
        jitter=args.baseline_jitter,
        sigma_floor=args.baseline_sigma_floor,
        add_sigma_eff_col=True,
    )

    # Build mag grids from min/max/points if bounds are provided
    mag_grid_dip = None
    mag_grid_jump = None
    if args.mag_min_dip is not None and args.mag_max_dip is not None:
        mag_grid_dip = np.linspace(args.mag_min_dip, args.mag_max_dip, args.mag_points)
    if args.mag_min_jump is not None and args.mag_max_jump is not None:
        mag_grid_jump = np.linspace(args.mag_min_jump, args.mag_max_jump, args.mag_points)

    return dict(
        trigger_mode=args.trigger_mode,
        logbf_threshold_dip=args.logbf_threshold_dip,
        logbf_threshold_jump=args.logbf_threshold_jump,
        significance_threshold=args.significance_threshold,
        p_points=args.p_points,
        p_min_dip=args.p_min_dip,
        p_max_dip=args.p_max_dip,
        p_min_jump=args.p_min_jump,
        p_max_jump=args.p_max_jump,
        mag_points=args.mag_points,
        mag_grid_dip=mag_grid_dip,
        mag_grid_jump=mag_grid_jump,
        run_min_points=args.run_min_points,
        max_gap_points=args.run_max_gap_points,
        run_max_gap_days=args.run_max_gap_days,
        run_min_duration_days=args.run_min_duration_days,
        compute_event_prob=(not args.no_event_prob),
        baseline_func=_baseline_func_from_name(args.baseline_func),
        baseline_kwargs=baseline_kwargs,
    )


def _default_detection_func(df: pd.DataFrame, detection_kwargs: dict, min_mag_offset: float = 0.0) -> dict:
    res = score_lightcurve(df, **detection_kwargs)
    dip = res["dip"]
    jump = res["jump"]

    baseline_mag = float(dip.get("baseline_mag", jump.get("baseline_mag", np.nan)))
    dip_best_mag_event = float(dip.get("best_mag_event", np.nan))
    jump_best_mag_event = float(jump.get("best_mag_event", np.nan))
    dip_best_t0 = float(dip.get("best_t0", np.nan))
    jump_best_t0 = float(jump.get("best_t0", np.nan))

    # Apply signal amplitude filter if min_mag_offset > 0
    dip_significant = bool(dip["significant"])
    jump_significant = bool(jump["significant"])
    if min_mag_offset > 0 and np.isfinite(baseline_mag):
        dip_diff = abs(dip_best_mag_event - baseline_mag) if np.isfinite(dip_best_mag_event) else 0.0
        jump_diff = abs(jump_best_mag_event - baseline_mag) if np.isfinite(jump_best_mag_event) else 0.0
        if dip_diff <= min_mag_offset:
            dip_significant = False
        if jump_diff <= min_mag_offset:
            jump_significant = False

    return dict(
        detected=dip_significant,
        dip_significant=dip_significant,
        jump_significant=jump_significant,
        dip_bayes_factor=float(dip["bayes_factor"]),
        jump_bayes_factor=float(jump["bayes_factor"]),
        dip_best_p=float(dip["best_p"]),
        jump_best_p=float(jump["best_p"]),
        baseline_mag=baseline_mag,
        dip_best_mag_event=dip_best_mag_event,
        jump_best_mag_event=jump_best_mag_event,
        dip_best_t0=dip_best_t0,
        jump_best_t0=jump_best_t0,
    )


def _processing_error_result(
    trial_index: int,
    trial_seed: int,
    *,
    error_stage: str,
    error: object,
    detected_key: str = "detected",
    **values: object,
) -> dict:
    """Build an error row that cannot be mistaken for a nondetection."""
    return {
        "trial_index": int(trial_index),
        "trial_seed": int(trial_seed),
        "trial_status": "processing_error",
        "processing_error": True,
        "injection_performed": False,
        "observable": False,
        detected_key: None,
        "error_stage": str(error_stage),
        "error": str(error),
        **values,
    }


def _simulate_trial(
    trial_index: int,
    *,
    control_ids: np.ndarray,
    control_dirs: np.ndarray,
    amp_range: tuple[float, float],
    dur_range: tuple[float, float],
    skew_range: tuple[float, float],
    mag_err_poly: np.poly1d | None,
    detection_kwargs: dict,
    min_mag_offset: float,
    measure_pre_injection: bool,
    seed: int,
    file_ext: str | None = None,
    experiment_id: str | None = None,
) -> dict:
    per_trial_seed = deterministic_trial_seed(seed, trial_index)
    rng = np.random.default_rng(per_trial_seed)
    
    # improved random sampling for MC coverage
    # Sample uniformly in fractional depth: 1.0 - 10 ** (-0.4 * amplitude)
    fd_min = 1.0 - 10 ** (-0.4 * amp_range[0])
    fd_max = 1.0 - 10 ** (-0.4 * amp_range[1])
    fd = rng.uniform(fd_min, fd_max)
    # Avoid exact 1.0 which leads to log10(0) -> -inf amplitude
    fd = min(fd, 0.99999999)
    amplitude = -2.5 * np.log10(1.0 - fd)
        
    # Duration: Log-Uniform
    log_dur_min = np.log10(dur_range[0])
    log_dur_max = np.log10(dur_range[1])
    duration = 10 ** rng.uniform(log_dur_min, log_dur_max)

    # Retry loop to find a star with valid magnitude (12-15)
    max_attempts = INJECTION_MAX_ATTEMPTS
    for attempt in range(max_attempts):
        control_idx = int(rng.integers(0, len(control_ids)))
        asas_sn_id = str(control_ids[control_idx])
        lc_dir = Path(str(control_dirs[control_idx]))

        try:
            df = _load_lc(asas_sn_id, lc_dir, file_ext=file_ext)
            if df.empty or len(df) < 10:
                if attempt == max_attempts - 1:
                    return _processing_error_result(
                        trial_index,
                        per_trial_seed,
                        error_stage="load_control",
                        error="empty_or_short_lc_max_retries",
                        experiment_seed=int(seed),
                        experiment_fingerprint=experiment_id,
                        amplitude=float(amplitude),
                        designed_amplitude_mag=float(amplitude),
                        fractional_depth=float(fd),
                        designed_fractional_depth=float(fd),
                        duration=float(duration),
                        designed_duration_days=float(duration),
                        asas_sn_id=asas_sn_id,
                        control_attempts=int(attempt + 1),
                    )
                continue

            median_mag = float(np.nanmedian(df["mag"].values))
            
            # Only inject into stars with median magnitude between 12 and 15
            if median_mag < INJECTION_MAG_LO or median_mag > INJECTION_MAG_HI:
                if attempt == max_attempts - 1:
                     return _processing_error_result(
                        trial_index,
                        per_trial_seed,
                        error_stage="select_control",
                        error="magnitude_out_of_range",
                        experiment_seed=int(seed),
                        experiment_fingerprint=experiment_id,
                        amplitude=float(amplitude),
                        designed_amplitude_mag=float(amplitude),
                        fractional_depth=float(fd),
                        designed_fractional_depth=float(fd),
                        duration=float(duration),
                        designed_duration_days=float(duration),
                        median_mag=median_mag,
                        asas_sn_id=asas_sn_id,
                        control_attempts=int(attempt + 1),
                    )
                continue
            
            # Found a valid star
            break
            
        except Exception as exc:
            if attempt == max_attempts - 1:
                return _processing_error_result(
                    trial_index,
                    per_trial_seed,
                    error_stage="load_control",
                    error=exc,
                    experiment_seed=int(seed),
                    experiment_fingerprint=experiment_id,
                    amplitude=float(amplitude),
                    designed_amplitude_mag=float(amplitude),
                    fractional_depth=float(fd),
                    designed_fractional_depth=float(fd),
                    duration=float(duration),
                    designed_duration_days=float(duration),
                    asas_sn_id=asas_sn_id,
                    control_attempts=int(attempt + 1),
                )
            continue

    injection_performed = False
    error_context: dict[str, object] = {
        "median_mag": median_mag,
        "control_attempts": int(attempt + 1),
        "n_points": int(len(df)),
    }
    try:
        t_min = float(df["JD"].min())
        t_max = float(df["JD"].max())
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            return _processing_error_result(
                trial_index,
                per_trial_seed,
                error_stage="validate_control",
                error="invalid_time_range",
                experiment_seed=int(seed),
                experiment_fingerprint=experiment_id,
                amplitude=float(amplitude),
                designed_amplitude_mag=float(amplitude),
                fractional_depth=float(fd),
                designed_fractional_depth=float(fd),
                duration=float(duration),
                designed_duration_days=float(duration),
                asas_sn_id=asas_sn_id,
                control_attempts=int(attempt + 1),
            )

        # Draw the event center over the actual survey window.  We do not trim
        # the edges to force every injection to be easy to observe; cadence and
        # window losses are part of the end-to-end experiment.
        t_center = rng.uniform(t_min, t_max)
        skewness = rng.uniform(skew_range[0], skew_range[1])
        injected_scale_days = float(duration / 2.355)
        injected_peak_time = float(
            t_center + injected_scale_days * _skewnormal_mode_standardized(skewness)
        )

        times = pd.to_numeric(df["JD"], errors="coerce").to_numpy(dtype=float)
        injected_profile = skewnormal_dip(
            times, t_center, duration, amplitude, skewness
        )
        peak_observed = float(np.nanmax(injected_profile)) if injected_profile.size else 0.0
        observed_peak_fraction = (
            float(peak_observed / amplitude) if amplitude > 0 else np.nan
        )
        fwhm_mask = injected_profile >= (0.5 * amplitude)
        support_mask = injected_profile >= (0.1 * amplitude)
        n_fwhm_points = int(np.count_nonzero(fwhm_mask))
        n_support_points = int(np.count_nonzero(support_mask))
        support_start = t_center - duration
        support_end = t_center + duration
        overlap = max(0.0, min(t_max, support_end) - max(t_min, support_start))
        window_coverage_fraction = float(min(1.0, overlap / (2.0 * duration)))
        observable = bool(n_fwhm_points >= 1 and n_support_points >= 3)
        error_context.update(
            {
                "t_center": float(t_center),
                "designed_t0_jd": float(t_center),
                "designed_location_jd": float(t_center),
                "designed_peak_time_jd": injected_peak_time,
                "skewness": float(skewness),
                "injected_skewness": float(skewness),
                "time_min_jd": t_min,
                "time_max_jd": t_max,
                "time_span_days": float(t_max - t_min),
                "n_fwhm_points": n_fwhm_points,
                "n_support_points": n_support_points,
                "observed_peak_fraction": observed_peak_fraction,
                "window_coverage_fraction": window_coverage_fraction,
                "window_coverage_definition": "fraction_of_location_plusminus_duration_inside_observed_span",
                "observable": observable,
                "observable_definition": "at_least_1_fwhm_point_and_3_points_above_10pct_depth",
            }
        )
        
        # Measure pre-injection detection rate if requested
        pre_injection_result = {
            "paired_control_evaluated": False,
            "pre_injection_detected": None,
        }
        if measure_pre_injection:
            pre_inj = _default_detection_func(
                df,
                detection_kwargs,
                min_mag_offset=min_mag_offset,
            )
            # Prefix all keys with pre_injection_
            pre_injection_result = {f"pre_injection_{k}": v for k, v in pre_inj.items()}
            pre_injection_result["paired_control_evaluated"] = True

        df_injected = inject_dip(
            df,
            t_center,
            duration,
            amplitude,
            skewness,
            mag_err_poly,
            rng=rng,
        )
        injection_performed = True
        error_context.update(
            {
                "injected_amplitude_mag": float(amplitude),
                "injected_fractional_depth": float(fd),
                "injected_duration_days": float(duration),
                "injected_scale_days": injected_scale_days,
                "injected_t0_jd": float(t_center),
                "injected_location_jd": float(t_center),
                "injected_peak_time_jd": injected_peak_time,
                "injected_skewness": float(skewness),
                "duration_parameter_definition": "2.355_times_skewnormal_scale",
            }
        )
        detection_result = _default_detection_func(
            df_injected,
            detection_kwargs,
            min_mag_offset=min_mag_offset,
        )

        # Convert amplitude (mag) to fractional transit depth
        fractional_depth = 1.0 - 10 ** (-0.4 * amplitude)
        pre_detected = pre_injection_result.get("pre_injection_detected")
        post_triggered = bool(detection_result["detected"])
        recovered_t0 = float(detection_result.get("dip_best_t0", np.nan))
        recovery_time_offset_days = (
            float(abs(recovered_t0 - injected_peak_time))
            if np.isfinite(recovered_t0)
            else np.nan
        )
        recovery_time_tolerance_days = float(duration)
        localization_matched = bool(
            post_triggered
            and np.isfinite(recovery_time_offset_days)
            and recovery_time_offset_days <= recovery_time_tolerance_days
        )
        post_detected = localization_matched
        paired_detected = (
            post_detected and not bool(pre_detected)
            if pre_detected is not None
            else post_detected
        )

        return {
            "trial_index": int(trial_index),
            "trial_seed": int(per_trial_seed),
            "experiment_seed": int(seed),
            "experiment_fingerprint": experiment_id,
            "trial_status": "completed",
            "processing_error": False,
            "injection_performed": True,
            "error_stage": None,
            "error": None,
            "amplitude": float(amplitude),
            "designed_amplitude_mag": float(amplitude),
            "injected_amplitude_mag": float(amplitude),
            "fractional_depth": float(fractional_depth),
            "designed_fractional_depth": float(fractional_depth),
            "injected_fractional_depth": float(fractional_depth),
            "duration": float(duration),
            "designed_duration_days": float(duration),
            "injected_duration_days": float(duration),
            "injected_scale_days": injected_scale_days,
            "duration_parameter_definition": "2.355_times_skewnormal_scale",
            "parameter_sampling": "uniform_fractional_depth_log_uniform_duration_uniform_skewness",
            "t0_sampling": "uniform_observed_time_span",
            "skewness": float(skewness),
            "injected_skewness": float(skewness),
            "t_center": float(t_center),
            "designed_t0_jd": float(t_center),
            "designed_location_jd": float(t_center),
            "designed_peak_time_jd": injected_peak_time,
            "injected_t0_jd": float(t_center),
            "injected_location_jd": float(t_center),
            "injected_peak_time_jd": injected_peak_time,
            "median_mag": median_mag,
            "asas_sn_id": asas_sn_id,
            "control_attempts": int(attempt + 1),
            "n_points": int(len(df)),
            "time_min_jd": t_min,
            "time_max_jd": t_max,
            "time_span_days": float(t_max - t_min),
            "n_fwhm_points": n_fwhm_points,
            "n_support_points": n_support_points,
            "observed_peak_fraction": observed_peak_fraction,
            "window_coverage_fraction": window_coverage_fraction,
            "window_coverage_definition": "fraction_of_location_plusminus_duration_inside_observed_span",
            "observable": observable,
            "observable_definition": "at_least_1_fwhm_point_and_3_points_above_10pct_depth",
            **detection_result,
            **pre_injection_result,
            "post_injection_detected": post_detected,
            "post_injection_triggered": post_triggered,
            "post_injection_localization_matched": localization_matched,
            "recovery_time_offset_days": recovery_time_offset_days,
            "recovery_time_tolerance_days": recovery_time_tolerance_days,
            "recovery_definition": "dip_trigger_within_one_duration_of_injected_peak_and_not_control_detection",
            "detected": paired_detected,
            "detected_above_paired_control": paired_detected,
        }
    except Exception as exc:
        return _processing_error_result(
            trial_index,
            per_trial_seed,
            error_stage="inject_or_detect",
            error=exc,
            experiment_seed=int(seed),
            experiment_fingerprint=experiment_id,
            injection_performed=injection_performed,
            amplitude=float(amplitude),
            designed_amplitude_mag=float(amplitude),
            fractional_depth=float(fd),
            designed_fractional_depth=float(fd),
            duration=float(duration),
            designed_duration_days=float(duration),
            asas_sn_id=asas_sn_id,
            **error_context,
        )


def _init_worker(
    control_ids: np.ndarray,
    control_dirs: np.ndarray,
    amp_range: tuple[float, float],
    dur_range: tuple[float, float],
    skew_range: tuple[float, float],
    mag_err_poly: np.poly1d | None,
    detection_kwargs: dict,
    min_mag_offset: float,
    measure_pre_injection: bool,
    seed: int,
    file_ext: str | None = None,
    experiment_id: str | None = None,
) -> None:
    _GLOBAL["control_ids"] = control_ids
    _GLOBAL["control_dirs"] = control_dirs
    _GLOBAL["amp_range"] = amp_range
    _GLOBAL["dur_range"] = dur_range
    _GLOBAL["skew_range"] = skew_range
    _GLOBAL["mag_err_poly"] = mag_err_poly
    _GLOBAL["detection_kwargs"] = detection_kwargs
    _GLOBAL["min_mag_offset"] = min_mag_offset
    _GLOBAL["measure_pre_injection"] = measure_pre_injection
    _GLOBAL["seed"] = seed
    _GLOBAL["file_ext"] = file_ext
    _GLOBAL["experiment_id"] = experiment_id


def _process_trial_batch(trial_indices: list[int]) -> list[dict]:
    results = []
    for trial_index in trial_indices:
        results.append(
            _simulate_trial(
                trial_index,
                control_ids=_GLOBAL["control_ids"],
                control_dirs=_GLOBAL["control_dirs"],
                amp_range=_GLOBAL["amp_range"],
                dur_range=_GLOBAL["dur_range"],
                skew_range=_GLOBAL["skew_range"],
                mag_err_poly=_GLOBAL["mag_err_poly"],
                detection_kwargs=_GLOBAL["detection_kwargs"],
                min_mag_offset=float(_GLOBAL["min_mag_offset"]),
                measure_pre_injection=bool(_GLOBAL["measure_pre_injection"]),
                seed=int(_GLOBAL["seed"]),
                file_ext=_GLOBAL.get("file_ext"),
                experiment_id=_GLOBAL.get("experiment_id"),
            )
        )
    return results



class ParquetAppendWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.columns = None
        if self.path.exists() and self.path.stat().st_size > 0:
            try:
                self.columns = pd.read_parquet(self.path).columns.tolist()
            except Exception:
                self.columns = None

    def write_chunk(self, chunk_results: list[dict]) -> None:
        if not chunk_results:
            return
        df_chunk = pd.DataFrame(chunk_results)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > 0:
            existing = pd.read_parquet(self.path)
            df_chunk = pd.concat([existing, df_chunk], ignore_index=True, sort=False)
        # A first chunk made entirely of errors must not permanently discard
        # science columns appearing in later chunks.  Union the schema and make
        # retries idempotent by trial identity.
        self.columns = list(dict.fromkeys([*(self.columns or []), *df_chunk.columns.tolist()]))
        df_chunk = df_chunk.reindex(columns=self.columns)
        if "trial_index" in df_chunk.columns:
            df_chunk["trial_index"] = pd.to_numeric(df_chunk["trial_index"], errors="raise").astype(int)
            df_chunk = (
                df_chunk.drop_duplicates(subset=["trial_index"], keep="last")
                .sort_values("trial_index", kind="stable")
                .reset_index(drop=True)
            )
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        df_chunk.to_parquet(tmp_path, index=False, compression="zstd")
        tmp_path.replace(self.path)

    def close(self) -> None:
        return


def _write_checkpoint(path: Path, last_index: int) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(str(int(last_index)))
    tmp_path.replace(path)


def _read_checkpoint(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
        if text:
            return int(text)
    except Exception:
        return None
    return None


def _completed_trial_indices(output_path: Path | None) -> set[int]:
    if output_path is None or not output_path.exists() or output_path.stat().st_size == 0:
        return set()
    existing = pd.read_parquet(output_path, columns=["trial_index"])
    values = pd.to_numeric(existing["trial_index"], errors="raise").astype(int)
    if values.duplicated().any():
        raise ValueError(f"Duplicate trial_index values in resumable output: {output_path}")
    return set(values.tolist())



def run_injection_recovery(
    control_sample: pd.DataFrame,
    *,
    detection_kwargs: dict,
    min_mag_offset: float = 0.0,
    measure_pre_injection: bool = True,
    total_trials: int = INJECTION_TOTAL_TRIALS,
    amplitude_range: tuple[float, float] = (0.05, 5.0),
    duration_range: tuple[float, float] = (1.0, 300.0),
    skewness_range: tuple[float, float] = (-0.5, 0.5),
    mag_err_order: int = 5,
    mag_err_sample: int = 100,
    seed: int = INJECTION_SEED,
    workers: int = 1,
    task_size: int = 50,
    checkpoint_interval: int = 100000,
    chunk_size: int = 1000,
    output_path: Path | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = True,
    overwrite: bool = False,
    show_progress: bool = True,
    file_ext: str | None = None,
) -> pd.DataFrame | None:
    """
    Run injection-recovery with optional parallelism and checkpointing.
    Uses Monte Carlo sampling for Amplitude and Duration.
    """
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists() and overwrite:
            output_path.unlink()
        if output_path.exists() and not resume and not overwrite:
            raise SystemExit(f"Output exists: {output_path} (use --overwrite or --no-resume)")
    if overwrite:
        resume = False

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
    elif output_path is not None:
        checkpoint_path = output_path.with_name(f"{output_path.stem}_PROCESSED.txt")

    if checkpoint_path and checkpoint_path.exists() and overwrite:
        checkpoint_path.unlink()

    if int(total_trials) < 1:
        raise ValueError("total_trials must be positive")
    if not (0.0 < float(amplitude_range[0]) <= float(amplitude_range[1])):
        raise ValueError("amplitude_range must be positive and increasing")
    if not (0.0 < float(duration_range[0]) <= float(duration_range[1])):
        raise ValueError("duration_range must be positive and increasing")
    if not (
        np.isfinite(skewness_range[0])
        and np.isfinite(skewness_range[1])
        and float(skewness_range[0]) <= float(skewness_range[1])
    ):
        raise ValueError("skewness_range must be finite and increasing")
    if int(workers) < 1 or int(task_size) < 1 or int(checkpoint_interval) < 1:
        raise ValueError("workers, task_size, and checkpoint_interval must be positive")
    if int(INJECTION_MAX_ATTEMPTS) < 1:
        raise ValueError("INJECTION_MAX_ATTEMPTS must be positive")

    id_col = _get_id_col(control_sample)
    control_ids = control_sample[id_col].astype(str).to_numpy()
    control_dirs = []
    for _, row in control_sample.iterrows():
        lc_dir = _resolve_lc_path(row)
        if lc_dir is None:
            control_dirs.append("")
        else:
            control_dirs.append(str(lc_dir))
    control_dirs = np.asarray(control_dirs, dtype=object)

    if len(control_ids) == 0:
        raise SystemExit("Control sample is empty.")

    # Observed light curves already contain measurement noise.  The historical
    # error-polynomial fit is intentionally not used for injection because a
    # second draw would double-count that uncertainty.
    mag_err_poly = None

    amp_range = (float(amplitude_range[0]), float(amplitude_range[1]))
    dur_range = (float(duration_range[0]), float(duration_range[1]))
    skew_range = (float(skewness_range[0]), float(skewness_range[1]))
    experiment_id = experiment_fingerprint(
        {
            "contract_version": 2,
            "family": "skewnormal_dip",
            "seed": int(seed),
            "amplitude_range": amp_range,
            "duration_range": dur_range,
            "skewness_range": skew_range,
            "control_ids": control_ids,
            "control_paths": control_dirs,
            "detection_kwargs": detection_kwargs,
            "min_mag_offset": float(min_mag_offset),
            "paired_control": bool(measure_pre_injection),
            "file_ext": file_ext,
            "max_attempts": int(INJECTION_MAX_ATTEMPTS),
            "magnitude_limits": [float(INJECTION_MAG_LO), float(INJECTION_MAG_HI)],
        }
    )
    if resume:
        _assert_resume_fingerprint(output_path, experiment_id)
    completed_indices = _completed_trial_indices(output_path) if resume else set()
    invalid_indices = sorted(i for i in completed_indices if i < 0 or i >= int(total_trials))
    if invalid_indices:
        raise ValueError(
            "Resumable output contains trial indices outside the requested design: "
            f"{invalid_indices[:10]}"
        )
    pending_indices = [i for i in range(int(total_trials)) if i not in completed_indices]
    if not pending_indices:
        print("All designed trials are already present in the output table.")
        return None

    writer = ParquetAppendWriter(output_path) if output_path else None
    results: list[dict] = []

    pbar = tqdm(total=total_trials, initial=len(completed_indices), disable=not show_progress)
    
    def flush_results(is_final: bool = False) -> None:
        nonlocal results
        if not results:
            return
        if writer is None:
            return
        writer.write_chunk(results)
        if is_final:
            writer.close()
        results = []

    if workers <= 1:
        for trial_index in pending_indices:
            res = _simulate_trial(
                trial_index,
                control_ids=control_ids,
                control_dirs=control_dirs,
                amp_range=amp_range,
                dur_range=dur_range,
                skew_range=skew_range,
                mag_err_poly=mag_err_poly,
                detection_kwargs=detection_kwargs,
                min_mag_offset=min_mag_offset,
                measure_pre_injection=measure_pre_injection,
                seed=seed,
                file_ext=file_ext,
                experiment_id=experiment_id,
            )
            results.append(res)
            pbar.update(1)
            if chunk_size and len(results) >= chunk_size:
                flush_results()
            if checkpoint_path and (trial_index + 1) % checkpoint_interval == 0:
                flush_results()
                _write_checkpoint(checkpoint_path, trial_index)
        flush_results(is_final=True)
        if checkpoint_path:
            _write_checkpoint(checkpoint_path, total_trials - 1)
        pbar.close()
        if output_path:
            return None
        return pd.DataFrame(results).sort_values("trial_index", kind="stable").reset_index(drop=True)



    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(
            control_ids,
            control_dirs,
            amp_range,
            dur_range,
            skew_range,
            mag_err_poly,
            detection_kwargs,
            min_mag_offset,
            measure_pre_injection,
            seed,
            file_ext,
            experiment_id,
        ),
    ) as ex:
        for offset in range(0, len(pending_indices), checkpoint_interval):
            batch_indices = pending_indices[offset:offset + checkpoint_interval]
            tasks = [batch_indices[i:i + task_size] for i in range(0, len(batch_indices), task_size)]

            futures = {ex.submit(_process_trial_batch, task): task for task in tasks}
            for fut in as_completed(futures):
                batch_results = sorted(fut.result(), key=lambda row: int(row["trial_index"]))
                results.extend(batch_results)
                pbar.update(len(batch_results))
                if chunk_size and len(results) >= chunk_size:
                    flush_results()

            flush_results()
            if checkpoint_path:
                _write_checkpoint(checkpoint_path, max(batch_indices))

    flush_results(is_final=True)
    if checkpoint_path:
        _write_checkpoint(checkpoint_path, total_trials - 1)
    pbar.close()
    if output_path:
        return None
    return pd.DataFrame(results).sort_values("trial_index", kind="stable").reset_index(drop=True)


def compute_quality_metrics(results_df: pd.DataFrame, *, confidence: float = 0.95) -> dict:
    """
    Compute quality metrics from injection-recovery results.
    
    Returns a dict with:
    - total_trials: Total number of trials attempted
    - successful_trials: Trials that completed without error
    - failed_trials: Trials that failed with an error
    - failure_rate: Fraction of trials that failed
    - detection_rate: Fraction of successful trials that detected the injection
    - error_breakdown: Dict mapping error type to count
    - error_percentages: Dict mapping error type to percentage of total
    """
    total = len(results_df)
    if total == 0:
        empty_efficiency = summarize_injection_efficiency(
            results_df, detected_col="detected", confidence=confidence
        )
        return {
            "total_trials": 0,
            "successful_trials": 0,
            "failed_trials": 0,
            "failure_rate": 0.0,
            "detection_rate": 0.0,
            "pre_injection_detection_rate": None,
            "post_injection_detection_rate": None,
            "net_completeness": None,
            "efficiency": empty_efficiency,
            "end_to_end_efficiency": None,
            "end_to_end_ci_low": None,
            "end_to_end_ci_high": None,
            "conditional_observable_efficiency": None,
            "conditional_observable_ci_low": None,
            "conditional_observable_ci_high": None,
            "error_breakdown": {},
            "error_percentages": {},
        }
    
    # Identify processing failures separately from genuine, completed
    # nondetections.  Older result files without trial_status remain readable.
    has_error = results_df.get(
        "processing_error",
        results_df.get("error", pd.Series([None] * total)).notna(),
    )
    has_error = pd.Series(has_error, index=results_df.index).fillna(False).astype(bool)
    failed = has_error.sum()
    successful = total - failed
    
    # Detection rate among successful trials
    if successful > 0:
        detected_col = "detected" if "detected" in results_df.columns else "dip_significant"
        if detected_col in results_df.columns:
            successful_mask = ~has_error
            detection_rate = results_df.loc[successful_mask, detected_col].sum() / successful
        else:
            detection_rate = np.nan
        
        # Pre-injection detection rate if available
        pre_inj_col = "pre_injection_detected"
        if pre_inj_col in results_df.columns:
            pre_values = results_df.loc[successful_mask, pre_inj_col].astype("boolean")
            pre_denominator = int(pre_values.notna().sum())
            pre_inj_rate = (
                float(pre_values.dropna().sum() / pre_denominator)
                if pre_denominator
                else np.nan
            )
            pre_injection_detection_rate = (
                float(pre_inj_rate) if np.isfinite(pre_inj_rate) else None
            )
            if "post_injection_detected" in results_df.columns:
                post_rate = (
                    results_df.loc[successful_mask, "post_injection_detected"].sum()
                    / successful
                )
                post_injection_detection_rate = (
                    float(post_rate) if np.isfinite(post_rate) else None
                )
                # New-schema ``detected`` is already the paired recovery
                # outcome, so it must not have the control rate subtracted a
                # second time.
                net_completeness = (
                    float(detection_rate)
                    if pre_denominator and np.isfinite(detection_rate)
                    else None
                )
            elif np.isfinite(detection_rate) and np.isfinite(pre_inj_rate):
                post_injection_detection_rate = float(detection_rate)
                net_completeness = float(detection_rate - pre_inj_rate)
            else:
                post_injection_detection_rate = None
                net_completeness = None
        else:
            pre_injection_detection_rate = None
            post_injection_detection_rate = None
            net_completeness = None
    else:
        detection_rate = np.nan
        pre_injection_detection_rate = None
        post_injection_detection_rate = None
        net_completeness = None
    
    # Error breakdown
    error_breakdown = {}
    error_percentages = {}
    if "error" in results_df.columns:
        error_counts = results_df["error"].dropna().value_counts()
        for error_type, count in error_counts.items():
            error_breakdown[str(error_type)] = int(count)
            error_percentages[str(error_type)] = float(count / total * 100)
    
    efficiency = summarize_injection_efficiency(
        results_df, detected_col=("detected" if "detected" in results_df.columns else "dip_significant"),
        confidence=confidence,
    )

    return {
        "total_trials": int(total),
        "successful_trials": int(successful),
        "failed_trials": int(failed),
        "failure_rate": float(failed / total) if total > 0 else 0.0,
        "detection_rate": float(detection_rate) if np.isfinite(detection_rate) else None,
        "pre_injection_detection_rate": pre_injection_detection_rate,
        "post_injection_detection_rate": post_injection_detection_rate,
        "net_completeness": net_completeness,
        "efficiency": efficiency,
        "end_to_end_efficiency": efficiency["end_to_end"]["efficiency"],
        "end_to_end_ci_low": efficiency["end_to_end"]["ci_low"],
        "end_to_end_ci_high": efficiency["end_to_end"]["ci_high"],
        "conditional_observable_efficiency": efficiency["conditional_observable"]["efficiency"],
        "conditional_observable_ci_low": efficiency["conditional_observable"]["ci_low"],
        "conditional_observable_ci_high": efficiency["conditional_observable"]["ci_high"],
        "error_breakdown": error_breakdown,
        "error_percentages": error_percentages,
    }


def print_quality_summary(metrics: dict, output_path: Path | None = None) -> None:
    """
    Print a formatted summary of injection-recovery quality metrics.
    Optionally saves to a text file.
    """
    lines = []
    lines.append("")
    lines.append("="*60)
    lines.append("INJECTION-RECOVERY QUALITY METRICS")
    lines.append("="*60)
    
    total = metrics.get("total_trials", 0)
    successful = metrics.get("successful_trials", 0)
    failed = metrics.get("failed_trials", 0)
    detection_rate = metrics.get("detection_rate")
    pre_inj_rate = metrics.get("pre_injection_detection_rate")
    net_compl = metrics.get("net_completeness")
    efficiency = metrics.get("efficiency", {})
    
    success_pct = (successful / total * 100) if total > 0 else 0.0
    fail_pct = (failed / total * 100) if total > 0 else 0.0
    
    lines.append(f"Total trials:      {total:,}")
    lines.append(f"Successful trials: {successful:,} ({success_pct:.1f}%)")
    lines.append(f"Failed trials:     {failed:,} ({fail_pct:.1f}%)")
    
    if detection_rate is not None:
        det_pct = detection_rate * 100
        lines.append(f"Detection rate:    {det_pct:.1f}% (of successful trials)")
    else:
        lines.append("Detection rate:    N/A")

    end_to_end = efficiency.get("end_to_end")
    if end_to_end and end_to_end.get("efficiency") is not None:
        lines.append(
            "End-to-end:       "
            f"{100 * end_to_end['efficiency']:.1f}% "
            f"({100 * end_to_end['confidence']:.0f}% CI {100 * end_to_end['ci_low']:.1f}--{100 * end_to_end['ci_high']:.1f}%; "
            f"{end_to_end['successes']}/{end_to_end['trials']})"
        )
    observable = efficiency.get("conditional_observable")
    if observable and observable.get("efficiency") is not None:
        lines.append(
            "Observable-only:  "
            f"{100 * observable['efficiency']:.1f}% "
            f"({100 * observable['confidence']:.0f}% CI {100 * observable['ci_low']:.1f}--{100 * observable['ci_high']:.1f}%; "
            f"{observable['successes']}/{observable['trials']})"
        )
    
    # Show pre-injection and net completeness if available
    if pre_inj_rate is not None:
        pre_pct = pre_inj_rate * 100
        lines.append(f"Pre-injection rate: {pre_pct:.1f}%")
    if net_compl is not None:
        net_pct = net_compl * 100
        lines.append(f"Net completeness:  {net_pct:.1f}%")
    
    lines.append("")
    
    error_breakdown = metrics.get("error_breakdown", {})
    error_percentages = metrics.get("error_percentages", {})
    if error_breakdown:
        lines.append("Error breakdown:")
        # Sort by count descending
        sorted_errors = sorted(error_breakdown.items(), key=lambda x: -x[1])
        for error_type, count in sorted_errors:
            pct = error_percentages.get(error_type, 0)
            lines.append(f"  {error_type}: {count:,} ({pct:.1f}%)")
    else:
        lines.append("No errors recorded.")
    
    lines.append("=" * 60)
    lines.append("")
    
    summary = "\n".join(lines)
    print(summary)
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(summary)
        print(f"Quality metrics saved to: {output_path}")


def compute_detection_efficiency(
    results_df: pd.DataFrame,
    amplitude_bins: int = 20,
    duration_bins: int = 20,
    detected_col: str = "detected",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute detection efficiency grid.
    """
    details = compute_detection_efficiency_details(
        results_df,
        amplitude_bins=amplitude_bins,
        duration_bins=duration_bins,
        detected_col=detected_col,
    )
    return details["amplitude_centers"], details["duration_centers"], details["efficiency_end_to_end"]


def _safe_linear_edges(values: np.ndarray, bins: int) -> np.ndarray:
    if int(bins) < 1:
        raise ValueError("bins must be positive")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot bin an empty/non-finite parameter column")
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if lo == hi:
        pad = max(abs(lo) * 1e-6, 1e-9)
        lo, hi = lo - pad, hi + pad
    return np.linspace(lo, hi, int(bins) + 1)


def _safe_log_edges(values: np.ndarray, bins: int) -> np.ndarray:
    if int(bins) < 1:
        raise ValueError("bins must be positive")
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        raise ValueError("Cannot log-bin a column without positive finite values")
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if lo == hi:
        factor = 1.0 + 1e-6
        lo, hi = lo / factor, hi * factor
    return np.logspace(np.log10(lo), np.log10(hi), int(bins) + 1)


def _binned_efficiency_arrays(
    sample: np.ndarray,
    detected: pd.Series,
    processing_error: np.ndarray,
    observable: np.ndarray,
    bins: list[np.ndarray],
    *,
    confidence: float = 0.95,
) -> dict:
    finite = np.all(np.isfinite(sample), axis=1)
    detected_bool = detected.fillna(False).astype(bool).to_numpy()
    detected_known = detected.notna().to_numpy()
    completed = finite & ~processing_error & detected_known
    observable_mask = completed & observable

    def hist(mask: np.ndarray) -> np.ndarray:
        return np.histogramdd(sample[mask], bins=bins)[0].astype(int)

    n_designed = hist(finite)
    n_completed = hist(completed)
    n_observable = hist(observable_mask)
    recovered_all = hist(finite & detected_bool)
    recovered_completed = hist(completed & detected_bool)
    recovered_observable = hist(observable_mask & detected_bool)

    def rate(success: np.ndarray, count: np.ndarray) -> np.ndarray:
        out = np.full(count.shape, np.nan, dtype=float)
        np.divide(success, count, out=out, where=count > 0)
        return out

    def intervals(success: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        low = np.full(count.shape, np.nan, dtype=float)
        high = np.full(count.shape, np.nan, dtype=float)
        for index in np.ndindex(count.shape):
            if count[index] > 0:
                lo, hi = binomial_confidence_interval(
                    int(success[index]), int(count[index]), confidence=confidence
                )
                low[index], high[index] = float(lo), float(hi)
        return low, high

    e2e_low, e2e_high = intervals(recovered_all, n_designed)
    obs_low, obs_high = intervals(recovered_observable, n_observable)
    return {
        "n_input_trials": int(len(sample)),
        "n_unbinned_trials": int(len(sample) - finite.sum()),
        "bin_denominator_definition": "all_designed_trials_with_finite_bin_coordinates",
        "n_designed": n_designed,
        "n_completed": n_completed,
        "n_observable": n_observable,
        "n_recovered_end_to_end": recovered_all,
        "n_recovered_completed": recovered_completed,
        "n_recovered_observable": recovered_observable,
        "efficiency_end_to_end": rate(recovered_all, n_designed),
        "efficiency_completed": rate(recovered_completed, n_completed),
        "efficiency_observable": rate(recovered_observable, n_observable),
        "end_to_end_ci_low": e2e_low,
        "end_to_end_ci_high": e2e_high,
        "observable_ci_low": obs_low,
        "observable_ci_high": obs_high,
        "confidence": float(confidence),
        "ci_method": "wilson_score",
    }


def compute_detection_efficiency_details(
    results_df: pd.DataFrame,
    amplitude_bins: int = 20,
    duration_bins: int = 20,
    detected_col: str = "detected",
    confidence: float = 0.95,
) -> dict:
    """Return counts, intervals, and both required efficiency estimands."""
    amp = pd.to_numeric(results_df["amplitude"], errors="coerce").to_numpy(dtype=float)
    duration = pd.to_numeric(results_df["duration"], errors="coerce").to_numpy(dtype=float)
    amp_edges = _safe_linear_edges(amp, amplitude_bins)
    dur_edges = _safe_log_edges(duration, duration_bins)
    detected = results_df[detected_col].astype("boolean")
    processing_error = results_df.get(
        "processing_error", results_df.get("error", pd.Series(None, index=results_df.index)).notna()
    )
    processing_error = pd.Series(processing_error, index=results_df.index).fillna(False).astype(bool).to_numpy()
    observable = results_df.get("observable", pd.Series(True, index=results_df.index))
    observable = pd.Series(observable, index=results_df.index).fillna(False).astype(bool).to_numpy()
    arrays = _binned_efficiency_arrays(
        np.column_stack([amp, duration]),
        detected,
        processing_error,
        observable,
        [amp_edges, dur_edges],
        confidence=confidence,
    )
    return {
        **arrays,
        "amplitude_edges": amp_edges,
        "duration_edges": dur_edges,
        "amplitude_centers": (amp_edges[:-1] + amp_edges[1:]) / 2.0,
        "duration_centers": np.sqrt(dur_edges[:-1] * dur_edges[1:]),
    }


def compute_detection_efficiency_3d(
    results_df: pd.DataFrame,
    depth_bins: int = 20,
    duration_bins: int = 20,
    mag_bins: int = 10,
    detected_col: str = "detected",
) -> dict:
    """
    Compute 3D detection efficiency cube.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from injection-recovery trials with columns:
        fractional_depth, duration, median_mag, detected
    depth_bins : int
        Number of bins for fractional transit depth
    duration_bins : int
        Number of bins for duration (log-spaced)
    mag_bins : int
        Number of bins for median magnitude
    detected_col : str
        Column name for detection boolean

    Returns
    -------
    dict with keys:
        efficiency : np.ndarray, shape (depth_bins, duration_bins, mag_bins)
        depth_centers : np.ndarray
        duration_centers : np.ndarray
        mag_centers : np.ndarray
        depth_edges : np.ndarray
        duration_edges : np.ndarray
        mag_edges : np.ndarray
    """
    if min(int(depth_bins), int(duration_bins), int(mag_bins)) < 1:
        raise ValueError("All efficiency-cube bin counts must be positive")
    depth_edges = np.linspace(0.0, 1.0, depth_bins + 1)
    duration_values = pd.to_numeric(results_df["duration"], errors="coerce").to_numpy(dtype=float)
    mag_values = pd.to_numeric(results_df["median_mag"], errors="coerce").to_numpy(dtype=float)
    dur_edges = _safe_log_edges(duration_values, duration_bins)
    mag_edges = _safe_linear_edges(mag_values, mag_bins)

    depth_centers = (depth_edges[:-1] + depth_edges[1:]) / 2
    dur_centers = np.sqrt(dur_edges[:-1] * dur_edges[1:])
    mag_centers = (mag_edges[:-1] + mag_edges[1:]) / 2

    sample = np.column_stack([
        pd.to_numeric(results_df["fractional_depth"], errors="coerce").to_numpy(dtype=float),
        duration_values,
        mag_values,
    ])
    detected = results_df[detected_col].astype("boolean")
    processing_error = results_df.get(
        "processing_error", results_df.get("error", pd.Series(None, index=results_df.index)).notna()
    )
    processing_error = pd.Series(processing_error, index=results_df.index).fillna(False).astype(bool).to_numpy()
    observable = results_df.get("observable", pd.Series(True, index=results_df.index))
    observable = pd.Series(observable, index=results_df.index).fillna(False).astype(bool).to_numpy()
    arrays = _binned_efficiency_arrays(
        sample,
        detected,
        processing_error,
        observable,
        [depth_edges, dur_edges, mag_edges],
    )

    return dict(
        **arrays,
        # Backwards-compatible plotting alias.  Its estimand is explicit.
        efficiency=arrays["efficiency_end_to_end"],
        efficiency_estimand="end_to_end",
        depth_centers=depth_centers,
        duration_centers=dur_centers,
        mag_centers=mag_centers,
        depth_edges=depth_edges,
        duration_edges=dur_edges,
        mag_edges=mag_edges,
    )


def save_efficiency_cube(
    cube_dict: dict,
    output_path: Path | str,
) -> None:
    """
    Save 3D efficiency cube to .npz file.
    """
    serializable = {
        key: value
        for key, value in cube_dict.items()
        if isinstance(value, (np.ndarray, str, int, float, bool, np.number))
    }
    np.savez(output_path, **serializable)


def load_efficiency_cube(input_path: Path | str) -> dict:
    """
    Load 3D efficiency cube from .npz file.
    """
    data = np.load(input_path)
    return {k: data[k] for k in data.files}


def _efficiency_for_plot(cube: dict, *, min_bin_trials: int = MIN_EFFICIENCY_BIN_TRIALS) -> np.ndarray:
    """Mask bins too sparse to support a plotted efficiency measurement."""
    efficiency = np.asarray(cube["efficiency"], dtype=float).copy()
    if "n_designed" in cube:
        counts = np.asarray(cube["n_designed"], dtype=int)
        efficiency[counts < max(1, int(min_bin_trials))] = np.nan
    return efficiency


def _marginal_efficiency_over_axis(cube: dict, axis_index: int) -> np.ndarray:
    if "n_designed" in cube and "n_recovered_end_to_end" in cube:
        counts = np.asarray(cube["n_designed"], dtype=float).sum(axis=axis_index)
        recovered = np.asarray(cube["n_recovered_end_to_end"], dtype=float).sum(axis=axis_index)
        output = np.full(counts.shape, np.nan, dtype=float)
        np.divide(recovered, counts, out=output, where=counts > 0)
        return output
    return np.nanmean(_efficiency_for_plot(cube), axis=axis_index)


def efficiency_grid_depth_timescale(
    cube: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(duration_centers, depth_centers, efficiency_2d)`` marginalized over magnitude."""
    eff = _efficiency_for_plot(cube)
    if eff.ndim == 3:
        eff_2d = _marginal_efficiency_over_axis(cube, 2)
        if "n_designed" in cube:
            marginalized_counts = np.asarray(cube["n_designed"], dtype=int).sum(axis=2)
            eff_2d = np.asarray(eff_2d, dtype=float)
            eff_2d[marginalized_counts < MIN_EFFICIENCY_BIN_TRIALS] = np.nan
    elif eff.ndim == 2:
        eff_2d = eff
    else:
        raise ValueError(f"Unexpected efficiency rank: {eff.ndim}")

    if "duration_centers" in cube:
        dur_centers = np.asarray(cube["duration_centers"], dtype=float)
    elif "dur_centers" in cube:
        dur_centers = np.asarray(cube["dur_centers"], dtype=float)
    else:
        raise KeyError("Missing duration axis in efficiency cube")

    if "depth_centers" in cube:
        depth_centers = np.asarray(cube["depth_centers"], dtype=float)
    elif "amp_centers" in cube:
        amp_centers = np.asarray(cube["amp_centers"], dtype=float)
        depth_centers = 1.0 - np.power(10.0, -0.4 * amp_centers)
    else:
        raise KeyError("Missing depth/amplitude axis in efficiency cube")

    return dur_centers, depth_centers, np.asarray(eff_2d, dtype=float)


def load_efficiency_grid_depth_timescale(
    source: Path | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load or compute the depth–timescale efficiency grid used for completeness overlays."""
    source = Path(source)
    if source.suffix == ".parquet":
        results_df = pd.read_parquet(source)
        if {"fractional_depth", "median_mag", "duration"}.issubset(results_df.columns):
            cube = compute_detection_efficiency_3d(results_df)
            return efficiency_grid_depth_timescale(cube)
        if {"amplitude", "duration"}.issubset(results_df.columns):
            details = compute_detection_efficiency_details(results_df)
            amp_centers = np.asarray(details["amplitude_centers"], dtype=float)
            depth_centers = 1.0 - np.power(10.0, -0.4 * amp_centers)
            return (
                np.asarray(details["duration_centers"], dtype=float),
                depth_centers,
                np.asarray(details["efficiency_end_to_end"], dtype=float),
            )
        raise ValueError(
            f"Unsupported injection results columns in {source}; "
            "expected fractional_depth/median_mag/duration or amplitude/duration"
        )
    if source.suffix == ".npz":
        return efficiency_grid_depth_timescale(load_efficiency_cube(source))
    raise ValueError(f"Unsupported efficiency source: {source}")


def plot_detection_efficiency(
    amp_centers: np.ndarray,
    dur_centers: np.ndarray,
    efficiency_grid: np.ndarray,
    output_path: Path | str | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    xlabel: str = "Duration [days]",
    ylabel: str = "Amplitude [mag]",
    xlog: bool = True,
    cmap: str = "magma",
    show: bool = True,
) -> plt.Figure:
    """
    Plot 2D detection efficiency heatmap.
    """
    figsize = FIG_SINGLE_COL_HEATMAP
    text = scaled_publication_text_sizes(figsize)
    text["label"] = 10.0
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(
        dur_centers,
        amp_centers,
        efficiency_grid,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    if xlog:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=text["label"]*0.75)
    ax.set_ylabel(ylabel, fontsize=text["label"]*0.75)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Detection Efficiency", fontsize=text["colorbar"]*0.75)

    if output_path:
        save_publication_figure(fig, output_path, close=False)
        print(f"Saved to {output_path}")
    elif show:
        finalize_publication_figure(fig)
        plt.show()
    else:
        finalize_publication_figure(fig)

    return fig


def plot_efficiency_mag_slices(
    cube: dict,
    *,
    output_dir: Path | str | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "magma",
    show: bool = False,
) -> list[plt.Figure]:
    """
    Plot 2D efficiency heatmaps for each magnitude bin.

    Parameters
    ----------
    cube : dict
        Efficiency cube from compute_detection_efficiency_3d() or load_efficiency_cube()
    output_dir : Path, optional
        Directory to save plots (one per mag bin)
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    figs = []
    for k, mag in enumerate(cube["mag_centers"]):
        eff_slice = _efficiency_for_plot(cube)[:, :, k]
        out_path = output_dir / f"efficiency_mag_{mag:.2f}.pdf" if output_dir else None

        fig = plot_detection_efficiency(
            cube["depth_centers"],
            cube["duration_centers"],
            eff_slice,
            xlabel="Duration [days]",
            ylabel="Fractional Depth",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            output_path=out_path,
            show=show and not output_dir,
        )
        figs.append(fig)
        if output_dir:
            plt.close(fig)

    return figs


def plot_efficiency_jointplot(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    efficiency_grid: np.ndarray,
    output_path: Path | str | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    xlabel: str = r"$\tau$ [days]",
    ylabel: str = "Fractional Depth",
    xlog: bool = True,
    ylog: bool = False,
    cmap: str = "magma",
    contour_kwargs: dict | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot 2D detection efficiency heatmap with 1D marginalized panels and contour lines.
    """
    figsize = FIG_SINGLE_COL_HEATMAP
    text = scaled_publication_text_sizes(figsize)
    text["label"] = 10.0
    fig, ax = plt.subplots(figsize=figsize)
    
    # Never interpolate or smooth across empty bins: that would turn an
    # unevaluated part of parameter space into an apparent measurement.
    plotted_eff = np.ma.masked_invalid(np.clip(efficiency_grid, vmin, vmax))
    
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        plotted_eff,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        rasterized=True,
    )
    
    # Contours
    if contour_kwargs is None:
        contour_kwargs = {
            "levels": [0.5, 0.9, 0.99],
            "manual1": [(6, 0.6), (20, 0.6)],
            "inline_spacing1": 4,
            "manual2": [(70, 0.5)],
            "inline_spacing2": 8
        }
        
    try:
        cs = ax.contour(x_centers, y_centers, plotted_eff, levels=contour_kwargs["levels"], colors='black', alpha=0.9, linewidths=0.6)
        texts1 = []
        texts2 = []
        if contour_kwargs.get("manual1"):
            texts1 = ax.clabel(cs, inline=True, inline_spacing=contour_kwargs.get("inline_spacing1", 5), fontsize=text["label"]*0.8, fmt='%g', manual=contour_kwargs["manual1"])
        if contour_kwargs.get("manual2"):
            texts2 = ax.clabel(cs, inline=True, inline_spacing=contour_kwargs.get("inline_spacing2", 5), fontsize=text["label"]*0.8, fmt='%g', manual=contour_kwargs["manual2"])
        for t in texts1 + texts2:
            t.set_rotation(0)
    except Exception:
        pass # Ignore if contours fail due to all 0s
    
    divider = make_axes_locatable(ax)
    
    if xlog:
        ax.set_xscale("log")
        ax.set_xlim(left=1.0)
    else:
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        if xlabel == "Median Magnitude":
            ax.set_xlim(15, 12)
        else:
            ax.set_xlim(x_edges.min(), x_edges.max())
            
    if ylog:
        ax.set_yscale("log")
        ax.set_ylim(bottom=1.0)
        
    if "Fractional Depth" in ylabel:
        ylabel = r"$\delta$"
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
        
        # Add a secondary y-axis for magnitude drop (Delta m)
        def delta_to_dm(delta):
            delta_safe = np.clip(delta, 0.0, 0.99999)
            return -2.5 * np.log10(1.0 - delta_safe)
        
        def dm_to_delta(dm):
            return 1.0 - 10.0**(-dm / 2.5)

        secax = ax.secondary_yaxis('right', functions=(delta_to_dm, dm_to_delta))
        dm_ticks = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 5.0])
        secax.set_yticks(dm_ticks)
        secax.set_yticklabels([f"{dm:.1f}" for dm in dm_ticks])
        secax.set_ylabel(r"$\Delta m$ [mag]", fontsize=text["label"])
        secax.tick_params(axis="y", labelsize=text["label"]*0.75)
    elif ylabel == "Median Magnitude":
        import matplotlib.ticker as ticker
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax.set_ylim(15, 12)
        
    ax.set_xlabel(xlabel, fontsize=text["label"])
    ax.set_ylabel(ylabel, fontsize=text["label"])
    ax.tick_params(axis="y", labelleft=True, labelsize=text["label"]*0.75)
    ax.tick_params(axis="x", labelsize=text["label"]*0.75)
    
    # Colorbar
    cax_pad = 0.7 if r"\delta" in ylabel else 0.15
    cax = divider.append_axes("right", size="7%", pad=cax_pad)
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cax.yaxis.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
    import matplotlib.ticker as ticker
    cax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')
    cbar.set_label("Efficiency", fontsize=text["label"], labelpad=8)
    cbar.ax.tick_params(labelsize=text["label"]*0.6)
    
    if output_path:
        save_publication_figure(fig, output_path, close=False)
        print(f"Saved to {output_path}")
    elif show:
        finalize_publication_figure(fig)
        plt.show()
    else:
        finalize_publication_figure(fig)

    return fig


def plot_efficiency_marginalized(
    cube: dict,
    *,
    axis: str = "mag",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "magma",
    contour_kwargs: dict = None,
    output_path: Path | str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot 2D efficiency marginalized (averaged) over one axis.

    Parameters
    ----------
    cube : dict
        Efficiency cube from compute_detection_efficiency_3d() or load_efficiency_cube()
    axis : str
        Axis to marginalize over: "mag", "duration", or "depth"
    """
    ylog = False

    def marginal_efficiency(axis_index: int) -> np.ndarray:
        if "n_designed" in cube and "n_recovered_end_to_end" in cube:
            counts = np.asarray(cube["n_designed"], dtype=float).sum(axis=axis_index)
            recovered = np.asarray(
                cube["n_recovered_end_to_end"], dtype=float
            ).sum(axis=axis_index)
            output = np.full(counts.shape, np.nan, dtype=float)
            np.divide(recovered, counts, out=output, where=counts > 0)
            return output
        # Legacy cubes lack counts; preserve readability without inventing
        # values for wholly empty slices.
        return np.nanmean(_efficiency_for_plot(cube), axis=axis_index)
    
    if axis == "mag":
        eff_2d = marginal_efficiency(2)
        x_centers = cube["duration_centers"]
        y_centers = cube["depth_centers"]
        x_edges = cube["duration_edges"]
        y_edges = cube["depth_edges"]
        xlabel = r"$\tau$ [days]"
        ylabel = "Fractional Depth"
        xlog = True
        contour_kwargs = {
            "levels": [0.5, 0.9, 0.99],
            "manual1": [(6, 0.6), (20, 0.6)],
            "inline_spacing1": 4,
            "manual2": [(70, 0.5)],
            "inline_spacing2": 8
        }
    elif axis == "duration":
        eff_2d = marginal_efficiency(1)
        x_centers = cube["mag_centers"]
        y_centers = cube["depth_centers"]
        x_edges = cube["mag_edges"]
        y_edges = cube["depth_edges"]
        xlabel = "Median Magnitude"
        ylabel = "Fractional Depth"
        xlog = False
        contour_kwargs = {
            "levels": [0.5],
            "manual1": [(13.5, 0.1)],
            "inline_spacing1": 18,
            "manual2": []
        }
    elif axis == "depth":
        eff_2d = marginal_efficiency(0)  # Shape is (duration, mag) -> x=mag, y=duration
        x_centers = cube["mag_centers"]
        y_centers = cube["duration_centers"]
        x_edges = cube["mag_edges"]
        y_edges = cube["duration_edges"]
        xlabel = "Median Magnitude"
        ylabel = r"$\tau$ [days]"
        xlog = False
        ylog = True
        contour_kwargs = {
            "levels": [0.5, 0.9],
            "manual1": [(14.0, 5), (13.5, 20)],
            "inline_spacing1": 6,
            "manual2": []
        }
    else:
        raise ValueError(f"Unknown axis: {axis}. Use 'mag', 'duration', or 'depth'.")

    if "n_designed" in cube:
        marginalized_counts = np.asarray(cube["n_designed"], dtype=int).sum(axis={"mag": 2, "duration": 1, "depth": 0}[axis])
        eff_2d = np.asarray(eff_2d, dtype=float)
        eff_2d[marginalized_counts < MIN_EFFICIENCY_BIN_TRIALS] = np.nan

    return plot_efficiency_jointplot(
        x_centers,
        y_centers,
        x_edges,
        y_edges,
        eff_2d,
        xlabel=xlabel,
        ylabel=ylabel,
        xlog=xlog,
        ylog=ylog,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        contour_kwargs=contour_kwargs,
        output_path=output_path,
        show=show,
    )


def plot_efficiency_threshold_contour(
    cube: dict,
    *,
    threshold: float = 0.5,
    output_path: Path | str | None = None,
    cmap: str = "plasma",
    show: bool = True,
) -> plt.Figure:
    """
    Plot the depth at which efficiency reaches a threshold, for each (duration, mag).

    This answers: "At what depth can we detect N% of dips?"

    Parameters
    ----------
    cube : dict
        Efficiency cube from compute_detection_efficiency_3d() or load_efficiency_cube()
    threshold : float
        Efficiency threshold (0-1), default 0.5 (50%)
    """
    n_dur = len(cube["duration_centers"])
    n_mag = len(cube["mag_centers"])
    depth_at_threshold = np.full((n_dur, n_mag), np.nan)
    plot_efficiency = _efficiency_for_plot(cube)

    for j in range(n_dur):
        for k in range(n_mag):
            eff = plot_efficiency[:, j, k]
            valid = np.isfinite(eff)
            if not valid.any():
                continue
            above = eff >= threshold
            if above.any():
                idx = np.where(above)[0][0]
                depth_at_threshold[j, k] = cube["depth_centers"][idx]

    figsize = FIG_SINGLE_COL_HEATMAP
    text = scaled_publication_text_sizes(figsize)
    text["label"] = 10.0
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(
        cube["mag_centers"],
        cube["duration_centers"],
        depth_at_threshold,
        cmap=cmap,
        shading="auto",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Median Magnitude", fontsize=text["label"]*0.75)
    ax.set_ylabel("Duration [days]", fontsize=text["label"]*0.75)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Fractional Depth", fontsize=text["colorbar"]*0.75)

    if output_path:
        save_publication_figure(fig, output_path, close=False)
        print(f"Saved: {output_path}")
    elif show:
        finalize_publication_figure(fig)
        plt.show()
    else:
        finalize_publication_figure(fig)

    return fig



def plot_efficiency_3d(
    cube: dict,
    *,
    opacity: float = 0.5,
    output_path: Path | str | None = None,
) -> None:
    """
    Create interactive 3D scatter plot using plotly.
    
    Shows detection efficiency as colored points in 3D space.
    """
    depth = cube["depth_centers"]
    duration = cube["duration_centers"]
    mag = cube["mag_centers"]

    D, Du, M = np.meshgrid(depth, duration, mag, indexing="ij")
    
    efficiency_flat = _efficiency_for_plot(cube).flatten()
    D_flat = D.flatten()
    Du_flat = Du.flatten()
    M_flat = M.flatten()
    
    # Filter out NaNs
    valid_mask = np.isfinite(efficiency_flat)
    if not valid_mask.any():
        print("Warning: No valid efficiency data to plot (all NaN)")
        return
    
    n_valid = valid_mask.sum()
    n_total = len(efficiency_flat)
    print(f"Plotting {n_valid}/{n_total} valid data points ({100*n_valid/n_total:.1f}%)")
    
    D_valid = D_flat[valid_mask]
    Du_valid = Du_flat[valid_mask]
    M_valid = M_flat[valid_mask]
    eff_valid = efficiency_flat[valid_mask]

    fig = go.Figure(data=go.Scatter3d(
        x=D_valid,
        y=np.log10(Du_valid),
        z=M_valid,
        mode='markers',
        marker=dict(
            size=5,
            color=eff_valid,
            colorscale='Viridis',
            opacity=opacity,
            colorbar=dict(title="Efficiency"),
            cmin=0,
            cmax=1
        ),
        text=[f"Depth: {d:.3f}<br>Dur: {10**du:.1f}d<br>Mag: {m:.1f}<br>Eff: {e:.2f}" 
              for d, du, m, e in zip(D_valid, np.log10(Du_valid), M_valid, eff_valid)],
        hoverinfo='text'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Fractional Depth",
            yaxis_title="log₁₀(Duration [days])",
            zaxis_title="Median Magnitude",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    if output_path:
        output_path = Path(output_path)
        if output_path.suffix == ".html":
            fig.write_html(str(output_path))
        else:
            fig.write_image(str(output_path))
        print(f"Saved: {output_path}")
    else:
        fig.show()



def plot_efficiency_isosurface(
    cube: dict,
    *,
    isovalue: float = 0.5,
    output_path: Path | str | None = None,
) -> None:
    """
    Plot 3D isosurface at a given efficiency level using plotly.

    Parameters
    ----------
    cube : dict
        Efficiency cube from compute_detection_efficiency_3d() or load_efficiency_cube()
    isovalue : float
        Efficiency value for isosurface (0-1)
    """
    depth = cube["depth_centers"]
    duration = cube["duration_centers"]
    mag = cube["mag_centers"]

    D, Du, M = np.meshgrid(depth, duration, mag, indexing="ij")
    
    # Filter out NaN values for plotly
    efficiency_flat = _efficiency_for_plot(cube).flatten()
    D_flat = D.flatten()
    Du_flat = Du.flatten()
    M_flat = M.flatten()
    
    valid_mask = np.isfinite(efficiency_flat)
    if not valid_mask.any():
        print("Warning: No valid efficiency data to plot (all NaN)")
        return
    
    n_valid = valid_mask.sum()
    n_total = len(efficiency_flat)
    print(f"Plotting {n_valid}/{n_total} valid data points ({100*n_valid/n_total:.1f}%)")
    
    D_valid = D_flat[valid_mask]
    Du_valid = Du_flat[valid_mask]
    M_valid = M_flat[valid_mask]
    eff_valid = efficiency_flat[valid_mask]

    fig = go.Figure(data=go.Isosurface(
        x=D_valid,
        y=np.log10(Du_valid),
        z=M_valid,
        value=eff_valid,
        isomin=isovalue - 0.01,
        isomax=isovalue + 0.01,
        surface_count=1,
        colorscale=[[0, "blue"], [1, "blue"]],
        showscale=False,
        opacity=0.6,
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Fractional Depth",
            yaxis_title="log₁₀(Duration [days])",
            zaxis_title="Median Magnitude",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    if output_path:
        output_path = Path(output_path)
        if output_path.suffix == ".html":
            fig.write_html(str(output_path))
        else:
            fig.write_image(str(output_path))
        print(f"Saved: {output_path}")
    else:
        fig.show()


def plot_efficiency_all(
    cube_or_path: dict | Path | str,
    output_dir: Path | str,
    *,
    thresholds: list[float] | None = None,
    show: bool = False,
) -> None:
    """
    Generate all standard plots for an efficiency cube.

    Parameters
    ----------
    cube_or_path : dict or Path
        Efficiency cube dict or path to .npz file
    output_dir : Path
        Directory to save all plots
    thresholds : list of float, optional
        Efficiency thresholds for contour plots (default: [0.5, 0.9])
    """
    if thresholds is None:
        thresholds = [0.5, 0.9, 0.99]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(cube_or_path, (str, Path)):
        print(f"Loading cube from {cube_or_path}...")
        cube = load_efficiency_cube(cube_or_path)
    else:
        cube = cube_or_path

    print("Generating magnitude slice plots...")
    slices_dir = output_dir / "mag_slices"
    plot_efficiency_mag_slices(cube, output_dir=slices_dir, show=False)

    print("Generating marginalized plots...")
    for axis in ["mag", "duration", "depth"]:
        plot_efficiency_marginalized(
            cube,
            axis=axis,
            output_path=output_dir / f"efficiency_marginalized_{axis}.pdf",
            show=show,
        )
        plt.close()

    print("Generating threshold contour plots...")
    for thresh in thresholds:
        plot_efficiency_threshold_contour(
            cube,
            threshold=thresh,
            output_path=output_dir / f"depth_at_{int(thresh*100)}pct_efficiency.pdf",
            show=show,
        )
        plt.close()

    print("Generating interactive 3D plots...")
    plot_efficiency_3d(
        cube,
        output_path=output_dir / "efficiency_3d_volume.html",
    )
    plot_efficiency_isosurface(
        cube,
        isovalue=0.5,
        output_path=output_dir / "efficiency_50pct_isosurface.html",
    )

    print(f"All plots saved to {output_dir}")


def compute_auxiliary_statistics(df_lc: pd.DataFrame, mag_col: str = "mag") -> dict:
    """
    Compute auxiliary time-series statistics.
    """
    mag = df_lc[mag_col].values
    mag_finite = mag[np.isfinite(mag)]

    if len(mag_finite) < 3:
        return {"skewness": np.nan, "von_neumann_ratio": np.nan}

    skewness_val = skew(mag_finite)
    diff_sq = np.diff(mag_finite) ** 2
    dev_sq = (mag_finite - mag_finite.mean()) ** 2
    von_neumann_ratio = diff_sq.sum() / dev_sq.sum() if dev_sq.sum() > 0 else np.nan
    return {"skewness": skewness_val, "von_neumann_ratio": von_neumann_ratio}


def _get_non_default_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    non_defaults = {}
    for action in parser._actions:
        if action.dest == "help":
            continue
        default = action.default
        value = getattr(args, action.dest, None)
        if value != default:
            non_defaults[action.dest] = value
    return non_defaults


def _generate_output_suffix(non_default_args: dict) -> str:
    ignored_keys = {
        "out_dir",
        "out",
        "cube_out",
        "plot_dir",
        "overwrite",
        "no_resume",
    }
    filtered_args = {k: v for k, v in non_default_args.items() if k not in ignored_keys}
    if not filtered_args:
        return ""

    def format_value(val: object) -> str:
        if isinstance(val, bool):
            return "1" if val else "0"
        if isinstance(val, float):
            return f"{val:.2g}".replace(".", "p")
        if isinstance(val, Path):
            return val.stem[:20]
        if isinstance(val, str):
            return Path(val).stem[:20] if ("/" in val or "\\" in val) else val[:15]
        return str(val)[:15]

    parts = []
    priority_keys = [
        "trigger_mode",
        "logbf_threshold_dip",
        "logbf_threshold_jump",
        "significance_threshold",
        "baseline_func",
        "min_mag_offset",
        "amp_min",
        "amp_max",
        "dur_min",
        "dur_max",
        "total_trials",
        "skew_min",
        "skew_max",
        "mag_err_order",
        "control_sample_size",
        "workers",
    ]

    for key in priority_keys:
        if key in filtered_args:
            val = filtered_args[key]
            short_key = key.replace("threshold_", "thr_").replace("logbf_", "bf_").replace("_", "")
            parts.append(f"{short_key}={format_value(val)}")

    for key, val in filtered_args.items():
        if key not in priority_keys and len(parts) < 8:
            short_key = key.replace("_", "")[:12]
            parts.append(f"{short_key}={format_value(val)}")

    suffix = "_".join(parts)
    if len(suffix) > 150:
        suffix = suffix[:150]
    return suffix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run injection-recovery tests for dip detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Output structure (default --output-dir {DEFAULT_OUTPUT_DIR / 'dip_injection'}):
  {DEFAULT_OUTPUT_DIR / 'dip_injection'}/
    20250121_143052/             # Timestamped run directory
      run_params.json            # Full parameter dump
      results/
        injection_results.parquet  # Trial-by-trial results
        injection_results_PROCESSED.txt  # Checkpoint
      cubes/
        efficiency_cube.npz        # 3D efficiency cube
      plots/
        mag_slices/                # Per-magnitude heatmaps
        efficiency_marginalized_*.png
        depth_at_*pct_efficiency.png
        efficiency_3d_volume.html  # Interactive 3D (if plotly)
    20250121_150318_custom_tag/  # Optional --run-tag appended
      ...
    latest -> 20250121_150318_custom_tag/  # Symlink to latest run

Each run gets a unique timestamped directory. Use --run-tag to append a custom label.
""",
    )
    g_io = parser.add_argument_group("Input / output")
    g_injection = parser.add_argument_group("Injection parameters")
    g_detection = parser.add_argument_group("Detection")
    g_workers = parser.add_argument_group("Workers & chunks")
    g_postprocess = parser.add_argument_group("Postprocess")

    g_io.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "lc_manifest_all.parquet",
                        help=f"Manifest parquet path (default: {DEFAULT_OUTPUT_DIR / 'lc_manifest_all.parquet'})")
    g_io.add_argument("--output-dir", dest="out_dir", type=Path, default=DEFAULT_OUTPUT_DIR / "dip_injection",
                      help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR / 'dip_injection'})")
    g_io.add_argument("--run-tag", type=str, default=None,
                        help="Optional tag to append to run directory name (e.g., 'deep_dips_mag18')")
    g_io.add_argument("--output", type=Path, default=None,
                        help="Override Parquet output path (default: <out-dir>/<timestamp>/results/injection_results.parquet)")
    g_io.add_argument("--file-ext", type=str, default=None,
                        help=f"Light curve file extension (e.g., dat2, dat3). Default: {LIGHT_CURVE_FILE_EXTENSION} (from config).")
    g_injection.add_argument(
        "--control-sample-size",
        dest="control_sample_size",
        type=int,
        default=INJECTION_N_SAMPLE,
        help="Number of control LCs to sample.",
    )
    g_injection.add_argument("--min-points", type=int, default=INJECTION_MIN_POINTS, help="Minimum points in control sample if available.")
    g_injection.add_argument("--seed", type=int, default=INJECTION_SEED)
    g_injection.add_argument("--amp-min", type=float, default=0.001)
    g_injection.add_argument("--amp-max", type=float, default=5.0)
    g_injection.add_argument("--dur-min", type=float, default=1.0)
    g_injection.add_argument("--dur-max", type=float, default=300.0)
    g_injection.add_argument(
        "--total-trials",
        type=int,
        default=INJECTION_TOTAL_TRIALS,
        help="Number of Monte Carlo injection-recovery trials to run.",
    )
    g_injection.add_argument("--skew-min", type=float, default=-0.5)
    g_injection.add_argument("--skew-max", type=float, default=0.5)
    g_injection.add_argument("--mag-err-order", type=int, default=5,
                             help="Deprecated compatibility option; observed-noise injections do not draw new errors")
    g_injection.add_argument("--mag-err-sample", type=int, default=100,
                             help="Deprecated compatibility option; observed-noise injections do not fit an error model")

    g_detection.add_argument("--trigger-mode", choices=["posterior_prob", "logbf"], default=TRIGGER_MODE)
    g_detection.add_argument("--logbf-threshold-dip", type=float, default=LOGBF_THRESHOLD_DIP)
    g_detection.add_argument("--logbf-threshold-jump", type=float, default=LOGBF_THRESHOLD_JUMP)
    g_detection.add_argument("--significance-threshold", type=float, default=SIGNIFICANCE_THRESHOLD)
    g_detection.add_argument("--p-points", type=int, default=P_POINTS)
    g_detection.add_argument("--p-min-dip", type=float, default=None)
    g_detection.add_argument("--p-max-dip", type=float, default=None)
    g_detection.add_argument("--p-min-jump", type=float, default=None)
    g_detection.add_argument("--p-max-jump", type=float, default=None)
    g_detection.add_argument("--mag-points", type=int, default=MAG_POINTS)
    g_detection.add_argument("--mag-min-dip", type=float, default=None)
    g_detection.add_argument("--mag-max-dip", type=float, default=None)
    g_detection.add_argument("--mag-min-jump", type=float, default=None)
    g_detection.add_argument("--mag-max-jump", type=float, default=None)
    g_detection.add_argument("--run-min-points", type=int, default=RUN_MIN_POINTS)
    g_detection.add_argument("--run-max-gap-points", type=int, default=RUN_MAX_GAP_POINTS)
    g_detection.add_argument("--run-max-gap-days", type=float, default=None)
    g_detection.add_argument("--run-min-duration-days", type=float, default=0.0)
    g_detection.add_argument("--baseline-func", choices=BASELINE_CHOICES, default=BASELINE_FUNC)
    g_detection.add_argument("--baseline-s0", type=float, default=BASELINE_S0)
    g_detection.add_argument("--baseline-w0", type=float, default=BASELINE_W0)
    g_detection.add_argument("--baseline-q", type=float, default=BASELINE_Q)
    g_detection.add_argument("--baseline-jitter", type=float, default=BASELINE_JITTER)
    g_detection.add_argument("--baseline-sigma-floor", type=float, default=None)
    g_detection.add_argument("--no-event-prob", action="store_true", default=False)
    g_detection.add_argument("--min-mag-offset", type=float, default=MIN_MAG_OFFSET)
    g_detection.add_argument(
        "--measure-pre-injection",
        dest="measure_pre_injection",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Measure the detection rate before injecting each synthetic dip.",
    )
    g_detection.add_argument(
        "--no-measure-pre-injection",
        dest="measure_pre_injection",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Skip pre-injection detection measurement.",
    )

    g_workers.add_argument("--workers", type=int, default=WORKERS, help="Parallel workers.")
    g_workers.add_argument("--task-size", type=int, default=50, help="Trials per worker task.")
    g_workers.add_argument("--checkpoint-interval", type=int, default=100000, help="Trials per checkpoint update.")
    g_workers.add_argument("--chunk-size", type=int, default=INJECTION_CHUNK_SIZE, help="Rows per output flush.")
    g_workers.add_argument("--no-resume", action="store_true", help="Disable resume even if checkpoint exists.")
    g_workers.add_argument("--overwrite", action="store_true", help="Overwrite output/checkpoint.")

    g_postprocess.add_argument("--skip-cube", action="store_true", help="Skip computing efficiency cube.")
    g_postprocess.add_argument("--skip-plots", action="store_true", help="Skip generating plots.")
    g_postprocess.add_argument("--plot-only", action="store_true", help="Only generate plots from existing results (skips injection-recovery)")
    g_postprocess.add_argument("--cube-out", type=Path, default=None,
                        help="Override cube output path (default: <out-dir>/cubes/efficiency_cube.npz)")
    g_postprocess.add_argument("--plot-dir", type=Path, default=None,
                        help="Override plot directory (default: <out-dir>/plots)")
    g_postprocess.add_argument("--depth-bins", type=int, default=100, help="Number of depth bins for cube.")
    g_postprocess.add_argument("--duration-bins", type=int, default=100, help="Number of duration bins for cube.")
    g_postprocess.add_argument("--mag-bins", type=int, default=100, help="Number of magnitude bins for cube.")
    add_config_args(g_postprocess)
    parser.set_defaults(**INJECTION_CONFIG_DEFAULTS)

    args = parse_args_with_config(
        parser,
        command="injection",
        valid_keys=namespace_keys(parser, INJECTION_CONFIG_DEFAULTS),
        path_keys={"manifest", "out_dir", "output", "cube_out", "plot_dir"},
    )
    if args.plot_only and args.output is None:
        parser.error("--plot-only requires --output pointing to an existing results parquet")

    # Set up output paths with timestamped run directory
    base_out_dir = Path(args.out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.run_tag}" if args.run_tag else timestamp

    if args.output:
        results_out = Path(args.output)
        if results_out.parent.name == "results":
            run_dir = results_out.parent.parent
        else:
            run_dir = results_out.parent
        results_dir = results_out.parent
    else:
        run_dir = base_out_dir / run_name
        results_dir = run_dir / "results"
        results_out = results_dir / "injection_results.parquet"

    cubes_dir = run_dir / "cubes"
    plots_dir = run_dir / "plots"

    results_dir.mkdir(parents=True, exist_ok=True)

    cube_out = args.cube_out if args.cube_out else (cubes_dir / "efficiency_cube.npz")
    plot_dir = args.plot_dir if args.plot_dir else plots_dir

    # Save run parameters to JSON
    run_params_file = run_dir / (
        "postprocess_params.json" if args.plot_only else "run_params.json"
    )
    run_params = vars(args).copy()
    # Convert Path objects to strings for JSON serialization
    for key, value in run_params.items():
        if isinstance(value, Path):
            run_params[key] = str(value)
    with open(run_params_file, "w") as f:
        json.dump(run_params, f, indent=2, default=str)

    # Create/update 'latest' symlink only if we created a new run_dir
    if not args.output:
        latest_link = base_out_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        try:
            latest_link.symlink_to(run_name)
        except Exception as e:
            # Symlinks may fail on some filesystems, just warn
            print(f"Warning: Could not create 'latest' symlink: {e}")

    print(f"\nRun directory: {run_dir}")
    print(f"  Run params: {run_params_file}")
    print(f"  Results Parquet: {results_out}")
    if not args.skip_cube:
        print(f"  Efficiency cube: {cube_out}")
    if not args.skip_plots:
        print(f"  Plots directory: {plot_dir}")
    if not args.output:
        print(f"  Latest symlink: {latest_link} -> {run_name}\n")

    if not args.plot_only:
        manifest = pd.read_parquet(args.manifest)
        control_sample = select_control_sample(
            manifest,
            n_sample=args.control_sample_size,
            min_points=args.min_points,
            seed=args.seed,
        )

        detection_kwargs = _build_detection_kwargs(args)

        run_injection_recovery(
            control_sample,
            detection_kwargs=detection_kwargs,
            min_mag_offset=args.min_mag_offset,
            measure_pre_injection=args.measure_pre_injection,
            total_trials=max(1, args.total_trials),
            amplitude_range=(args.amp_min, args.amp_max),
            duration_range=(args.dur_min, args.dur_max),
            skewness_range=(args.skew_min, args.skew_max),
            mag_err_order=args.mag_err_order,
            mag_err_sample=args.mag_err_sample,
            seed=args.seed,
            workers=max(1, args.workers),
            task_size=max(1, args.task_size),
            checkpoint_interval=max(1, args.checkpoint_interval),
            chunk_size=max(1, args.chunk_size),
            output_path=results_out,
            checkpoint_path=None,
            resume=not args.no_resume,
            overwrite=args.overwrite,
            show_progress=True,
            file_ext=args.file_ext,
        )


    # Post-processing: load results and compute metrics
    print(f"\nLoading results from {results_out}...")
    results_df = pd.read_parquet(results_out)

    # Compute and display quality metrics
    metrics = compute_quality_metrics(results_df)
    output_tag = ""  # No suffix needed with timestamped directories
    metrics_path = results_dir / f"quality_metrics{output_tag}.txt"
    print_quality_summary(metrics, output_path=metrics_path)
    metrics_json_path = results_dir / f"quality_metrics{output_tag}.json"
    metrics_json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    # Compute cube and generate plots (unless skipped)
    if not args.skip_cube or not args.skip_plots:
        if "fractional_depth" not in results_df.columns or "median_mag" not in results_df.columns:
            print("Warning: Results missing fractional_depth or median_mag columns, skipping 3D cube.")
        elif results_df.empty:
            print("Warning: No designed trials are available; skipping 3D cube.")
        else:
            print("Computing end-to-end and observable-conditional 3D efficiency cubes...")
            cube = compute_detection_efficiency_3d(
                results_df,
                depth_bins=args.depth_bins,
                duration_bins=args.duration_bins,
                mag_bins=args.mag_bins,
            )

            if not args.skip_cube:
                cubes_dir.mkdir(parents=True, exist_ok=True)
                save_efficiency_cube(cube, cube_out)
                print(f"Saved efficiency cube to {cube_out}")

            if not args.skip_plots:
                print(f"Generating plots in {plot_dir}...")
                plot_efficiency_all(cube, plot_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
