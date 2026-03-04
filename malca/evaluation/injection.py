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

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skewnorm, skew
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from malca.utils import read_lc_dat2
from malca.events import run_bayesian_significance
from malca.baseline import (
    global_median_baseline,
    per_camera_median_baseline,
    per_camera_gp_baseline,
)
from malca.config.config_pipeline import (
    WORKERS,
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
    INJECTION_MIN_POINTS,
    INJECTION_SEED,
    INJECTION_MAX_ATTEMPTS,
)
from malca.config.config_io import INJECTION_CHUNK_SIZE


_GLOBAL: dict[str, object] = {}


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
    sigma = duration / 2.355
    dip = amplitude * skewnorm.pdf(t, a=skewness, loc=t_center, scale=sigma)
    if dip.size and dip.max() > 0:
        dip = dip * (amplitude / dip.max())
    return dip + offset


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
    mag_col: str = "mag",
    time_col: str = "JD",
    err_col: str = "error",
) -> pd.DataFrame:
    """
    Inject synthetic dip into light curve.

    Adds dip to original observed magnitudes, preserving real cadence,
    systematics, and noise. Adds per-point noise based on measurement errors.
    """
    df_out = df_lc.copy()
    if df_out.empty:
        return df_out

    t = df_out[time_col].values
    mag_old = df_out[mag_col].values
    dip_profile = skewnormal_dip(t, t_center, duration, amplitude, skewness)

    # Per-point measurement noise based on errors
    if mag_err_poly is not None:
        sigma_i = np.asarray(mag_err_poly(mag_old), dtype=float)
    else:
        sigma_i = df_out[err_col].values.astype(float)

    # Handle invalid error values
    valid_mask = np.isfinite(sigma_i) & (sigma_i > 0)
    if valid_mask.any():
        fallback = np.nanmedian(sigma_i[valid_mask])
    else:
        fallback = 0.01
    sigma_i = np.where(valid_mask, sigma_i, fallback)

    noise = np.random.normal(0.0, sigma_i, size=len(t))
    df_out[mag_col] = mag_old + dip_profile + noise
    return df_out


def _get_id_col(df: pd.DataFrame) -> str:
    for col in ("asas_sn_id", "source_id", "id"):
        if col in df.columns:
            return col
    raise KeyError("Manifest is missing a usable ID column (expected asas_sn_id/source_id/id).")


def _resolve_lc_dir(row: pd.Series) -> Path | None:
    if "lc_dir" in row and pd.notna(row["lc_dir"]):
        return Path(str(row["lc_dir"]))
    if "dat_path" in row and pd.notna(row["dat_path"]):
        return Path(str(row["dat_path"])).parent
    if "path" in row and pd.notna(row["path"]):
        p = Path(str(row["path"]))
        return p if p.is_dir() else p.parent
    return None


def _load_lc(asas_sn_id: str, lc_dir: Path) -> pd.DataFrame:
    df_g, df_v = read_lc_dat2(asas_sn_id, str(lc_dir))
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
    baseline_map = {
        "gp": per_camera_gp_baseline,
        "global_median": global_median_baseline,
        "per_camera_median": per_camera_median_baseline,
    }

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
        baseline_func=baseline_map.get(args.baseline_func, per_camera_gp_baseline),
        baseline_kwargs=baseline_kwargs,
    )


def _default_detection_func(df: pd.DataFrame, detection_kwargs: dict, min_mag_offset: float = 0.0) -> dict:
    res = run_bayesian_significance(df, **detection_kwargs)
    dip = res["dip"]
    jump = res["jump"]

    baseline_mag = float(dip.get("baseline_mag", jump.get("baseline_mag", np.nan)))
    dip_best_mag_event = float(dip.get("best_mag_event", np.nan))
    jump_best_mag_event = float(jump.get("best_mag_event", np.nan))

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
    )


def _trial_indices_to_params(
    trial_index: int,
    n_injections_per_grid: int,
    n_durations: int,
) -> tuple[int, int, int]:
    n_per_amp = n_durations * n_injections_per_grid
    amp_idx = trial_index // n_per_amp
    rem = trial_index % n_per_amp
    dur_idx = rem // n_injections_per_grid
    inj_idx = rem % n_injections_per_grid
    return amp_idx, dur_idx, inj_idx



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
) -> dict:
    rng = np.random.default_rng(seed + int(trial_index))
    
    # improved random sampling for MC coverage
    # Amplitude: Uniform
    amplitude = rng.uniform(amp_range[0], amp_range[1])
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
            df = _load_lc(asas_sn_id, lc_dir)
            if df.empty or len(df) < 10:
                if attempt == max_attempts - 1:
                    return dict(
                        trial_index=trial_index,
                        amplitude=amplitude,
                        duration=duration,
                        asas_sn_id=asas_sn_id,
                        detected=False,
                        error="empty_or_short_lc_max_retries",
                    )
                continue

            median_mag = float(np.nanmedian(df["mag"].values))
            
            # Only inject into stars with median magnitude between 12 and 15
            if median_mag < INJECTION_MAG_LO or median_mag > INJECTION_MAG_HI:
                if attempt == max_attempts - 1:
                     return dict(
                        trial_index=trial_index,
                        amplitude=amplitude,
                        duration=duration,
                        median_mag=median_mag,
                        asas_sn_id=asas_sn_id,
                        detected=False,
                        error="magnitude_out_of_range",
                    )
                continue
            
            # Found a valid star
            break
            
        except Exception as exc:
            if attempt == max_attempts - 1:
                return dict(
                    trial_index=trial_index,
                    amplitude=amplitude,
                    duration=duration,
                    asas_sn_id=asas_sn_id,
                    detected=False,
                    error=str(exc),
                )
            continue

    try:
        t_min = float(df["JD"].min())
        t_max = float(df["JD"].max())
        if not np.isfinite(t_min) or not np.isfinite(t_max) or (t_max - t_min <= 2 * duration):
            return dict(
                trial_index=trial_index,
                amplitude=amplitude,
                duration=duration,
                asas_sn_id=asas_sn_id,
                detected=False,
                error="invalid_time_range",
            )

        t_center = rng.uniform(t_min + duration, t_max - duration)
        skewness = rng.uniform(skew_range[0], skew_range[1])
        
        # Measure pre-injection detection rate if requested
        pre_injection_result = {}
        if measure_pre_injection:
            pre_inj = _default_detection_func(
                df,
                detection_kwargs,
                min_mag_offset=min_mag_offset,
            )
            # Prefix all keys with pre_injection_
            pre_injection_result = {f"pre_injection_{k}": v for k, v in pre_inj.items()}

        df_injected = inject_dip(df, t_center, duration, amplitude, skewness, mag_err_poly)
        detection_result = _default_detection_func(
            df_injected,
            detection_kwargs,
            min_mag_offset=min_mag_offset,
        )

        # Convert amplitude (mag) to fractional transit depth
        fractional_depth = 1.0 - 10 ** (-0.4 * amplitude)

        return dict(
            trial_index=trial_index,
            amplitude=amplitude,
            fractional_depth=fractional_depth,
            duration=duration,
            skewness=float(skewness),
            t_center=float(t_center),
            median_mag=median_mag,
            asas_sn_id=asas_sn_id,
            **detection_result,
            **pre_injection_result,
        )
    except Exception as exc:
        return dict(
            trial_index=trial_index,
            amplitude=amplitude,
            duration=duration,
            asas_sn_id=asas_sn_id,
            detected=False,
            error=str(exc),
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
            )
        )
    return results



class CsvWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.columns = None
        if self.path.exists() and self.path.stat().st_size > 0:
            try:
                self.columns = pd.read_csv(self.path, nrows=0).columns.tolist()
            except Exception:
                self.columns = None

    def write_chunk(self, chunk_results: list[dict]) -> None:
        if not chunk_results:
            return
        df_chunk = pd.DataFrame(chunk_results)
        if self.columns is None:
            self.columns = list(df_chunk.columns)
        df_chunk = df_chunk.reindex(columns=self.columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = not self.path.exists() or self.path.stat().st_size == 0
        df_chunk.to_csv(self.path, mode="a", header=header, index=False)

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



def run_injection_recovery(
    control_sample: pd.DataFrame,
    *,
    detection_kwargs: dict,
    min_mag_offset: float = 0.0,
    measure_pre_injection: bool = False,
    total_trials: int = INJECTION_N_SAMPLE,
    amplitude_range: tuple[float, float] = (0.05, 2.0),
    duration_range: tuple[float, float] = (1.0, 300.0),
    skewness_range: tuple[float, float] = (-0.5, 0.5),
    mag_err_order: int = 5,
    mag_err_sample: int = 100,
    seed: int = INJECTION_SEED,
    workers: int = 1,
    task_size: int = 50,
    checkpoint_interval: int = 1000,
    chunk_size: int = 1000,
    output_path: Path | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = True,
    overwrite: bool = False,
    max_trials: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame | None:
    """
    Run injection-recovery with optional parallelism and checkpointing.
    Uses Monte Carlo sampling for Amplitude and Duration.
    """
    if max_trials is not None:
        total_trials = min(total_trials, max_trials)

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists() and overwrite and not resume:
            output_path.unlink()
        if output_path.exists() and not resume and not overwrite:
            raise SystemExit(f"Output exists: {output_path} (use --overwrite or --no-resume)")

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
    elif output_path is not None:
        checkpoint_path = output_path.with_name(f"{output_path.stem}_PROCESSED.txt")

    if checkpoint_path and checkpoint_path.exists() and overwrite and not resume:
        checkpoint_path.unlink()

    start_index = 0
    if resume and checkpoint_path and checkpoint_path.exists():
        last = _read_checkpoint(checkpoint_path)
        if last is not None:
            start_index = int(last) + 1

    if start_index >= total_trials:
        print("All trials already completed per checkpoint.")
        return None

    id_col = _get_id_col(control_sample)
    control_ids = control_sample[id_col].astype(str).to_numpy()
    control_dirs = []
    for _, row in control_sample.iterrows():
        lc_dir = _resolve_lc_dir(row)
        if lc_dir is None:
            control_dirs.append("")
        else:
            control_dirs.append(str(lc_dir))
    control_dirs = np.asarray(control_dirs, dtype=object)

    if len(control_ids) == 0:
        raise SystemExit("Control sample is empty.")

    print("Loading control sample light curves for error polynomial...")
    lc_sample = []
    for idx, row in control_sample.iterrows():
        if idx >= mag_err_sample:
            break
        asas_sn_id = str(row[id_col])
        lc_dir = _resolve_lc_dir(row)
        if lc_dir is None:
            continue
        try:
            df = _load_lc(asas_sn_id, lc_dir)
            if not df.empty:
                lc_sample.append(df)
        except Exception:
            continue

    print(f"Fitting {mag_err_order}th-order polynomial to magnitude errors...")
    mag_err_poly = estimate_magnitude_error_polynomial(lc_sample, order=mag_err_order)

    writer = CsvWriter(output_path) if output_path else None
    results: list[dict] = []

    pbar = tqdm(total=total_trials, initial=start_index, disable=not show_progress)
    
    # Ranges
    amp_range = (float(amplitude_range[0]), float(amplitude_range[1]))
    dur_range = (float(duration_range[0]), float(duration_range[1]))
    skew_range = (float(skewness_range[0]), float(skewness_range[1]))

    def flush_results(is_final: bool = False) -> None:
        nonlocal results
        if not results:
            return
        if writer is not None:
            writer.write_chunk(results)
        if is_final and writer is not None:
            writer.close()
        results = []

    if workers <= 1:
        for trial_index in range(start_index, total_trials):
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
        return None if output_path else pd.DataFrame(results)

    from concurrent.futures import ProcessPoolExecutor, as_completed

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
        ),
    ) as ex:
        for batch_start in range(start_index, total_trials, checkpoint_interval):
            batch_end = min(batch_start + checkpoint_interval, total_trials)
            batch_indices = list(range(batch_start, batch_end))
            tasks = [batch_indices[i:i + task_size] for i in range(0, len(batch_indices), task_size)]

            futures = {ex.submit(_process_trial_batch, task): task for task in tasks}
            for fut in as_completed(futures):
                batch_results = fut.result()
                results.extend(batch_results)
                pbar.update(len(batch_results))
                if chunk_size and len(results) >= chunk_size:
                    flush_results()

            flush_results()
            if checkpoint_path:
                _write_checkpoint(checkpoint_path, batch_end - 1)

    flush_results(is_final=True)
    if checkpoint_path:
        _write_checkpoint(checkpoint_path, total_trials - 1)
    pbar.close()
    return None if output_path else pd.DataFrame(results)


def compute_quality_metrics(results_df: pd.DataFrame) -> dict:
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
        return {
            "total_trials": 0,
            "successful_trials": 0,
            "failed_trials": 0,
            "failure_rate": 0.0,
            "detection_rate": 0.0,
            "error_breakdown": {},
            "error_percentages": {},
        }
    
    # Identify failures (rows with non-null 'error' column)
    has_error = results_df.get("error", pd.Series([None] * total)).notna()
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
            pre_inj_rate = results_df.loc[successful_mask, pre_inj_col].sum() / successful
            pre_injection_detection_rate = float(pre_inj_rate) if np.isfinite(pre_inj_rate) else None
            # Net completeness = post-injection rate - pre-injection rate
            if np.isfinite(detection_rate) and np.isfinite(pre_inj_rate):
                net_completeness = float(detection_rate - pre_inj_rate)
            else:
                net_completeness = None
        else:
            pre_injection_detection_rate = None
            net_completeness = None
    else:
        detection_rate = np.nan
        pre_injection_detection_rate = None
        net_completeness = None
    
    # Error breakdown
    error_breakdown = {}
    error_percentages = {}
    if "error" in results_df.columns:
        error_counts = results_df["error"].dropna().value_counts()
        for error_type, count in error_counts.items():
            error_breakdown[str(error_type)] = int(count)
            error_percentages[str(error_type)] = float(count / total * 100)
    
    return {
        "total_trials": int(total),
        "successful_trials": int(successful),
        "failed_trials": int(failed),
        "failure_rate": float(failed / total) if total > 0 else 0.0,
        "detection_rate": float(detection_rate) if np.isfinite(detection_rate) else None,
        "pre_injection_detection_rate": pre_injection_detection_rate,
        "net_completeness": net_completeness,
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
    amp_edges = np.linspace(
        results_df["amplitude"].min(),
        results_df["amplitude"].max(),
        amplitude_bins + 1,
    )
    dur_edges = np.logspace(
        np.log10(results_df["duration"].min()),
        np.log10(results_df["duration"].max()),
        duration_bins + 1,
    )

    amp_centers = (amp_edges[:-1] + amp_edges[1:]) / 2
    dur_centers = np.sqrt(dur_edges[:-1] * dur_edges[1:])

    efficiency_grid = np.zeros((amplitude_bins, duration_bins))

    for i in range(amplitude_bins):
        for j in range(duration_bins):
            mask = (
                (results_df["amplitude"] >= amp_edges[i])
                & (results_df["amplitude"] < amp_edges[i + 1])
                & (results_df["duration"] >= dur_edges[j])
                & (results_df["duration"] < dur_edges[j + 1])
            )
            efficiency_grid[i, j] = results_df.loc[mask, detected_col].mean() if mask.sum() else np.nan

    return amp_centers, dur_centers, efficiency_grid


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
    depth_edges = np.linspace(
        results_df["fractional_depth"].min(),
        results_df["fractional_depth"].max(),
        depth_bins + 1,
    )
    dur_edges = np.logspace(
        np.log10(results_df["duration"].min()),
        np.log10(results_df["duration"].max()),
        duration_bins + 1,
    )
    mag_edges = np.linspace(
        results_df["median_mag"].min(),
        results_df["median_mag"].max(),
        mag_bins + 1,
    )

    depth_centers = (depth_edges[:-1] + depth_edges[1:]) / 2
    dur_centers = np.sqrt(dur_edges[:-1] * dur_edges[1:])
    mag_centers = (mag_edges[:-1] + mag_edges[1:]) / 2

    efficiency = np.full((depth_bins, duration_bins, mag_bins), np.nan)

    for i in range(depth_bins):
        for j in range(duration_bins):
            for k in range(mag_bins):
                mask = (
                    (results_df["fractional_depth"] >= depth_edges[i])
                    & (results_df["fractional_depth"] < depth_edges[i + 1])
                    & (results_df["duration"] >= dur_edges[j])
                    & (results_df["duration"] < dur_edges[j + 1])
                    & (results_df["median_mag"] >= mag_edges[k])
                    & (results_df["median_mag"] < mag_edges[k + 1])
                )
                if mask.sum() > 0:
                    efficiency[i, j, k] = results_df.loc[mask, detected_col].mean()

    return dict(
        efficiency=efficiency,
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
    np.savez(
        output_path,
        efficiency=cube_dict["efficiency"],
        depth_centers=cube_dict["depth_centers"],
        duration_centers=cube_dict["duration_centers"],
        mag_centers=cube_dict["mag_centers"],
        depth_edges=cube_dict["depth_edges"],
        duration_edges=cube_dict["duration_edges"],
        mag_edges=cube_dict["mag_edges"],
    )


def load_efficiency_cube(input_path: Path | str) -> dict:
    """
    Load 3D efficiency cube from .npz file.
    """
    data = np.load(input_path)
    return {k: data[k] for k in data.files}


def plot_detection_efficiency(
    amp_centers: np.ndarray,
    dur_centers: np.ndarray,
    efficiency_grid: np.ndarray,
    output_path: Path | str | None = None,
    title: str = "Detection Efficiency",
    vmin: float = 0.0,
    vmax: float = 1.0,
    xlabel: str = "Duration (days)",
    ylabel: str = "Amplitude (mag)",
    xlog: bool = True,
    cmap: str = "viridis",
    show: bool = True,
    grid_info: str | None = None,
) -> plt.Figure:
    """
    Plot 2D detection efficiency heatmap.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
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
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    title_with_grid = title
    if grid_info:
        title_with_grid = f"{title}\n{grid_info}"
    ax.set_title(title_with_grid, fontsize=16)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Detection Efficiency", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {output_path}")
    elif show:
        plt.show()

    return fig


def plot_efficiency_mag_slices(
    cube: dict,
    *,
    output_dir: Path | str | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
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
    n_depth = len(cube["depth_centers"])
    n_dur = len(cube["duration_centers"])
    grid_info = f"Grid: {n_depth}×{n_dur}"
    
    for k, mag in enumerate(cube["mag_centers"]):
        eff_slice = cube["efficiency"][:, :, k]
        title = f"Detection Efficiency (mag = {mag:.2f})"
        out_path = output_dir / f"efficiency_mag_{mag:.2f}.png" if output_dir else None

        fig = plot_detection_efficiency(
            cube["depth_centers"],
            cube["duration_centers"],
            eff_slice,
            xlabel="Duration (days)",
            ylabel="Fractional Depth",
            title=title,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            output_path=out_path,
            show=show and not output_dir,
            grid_info=grid_info,
        )
        figs.append(fig)
        if output_dir:
            plt.close(fig)

    return figs


def plot_efficiency_marginalized(
    cube: dict,
    *,
    axis: str = "mag",
    output_path: Path | str | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
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
    if axis == "mag":
        eff_2d = np.nanmean(cube["efficiency"], axis=2)
        x_centers = cube["duration_centers"]
        y_centers = cube["depth_centers"]
        xlabel = "Duration (days)"
        ylabel = "Fractional Depth"
        title = "Detection Efficiency (averaged over magnitude)"
        xlog = True
        grid_info = f"Grid: {len(y_centers)}×{len(x_centers)}"
    elif axis == "duration":
        eff_2d = np.nanmean(cube["efficiency"], axis=1)
        x_centers = cube["mag_centers"]
        y_centers = cube["depth_centers"]
        xlabel = "Median Magnitude"
        ylabel = "Fractional Depth"
        title = "Detection Efficiency (averaged over duration)"
        xlog = False
        grid_info = f"Grid: {len(y_centers)}×{len(x_centers)}"
    elif axis == "depth":
        eff_2d = np.nanmean(cube["efficiency"], axis=0).T  # Transpose: (dur, mag) -> (mag, dur)
        x_centers = cube["duration_centers"]
        y_centers = cube["mag_centers"]
        xlabel = "Duration (days)"
        ylabel = "Median Magnitude"
        title = "Detection Efficiency (averaged over depth)"
        xlog = True
        grid_info = f"Grid: {len(y_centers)}×{len(x_centers)}"
    else:
        raise ValueError(f"Unknown axis: {axis}. Use 'mag', 'duration', or 'depth'.")

    return plot_detection_efficiency(
        y_centers,
        x_centers,
        eff_2d,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        xlog=xlog,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        output_path=output_path,
        show=show,
        grid_info=grid_info,
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

    for j in range(n_dur):
        for k in range(n_mag):
            eff = cube["efficiency"][:, j, k]
            valid = np.isfinite(eff)
            if not valid.any():
                continue
            above = eff >= threshold
            if above.any():
                idx = np.where(above)[0][0]
                depth_at_threshold[j, k] = cube["depth_centers"][idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(
        cube["mag_centers"],
        cube["duration_centers"],
        depth_at_threshold,
        cmap=cmap,
        shading="auto",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Median Magnitude", fontsize=12)
    ax.set_ylabel("Duration (days)", fontsize=12)
    ax.set_title(f"Fractional Depth at {threshold*100:.0f}% Detection Efficiency", fontsize=14)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Fractional Depth", fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    elif show:
        plt.show()

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
    
    efficiency_flat = cube["efficiency"].flatten()
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

    grid_info = f"Grid: {len(depth)}×{len(duration)}×{len(mag)}"
    
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
            yaxis_title="log₁₀(Duration/days)",
            zaxis_title="Median Magnitude",
        ),
        title=f"3D Detection Efficiency<br>{grid_info}",
        margin=dict(l=0, r=0, b=0, t=40),
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
    efficiency_flat = cube["efficiency"].flatten()
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
    
    grid_info = f"Grid: {len(depth)}×{len(duration)}×{len(mag)}"

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
            yaxis_title="log₁₀(Duration/days)",
            zaxis_title="Median Magnitude",
        ),
        title=f"Detection Efficiency = {isovalue*100:.0f}% Isosurface<br>{grid_info}",
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
        thresholds = [0.5, 0.9]

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
            output_path=output_dir / f"efficiency_marginalized_{axis}.png",
            show=show,
        )
        plt.close()

    print("Generating threshold contour plots...")
    for thresh in thresholds:
        plot_efficiency_threshold_contour(
            cube,
            threshold=thresh,
            output_path=output_dir / f"depth_at_{int(thresh*100)}pct_efficiency.png",
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


def _build_grid_linear(min_val: float, max_val: float, steps: int) -> np.ndarray:
    return np.linspace(float(min_val), float(max_val), int(steps))


def _build_grid_log(min_val: float, max_val: float, steps: int) -> np.ndarray:
    return np.logspace(np.log10(float(min_val)), np.log10(float(max_val)), int(steps))


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
        "n_injections_per_grid",
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
        epilog="""
Output structure (default --out-dir output/injection):
  output/injection/
    20250121_143052/             # Timestamped run directory
      run_params.json            # Full parameter dump
      results/
        injection_results.csv      # Trial-by-trial results
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
    parser.add_argument("--manifest", type=Path, default=Path("output/lc_manifest_all.parquet"),
                        help="Manifest parquet path (default: output/lc_manifest_all.parquet)")
    parser.add_argument("--out-dir", type=Path, default=Path("output/injection"),
                        help="Base output directory (default: output/injection)")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Optional tag to append to run directory name (e.g., 'deep_dips_mag18')")
    parser.add_argument("--out", type=Path, default=None,
                        help="Override CSV output path (default: <out-dir>/<timestamp>/results/injection_results.csv)")
    parser.add_argument(
        "--control-sample-size",
        dest="control_sample_size",
        type=int,
        default=INJECTION_N_SAMPLE,
        help="Number of control LCs to sample.",
    )
    parser.add_argument("--min-points", type=int, default=INJECTION_MIN_POINTS, help="Minimum points in control sample if available.")
    parser.add_argument("--seed", type=int, default=INJECTION_SEED)

    parser.add_argument("--amp-min", type=float, default=0.05)
    parser.add_argument("--amp-max", type=float, default=3.0)
    parser.add_argument("--amp-steps", type=int, default=100)
    parser.add_argument("--dur-min", type=float, default=1.0)
    parser.add_argument("--dur-max", type=float, default=300.0)
    parser.add_argument("--dur-steps", type=int, default=100)
    parser.add_argument("--n-injections-per-grid", type=int, default=100)
    parser.add_argument("--skew-min", type=float, default=-0.5)
    parser.add_argument("--skew-max", type=float, default=0.5)
    parser.add_argument("--mag-err-order", type=int, default=5)
    parser.add_argument("--mag-err-sample", type=int, default=100)

    parser.add_argument("--workers", type=int, default=WORKERS, help="Parallel workers.")
    parser.add_argument("--task-size", type=int, default=50, help="Trials per worker task.")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Trials per checkpoint update.")
    parser.add_argument("--chunk-size", type=int, default=INJECTION_CHUNK_SIZE, help="Rows per output flush.")
    parser.add_argument("--max-trials", type=int, default=None, help="Limit total trials (debug).")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume even if checkpoint exists.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output/checkpoint.")

    parser.add_argument("--trigger-mode", type=str, default=TRIGGER_MODE, choices=["logbf", "posterior_prob"],
                        help="Trigger mode for injection testing")
    parser.add_argument("--logbf-threshold-dip", type=float, default=LOGBF_THRESHOLD_DIP)
    parser.add_argument("--logbf-threshold-jump", type=float, default=LOGBF_THRESHOLD_JUMP)
    parser.add_argument("--significance-threshold", type=float, default=SIGNIFICANCE_THRESHOLD)
    parser.add_argument("--p-points", type=int, default=P_POINTS)
    parser.add_argument("--mag-points", type=int, default=MAG_POINTS, help="Number of points in the magnitude grid")
    parser.add_argument("--p-min-dip", type=float, default=None)
    parser.add_argument("--p-max-dip", type=float, default=None)
    parser.add_argument("--p-min-jump", type=float, default=None)
    parser.add_argument("--p-max-jump", type=float, default=None)
    parser.add_argument("--run-min-points", type=int, default=RUN_MIN_POINTS)
    parser.add_argument("--run-max-gap-points", type=int, default=RUN_MAX_GAP_POINTS)
    parser.add_argument("--run-max-gap-days", type=float, default=None)
    parser.add_argument("--run-min-duration-days", type=float, default=0.0)
    parser.add_argument(
        "--baseline-func",
        type=str,
        default=BASELINE_FUNC,
        choices=["gp", "global_median", "per_camera_median"],
        help="Baseline function",
    )
    # Baseline kwargs (GP kernel parameters)
    parser.add_argument("--baseline-s0", type=float, default=BASELINE_S0, help="GP kernel S0 parameter (default: 0.0005)")
    parser.add_argument("--baseline-w0", type=float, default=BASELINE_W0, help="GP kernel w0 parameter (default: pi/1000)")
    parser.add_argument("--baseline-q", type=float, default=BASELINE_Q, help="GP kernel Q parameter (default: 0.7)")
    parser.add_argument("--baseline-jitter", type=float, default=BASELINE_JITTER, help="GP jitter term (default: 0.006)")
    parser.add_argument("--baseline-sigma-floor", type=float, default=None, help="Minimum sigma floor (default: None)")
    # Magnitude grid bounds (override auto-detection)
    parser.add_argument("--mag-min-dip", type=float, default=None, help="Min magnitude for dip grid (overrides auto)")
    parser.add_argument("--mag-max-dip", type=float, default=None, help="Max magnitude for dip grid (overrides auto)")
    parser.add_argument("--mag-min-jump", type=float, default=None, help="Min magnitude for jump grid (overrides auto)")
    parser.add_argument("--mag-max-jump", type=float, default=None, help="Max magnitude for jump grid (overrides auto)")
    parser.add_argument("--no-event-prob", action="store_true", default=False,
                        help="Disable event probability computation for faster runs")
    parser.add_argument("--compute-event-prob", dest="no_event_prob", action="store_false",
                        help="Enable event probability computation (default)")
    parser.add_argument("--min-mag-offset", type=float, default=MIN_MAG_OFFSET,
                        help="Min magnitude offset for signal amplitude filter (0 to disable, default: 0.2)")
    parser.add_argument("--measure-pre-injection", action="store_true", default=True,
                        help="Measure detection rate on pre-injection light curves (default: enabled)")
    parser.add_argument("--no-measure-pre-injection", dest="measure_pre_injection", action="store_false",
                        help="Disable pre-injection detection rate measurement")


    # Post-processing options
    parser.add_argument("--skip-cube", action="store_true", help="Skip computing efficiency cube.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip generating plots.")
    parser.add_argument("--cube-out", type=Path, default=None,
                        help="Override cube output path (default: <out-dir>/cubes/efficiency_cube.npz)")
    parser.add_argument("--plot-dir", type=Path, default=None,
                        help="Override plot directory (default: <out-dir>/plots)")
    parser.add_argument("--depth-bins", type=int, default=20, help="Number of depth bins for cube.")
    parser.add_argument("--duration-bins", type=int, default=20, help="Number of duration bins for cube.")
    parser.add_argument("--mag-bins", type=int, default=10, help="Number of magnitude bins for cube.")

    args = parser.parse_args()

    # Set up output paths with timestamped run directory
    base_out_dir = Path(args.out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.run_tag}" if args.run_tag else timestamp

    run_dir = base_out_dir / run_name

    results_dir = run_dir / "results"
    cubes_dir = run_dir / "cubes"
    plots_dir = run_dir / "plots"

    results_dir.mkdir(parents=True, exist_ok=True)

    csv_out = args.out if args.out else (results_dir / "injection_results.csv")
    cube_out = args.cube_out if args.cube_out else (cubes_dir / "efficiency_cube.npz")
    plot_dir = args.plot_dir if args.plot_dir else plots_dir

    # Save run parameters to JSON
    run_params_file = run_dir / "run_params.json"
    run_params = vars(args).copy()
    # Convert Path objects to strings for JSON serialization
    for key, value in run_params.items():
        if isinstance(value, Path):
            run_params[key] = str(value)
    with open(run_params_file, "w") as f:
        json.dump(run_params, f, indent=2, default=str)

    # Create/update 'latest' symlink
    latest_link = base_out_dir / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    try:
        latest_link.symlink_to(run_name)
    except Exception as e:
        # Symlinks may fail on some filesystems, just warn
        print(f"Warning: Could not create 'latest' symlink: {e}")

    manifest = pd.read_parquet(args.manifest)
    control_sample = select_control_sample(
        manifest,
        n_sample=args.control_sample_size,
        min_points=args.min_points,
        seed=args.seed,
    )


    detection_kwargs = _build_detection_kwargs(args)

    print(f"\nRun directory: {run_dir}")
    print(f"  Run params: {run_params_file}")
    print(f"  Results CSV: {csv_out}")
    if not args.skip_cube:
        print(f"  Efficiency cube: {cube_out}")
    if not args.skip_plots:
        print(f"  Plots directory: {plot_dir}")
    print(f"  Latest symlink: {latest_link} -> {run_name}\n")


    # Calculate Total Trials equivalent to previous grid based approach
    # We maintain the same "density" concept for user convenience
    total_trials = args.amp_steps * args.dur_steps * args.n_injections_per_grid

    run_injection_recovery(
        control_sample,
        detection_kwargs=detection_kwargs,
        min_mag_offset=args.min_mag_offset,
        measure_pre_injection=args.measure_pre_injection,
        total_trials=total_trials,
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
        output_path=csv_out,
        checkpoint_path=None,
        resume=not args.no_resume,
        overwrite=args.overwrite,
        max_trials=args.max_trials,
        show_progress=True,
    )


    # Post-processing: load results and compute metrics
    print(f"\nLoading results from {csv_out}...")
    results_df = pd.read_csv(csv_out)

    # Compute and display quality metrics
    metrics = compute_quality_metrics(results_df)
    output_tag = ""  # No suffix needed with timestamped directories
    metrics_path = results_dir / f"quality_metrics{output_tag}.txt"
    print_quality_summary(metrics, output_path=metrics_path)

    results_ok = results_df
    if "error" in results_df.columns:
        results_ok = results_df.loc[results_df["error"].isna()].copy()
        if len(results_ok) < len(results_df):
            print(f"Using {len(results_ok)}/{len(results_df)} successful trials for efficiency cube/plots.")
    if "median_mag" in results_ok.columns:
        results_ok = results_ok[np.isfinite(results_ok["median_mag"])].copy()
    if "fractional_depth" in results_ok.columns:
        results_ok = results_ok[np.isfinite(results_ok["fractional_depth"])].copy()

    # Compute cube and generate plots (unless skipped)
    if not args.skip_cube or not args.skip_plots:
        if "fractional_depth" not in results_ok.columns or "median_mag" not in results_ok.columns:
            print("Warning: Results missing fractional_depth or median_mag columns, skipping 3D cube.")
        elif results_ok.empty:
            print("Warning: No successful trials with valid magnitudes/depths; skipping 3D cube.")
        else:
            print("Computing 3D efficiency cube...")
            cube = compute_detection_efficiency_3d(
                results_ok,
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
