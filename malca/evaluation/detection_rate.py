"""
Detection rate measurement for dipper pipeline.

Runs the detection pipeline on a sample of light curves WITHOUT injection
to measure the baseline detection rate. This provides:
1. Detection rate vs magnitude, timespan, cadence, etc.
2. Baseline for false positive estimation (cross-match with VSX, etc.)
3. Real-world candidate rate for occurrence rate calculations

Complements injection-recovery testing which measures completeness.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json
import multiprocessing as mp
import os

from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from malca.core.baseline import (
    global_median_baseline,
    per_camera_median_baseline,
    per_camera_gp_baseline,
    per_camera_gp_baseline_masked,
)
from malca.cli_config import add_config_args, namespace_keys, parse_args_with_config
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
    INJECTION_MIN_POINTS,
    INJECTION_SEED,
    DEFAULT_OUTPUT_DIR,
)
from malca.stv.events import score_lightcurve
from malca.core.utils import read_lc_dat2


DETECTION_RATE_CONFIG_DEFAULTS = {
    "trigger_mode": TRIGGER_MODE,
    "logbf_threshold_dip": LOGBF_THRESHOLD_DIP,
    "logbf_threshold_jump": LOGBF_THRESHOLD_JUMP,
    "significance_threshold": SIGNIFICANCE_THRESHOLD,
    "p_points": P_POINTS,
    "p_min_dip": None,
    "p_max_dip": None,
    "p_min_jump": None,
    "p_max_jump": None,
    "mag_points": MAG_POINTS,
    "mag_min_dip": None,
    "mag_max_dip": None,
    "mag_min_jump": None,
    "mag_max_jump": None,
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
    "no_event_prob": False,
    "min_mag_offset": MIN_MAG_OFFSET,
}

BASELINE_CHOICES = ("gp", "gp_masked", "global_median", "per_camera_median")








def get_id_column(df: pd.DataFrame) -> str:
    """Find the ID column in manifest."""
    for col in ("candidate_id", "asas_sn_id", "source_id", "id"):
        if col in df.columns:
            return col
    raise KeyError("Manifest is missing a usable ID column (expected asas_sn_id/source_id/id).")


def _load_lc(asas_sn_id: str, lc_dir: Path) -> pd.DataFrame:
    df_g, df_v = read_lc_dat2(asas_sn_id, str(lc_dir))
    if df_g is not None and not df_g.empty:
        return df_g
    if df_v is not None and not df_v.empty:
        return df_v
    return pd.DataFrame()


def select_control_sample(
    manifest: pd.DataFrame,
    n_sample: int = INJECTION_N_SAMPLE,
    min_points: int = INJECTION_MIN_POINTS,
    seed: int = INJECTION_SEED,
    reject_candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select control sample of light curves."""
    df = manifest.copy()
    id_col = get_id_column(df)
    ids = df[id_col].astype("string").str.strip()
    if bool((ids.isna() | ids.eq("")).any()):
        raise ValueError(f"Control manifest contains blank/null {id_col} values")
    if bool(ids.duplicated().any()):
        raise ValueError(f"Control manifest contains duplicate {id_col} values")

    # Reject known candidates if provided
    if reject_candidates is not None and "asas_sn_id" in reject_candidates.columns:
        exclude_ids = set(reject_candidates["asas_sn_id"].astype(str))
        df = df[~df["asas_sn_id"].astype(str).isin(exclude_ids)]

    # Filter by minimum points if available
    if "n_points" in df.columns:
        df = df[df["n_points"] >= min_points]

    # Sample
    rng = np.random.default_rng(seed)
    if len(df) <= n_sample:
        return df
    indices = rng.choice(len(df), size=n_sample, replace=False)
    return df.iloc[indices].reset_index(drop=True)


def _build_detection_kwargs(args: argparse.Namespace) -> dict:
    """Build kwargs for score_lightcurve from args."""
    baseline_kwargs = {
        "S0": args.baseline_s0,
        "w0": args.baseline_w0,
        "q": args.baseline_q,
        "jitter": args.baseline_jitter,
        "add_sigma_eff_col": True,
    }
    if args.baseline_sigma_floor is not None:
        baseline_kwargs["sigma_floor"] = args.baseline_sigma_floor

    baseline_map = {
        "gp": per_camera_gp_baseline,
        "gp_masked": per_camera_gp_baseline_masked,
        "global_median": global_median_baseline,
        "per_camera_median": per_camera_median_baseline,
    }
    try:
        baseline_func = baseline_map[str(args.baseline_func)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported baseline_func {args.baseline_func!r}; expected one of {', '.join(BASELINE_CHOICES)}"
        ) from exc

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
        min_mag_offset=args.min_mag_offset,
        baseline_func=baseline_func,
        baseline_kwargs=baseline_kwargs,
    )


def _extract_detection_result(
    dip: dict,
    jump: dict,
    min_mag_offset: float = 0.0,
) -> dict:
    """Extract detection results from score_lightcurve output."""
    dip_significant = bool(dip["significant"])
    jump_significant = bool(jump["significant"])

    baseline_mag = float(dip.get("baseline_mag", jump.get("baseline_mag", np.nan)))
    dip_best_mag_event = float(dip.get("best_mag_event", np.nan))
    jump_best_mag_event = float(jump.get("best_mag_event", np.nan))
    dip_best_delta_mag = float(dip.get("best_delta_mag", dip_best_mag_event))
    jump_best_delta_mag = float(jump.get("best_delta_mag", jump_best_mag_event))

    # Apply signal amplitude filter if min_mag_offset > 0
    if min_mag_offset > 0:
        dip_diff = abs(dip_best_delta_mag) if np.isfinite(dip_best_delta_mag) else np.nan
        jump_diff = abs(jump_best_delta_mag) if np.isfinite(jump_best_delta_mag) else np.nan
        if not np.isfinite(dip_diff) or dip_diff < min_mag_offset:
            dip_significant = False
        if not np.isfinite(jump_diff) or jump_diff < min_mag_offset:
            jump_significant = False

    return dict(
        detected=dip_significant or jump_significant,
        dip_significant=dip_significant,
        jump_significant=jump_significant,
        dip_bayes_factor=float(dip["bayes_factor"]),
        jump_bayes_factor=float(jump["bayes_factor"]),
        dip_best_p=float(dip["best_p"]),
        jump_best_p=float(jump["best_p"]),
        baseline_mag=baseline_mag,
        dip_best_mag_event=dip_best_mag_event,
        jump_best_mag_event=jump_best_mag_event,
        dip_best_delta_mag=dip_best_delta_mag,
        jump_best_delta_mag=jump_best_delta_mag,
    )


def _trial_failure(trial_index: int, asas_sn_id: str, status: str, error: str) -> dict:
    return {
        "trial_index": int(trial_index),
        "asas_sn_id": str(asas_sn_id),
        "trial_status": status,
        "detected": pd.NA,
        "dip_significant": pd.NA,
        "jump_significant": pd.NA,
        "median_mag": np.nan,
        "dip_bayes_factor": np.nan,
        "jump_bayes_factor": np.nan,
        "dip_best_p": np.nan,
        "jump_best_p": np.nan,
        "baseline_mag": np.nan,
        "dip_best_mag_event": np.nan,
        "jump_best_mag_event": np.nan,
        "dip_best_delta_mag": np.nan,
        "jump_best_delta_mag": np.nan,
        "error": str(error),
    }


def run_detection_rate_trial(
    trial_index: int,
    control_ids: np.ndarray,
    control_dirs: np.ndarray,
    detection_kwargs: dict,
    seed: int = INJECTION_SEED,
) -> dict:
    """Run detection on a single light curve (no injection)."""
    if len(control_ids) == 0:
        return _trial_failure(trial_index, "", "error", "empty_control_manifest")
    # Stable permutation makes the designed sample independent of worker count,
    # scheduling, and resume boundaries.  Every control is used once per cycle.
    permutation = np.random.default_rng(seed).permutation(len(control_ids))
    control_idx = int(permutation[int(trial_index) % len(permutation)])
    asas_sn_id = str(control_ids[control_idx])
    lc_dir = Path(control_dirs[control_idx])
    try:
        df = _load_lc(asas_sn_id, lc_dir)
    except Exception as exc:
        return _trial_failure(trial_index, asas_sn_id, "error", f"load_error:{type(exc).__name__}:{exc}")
    if df.empty or len(df) < 10:
        return _trial_failure(trial_index, asas_sn_id, "ineligible_short_lightcurve", "empty_or_short_lc")

    median_mag = float(np.nanmedian(df["mag"].values))
    if not np.isfinite(median_mag) or median_mag < INJECTION_MAG_LO or median_mag > INJECTION_MAG_HI:
        record = _trial_failure(trial_index, asas_sn_id, "ineligible_magnitude", "magnitude_out_of_range")
        record["median_mag"] = median_mag
        return record

    try:
        # Run detection on original LC (no injection)
        score_kwargs = {k: v for k, v in detection_kwargs.items() if k != "min_mag_offset"}
        result = score_lightcurve(df, **score_kwargs)
        detection_result = _extract_detection_result(
            result["dip"],
            result["jump"],
            min_mag_offset=detection_kwargs.get("min_mag_offset", 0.0),
        )

        return dict(
            trial_index=trial_index,
            asas_sn_id=asas_sn_id,
            median_mag=median_mag,
            trial_status="ok",
            error=None,
            **detection_result,
        )
    except Exception as exc:
        record = _trial_failure(trial_index, asas_sn_id, "error", f"score_error:{type(exc).__name__}:{exc}")
        record["median_mag"] = median_mag
        return record


def _init_worker(
    control_ids: np.ndarray,
    control_dirs: np.ndarray,
    detection_kwargs: dict,
    seed: int,
):
    """Initialize worker process state."""
    global _worker_control_ids, _worker_control_dirs, _worker_detection_kwargs, _worker_seed
    _worker_control_ids = control_ids
    _worker_control_dirs = control_dirs
    _worker_detection_kwargs = detection_kwargs
    _worker_seed = seed


def _worker_run_trial(trial_index: int) -> dict:
    """Worker function for parallel processing."""
    return run_detection_rate_trial(
        trial_index,
        _worker_control_ids,
        _worker_control_dirs,
        _worker_detection_kwargs,
        seed=_worker_seed,
    )


def run_detection_rate(
    manifest: pd.DataFrame,
    total_trials: int,
    detection_kwargs: dict,
    output_path: Path,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 1000,
    workers: int = 10,
    seed: int = 42,
    no_resume: bool = False,
) -> pd.DataFrame:
    """Run deterministic trials with recoverable, fingerprinted checkpoints."""
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".parquet":
        raise ValueError(f"Detection-rate output must be Parquet: {output_path}")
    if total_trials < 0:
        raise ValueError("total_trials must be non-negative")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")

    id_col = get_id_column(manifest)
    control_ids = manifest[id_col].values
    location_column = "lc_dir" if "lc_dir" in manifest.columns else "path"
    if location_column not in manifest.columns:
        raise ValueError("Control manifest requires an explicit lc_dir or path column")
    control_dirs = manifest[location_column].values
    fingerprint_payload = {
        "version": 2,
        "total_trials": int(total_trials),
        "seed": int(seed),
        "control_ids": [str(value) for value in control_ids],
        "control_locations": [str(value) for value in control_dirs],
        "detection_kwargs": _fingerprintable(detection_kwargs),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else output_path.with_suffix(".checkpoint.json")
    part_dir = output_path.with_name(f".{output_path.stem}.parts")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame()
    if no_resume:
        if output_path.exists() or checkpoint_path.exists() or part_dir.exists():
            raise FileExistsError("no_resume=True refuses existing output/checkpoint state; remove it explicitly")
    elif output_path.exists() or checkpoint_path.exists() or part_dir.exists():
        if not checkpoint_path.exists():
            raise RuntimeError("Cannot resume detection-rate output without its fingerprinted checkpoint")
        try:
            checkpoint = json.loads(checkpoint_path.read_text())
        except Exception as exc:
            raise RuntimeError("Legacy or unreadable detection-rate checkpoint; start a new run") from exc
        if checkpoint.get("fingerprint") != fingerprint:
            raise RuntimeError("Detection-rate checkpoint fingerprint does not match this run")
        if output_path.exists():
            existing = pd.read_parquet(output_path)

    part_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        _write_detection_checkpoint(checkpoint_path, fingerprint, total_trials)
    part_frames: list[pd.DataFrame] = []
    for part_path in sorted(part_dir.glob("part_*.parquet")):
        part_frames.append(pd.read_parquet(part_path))
    completed_frames = [frame for frame in [existing, *part_frames] if not frame.empty]
    completed = pd.concat(completed_frames, ignore_index=True) if completed_frames else pd.DataFrame()
    completed_indices = set(pd.to_numeric(completed.get("trial_index", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
    pending_indices = [idx for idx in range(total_trials) if idx not in completed_indices]
    if completed_indices - set(range(total_trials)):
        raise RuntimeError("Checkpoint contains trial indices outside this design")
    if pending_indices:
        print(f"Running {len(pending_indices)} pending trial(s); {len(completed_indices)} already complete")

    async_results = []
    rows_buffer: list[dict] = []

    with mp.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(control_ids, control_dirs, detection_kwargs, seed),
    ) as pool:
        for position, trial_index in enumerate(tqdm(pending_indices, desc="Detection rate trials"), start=1):
            result = pool.apply_async(_worker_run_trial, (trial_index,))
            async_results.append(result)

            if position % checkpoint_interval == 0:
                rows_buffer.extend(result.get() for result in async_results)
                async_results = []
                _write_detection_part(part_dir, rows_buffer)
                rows_buffer = []
                _write_detection_checkpoint(checkpoint_path, fingerprint, total_trials)

        if async_results:
            rows_buffer.extend(result.get() for result in async_results)
        if rows_buffer:
            _write_detection_part(part_dir, rows_buffer)
            _write_detection_checkpoint(checkpoint_path, fingerprint, total_trials)

    all_frames = [existing] if not existing.empty else []
    all_frames.extend(pd.read_parquet(path) for path in sorted(part_dir.glob("part_*.parquet")))
    final = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(columns=["trial_index"])
    if not final.empty:
        final["trial_index"] = pd.to_numeric(final["trial_index"], errors="raise").astype(int)
        final = final.sort_values("trial_index").drop_duplicates("trial_index", keep="last").reset_index(drop=True)
    expected_indices = set(range(total_trials))
    final_indices = set(final["trial_index"].astype(int)) if not final.empty else set()
    if final_indices != expected_indices:
        raise RuntimeError(
            f"Detection-rate stage incomplete: missing={sorted(expected_indices-final_indices)[:10]}, "
            f"unexpected={sorted(final_indices-expected_indices)[:10]}"
        )
    temp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    final.to_parquet(temp_output, index=False)
    os.replace(temp_output, output_path)
    for part_path in part_dir.glob("part_*.parquet"):
        part_path.unlink()
    part_dir.rmdir()
    _write_detection_checkpoint(checkpoint_path, fingerprint, total_trials, complete=True)
    return final


def _fingerprintable(value):
    if isinstance(value, dict):
        return {str(key): _fingerprintable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_fingerprintable(item) for item in value]
    if callable(value):
        return f"{value.__module__}.{getattr(value, '__qualname__', value.__name__)}"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_detection_part(part_dir: Path, rows: list[dict]) -> Path:
    frame = pd.DataFrame(rows)
    start = int(frame["trial_index"].min())
    stop = int(frame["trial_index"].max())
    path = part_dir / f"part_{start:09d}_{stop:09d}.parquet"
    temp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)
    return path


def _write_detection_checkpoint(
    path: Path,
    fingerprint: str,
    total_trials: int,
    *,
    complete: bool = False,
) -> None:
    payload = {
        "version": 2,
        "fingerprint": fingerprint,
        "total_trials": int(total_trials),
        "complete": bool(complete),
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def compute_detection_summary(results_df: pd.DataFrame) -> dict:
    """Compute detection rate summary statistics."""
    total = len(results_df)
    if total == 0:
        return {
            "total_trials": 0,
            "successful_trials": 0,
            "failed_trials": 0,
            "ineligible_trials": 0,
            "detections": 0,
            "detection_rate": None,
            "detection_rate_percent": None,
            "end_to_end_detection_yield": None,
        }

    status = (
        results_df["trial_status"].astype("string")
        if "trial_status" in results_df.columns
        else pd.Series(np.where(results_df.get("error", pd.Series(index=results_df.index)).notna(), "error", "ok"), index=results_df.index)
    )
    successful_mask = status.eq("ok")
    error_mask = status.eq("error")
    ineligible_mask = ~(successful_mask | error_mask)
    successful = int(successful_mask.sum())
    detected_col = "detected" if "detected" in results_df.columns else "dip_significant"
    if detected_col not in results_df.columns:
        raise ValueError(f"Detection-rate results are missing {detected_col!r}")
    detected = results_df[detected_col].astype("boolean")
    if bool(detected.loc[successful_mask].isna().any()):
        raise ValueError("Successful detection-rate rows contain unknown detection decisions")
    n_detected = int(detected.loc[successful_mask].sum())
    detection_rate = n_detected / successful if successful else np.nan
    end_to_end_yield = n_detected / total
    conditional_low, conditional_high = _wilson_interval(n_detected, successful)
    yield_low, yield_high = _wilson_interval(n_detected, total)

    return {
        "total_trials": int(total),
        "successful_trials": int(successful),
        "failed_trials": int(error_mask.sum()),
        "ineligible_trials": int(ineligible_mask.sum()),
        "detections": int(n_detected),
        "detection_rate": float(detection_rate) if np.isfinite(detection_rate) else None,
        "detection_rate_percent": float(detection_rate * 100) if np.isfinite(detection_rate) else None,
        "detection_rate_ci95_low": conditional_low,
        "detection_rate_ci95_high": conditional_high,
        "end_to_end_detection_yield": float(end_to_end_yield),
        "end_to_end_detection_yield_ci95_low": yield_low,
        "end_to_end_detection_yield_ci95_high": yield_high,
        "unique_controls_evaluated": int(results_df.get("asas_sn_id", pd.Series(dtype=str)).nunique()),
        "rate_denominator_definition": "trial_status == 'ok'",
        "yield_denominator_definition": "all designed trials; errors and ineligible rows are not relabelled as nondetections",
    }


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if trials <= 0:
        return None, None
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half_width = z * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials**2)) / denominator
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run detection rate measurement (no injection).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Output structure (default --output-dir {DEFAULT_OUTPUT_DIR / 'detection_rate'}):
  {DEFAULT_OUTPUT_DIR / 'detection_rate'}/
    20250121_143052/             # Timestamped run directory
      run_params.json            # Full parameter dump
      results/
        detection_rate_results.parquet     # Trial-by-trial results
        detection_rate_results_PROCESSED.txt  # Checkpoint
        detection_summary.json     # Detection rate summary
      plots/
        detection_rate_vs_mag.png
        detection_duration_dist.png
        detection_depth_dist.png
    20250121_150318_custom_tag/  # Optional --run-tag appended
      ...
    latest -> 20250121_150318_custom_tag/  # Symlink to latest run

Each run gets a unique timestamped directory. Use --run-tag to append a custom label.
""",
    )
    g_io = parser.add_argument_group("Input / output")
    g_sample = parser.add_argument_group("Sample")
    g_detection = parser.add_argument_group("Detection")
    g_workers = parser.add_argument_group("Workers")
    g_config = parser.add_argument_group("Config")

    g_io.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "lc_manifest_all.parquet",
                        help=f"Manifest parquet path (default: {DEFAULT_OUTPUT_DIR / 'lc_manifest_all.parquet'})")
    g_io.add_argument("--output-dir", dest="out_dir", type=Path, default=DEFAULT_OUTPUT_DIR / "detection_rate",
                        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR / 'detection_rate'})")
    g_io.add_argument("--run-tag", type=str, default=None,
                        help="Optional tag to append to run directory name (e.g., 'mag12-13')")
    g_io.add_argument("--output", type=Path, default=None,
                        help="Override output path (default: <out-dir>/<timestamp>/results/detection_rate_results.parquet)")
    g_sample.add_argument(
        "--control-sample-size",
        "--sample-size",
        dest="control_sample_size",
        type=int,
        default=INJECTION_N_SAMPLE,
        help="Number of light curves to sample.",
    )
    g_sample.add_argument("--min-points", type=int, default=INJECTION_MIN_POINTS, help="Minimum points in control sample if available.")
    g_sample.add_argument("--seed", type=int, default=INJECTION_SEED)

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

    g_workers.add_argument("--workers", type=int, default=WORKERS, help="Parallel workers.")
    g_workers.add_argument("--checkpoint-interval", type=int, default=1000, help="Trials per checkpoint update.")
    g_workers.add_argument("--max-trials", type=int, default=None, help="Limit total trials (debug).")
    g_workers.add_argument("--no-resume", action="store_true", help="Disable resume even if checkpoint exists.")
    g_workers.add_argument("--overwrite", action="store_true", help="Overwrite existing output if present.")

    add_config_args(g_config)
    parser.set_defaults(**DETECTION_RATE_CONFIG_DEFAULTS)

    args = parse_args_with_config(
        parser,
        command="detection-rate",
        valid_keys=namespace_keys(parser, DETECTION_RATE_CONFIG_DEFAULTS),
        path_keys={"manifest", "out_dir", "output"},
    )

    # Set up output paths with timestamped run directory
    base_out_dir = Path(args.out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.run_tag}" if args.run_tag else timestamp

    run_dir = base_out_dir / run_name

    results_dir = run_dir / "results"
    plots_dir = run_dir / "plots"

    results_dir.mkdir(parents=True, exist_ok=True)

    output_out = args.output if args.output else (results_dir / "detection_rate_results.parquet")
    summary_out = results_dir / "detection_summary.json"

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
    print(f"  Results file: {output_out}")
    print(f"  Summary: {summary_out}")
    print(f"  Latest symlink: {latest_link} -> {run_name}\n")

    total_trials = args.max_trials if args.max_trials else args.control_sample_size
    checkpoint_path = output_out.with_name(f"{output_out.stem}_PROCESSED.txt")

    if args.overwrite and output_out.exists():
        output_out.unlink()
        print(f"Overwriting existing output: {output_out}")
    if args.overwrite and checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"Running detection rate measurement on {total_trials} light curves...")
    results_df = run_detection_rate(
        control_sample,
        total_trials=total_trials,
        detection_kwargs=detection_kwargs,
        output_path=output_out,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
        workers=args.workers,
        seed=args.seed,
        no_resume=args.no_resume,
    )

    # Compute and save summary
    summary = compute_detection_summary(results_df)
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("DETECTION RATE SUMMARY")
    print("="*60)
    print(f"Total trials:       {summary['total_trials']}")
    print(f"Successful trials:  {summary['successful_trials']}")
    print(f"Failed trials:      {summary['failed_trials']}")
    print(f"Detections:         {summary['detections']}")
    rate_str = f"{summary['detection_rate_percent']:.2f}%" if summary['detection_rate_percent'] is not None else "N/A"
    print(f"Detection rate:     {rate_str}")
    print("="*60)
    print(f"\nResults saved to: {output_out}")
    print(f"Summary saved to: {summary_out}")


if __name__ == "__main__":
    main()
