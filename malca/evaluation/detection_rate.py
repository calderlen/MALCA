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
import json
import multiprocessing as mp

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pa_parquet

from malca.baseline import (
    global_median_baseline,
    per_camera_median_baseline,
    per_camera_gp_baseline,
    per_camera_gp_baseline_masked,
)
from malca.cli_config import add_config_args, apply_config, namespace_keys
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
)
from malca.stv.events import score_lightcurve
from malca.utils import read_lc_dat2


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
    for col in ("asas_sn_id", "source_id", "id"):
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

    # Apply signal amplitude filter if min_mag_offset > 0
    if min_mag_offset > 0 and np.isfinite(baseline_mag):
        dip_diff = abs(dip_best_mag_event - baseline_mag) if np.isfinite(dip_best_mag_event) else 0.0
        jump_diff = abs(jump_best_mag_event - baseline_mag) if np.isfinite(jump_best_mag_event) else 0.0
        if dip_diff <= min_mag_offset:
            dip_significant = False
        if jump_diff <= min_mag_offset:
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
    )


def run_detection_rate_trial(
    trial_index: int,
    control_ids: np.ndarray,
    control_dirs: np.ndarray,
    detection_kwargs: dict,
    seed: int = INJECTION_SEED,
) -> dict:
    """Run detection on a single light curve (no injection)."""
    rng = np.random.default_rng(seed + int(trial_index))

    # Select random LC from control sample
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            control_idx = rng.integers(0, len(control_ids))
            asas_sn_id = str(control_ids[control_idx])
            lc_dir = Path(control_dirs[control_idx])

            df = _load_lc(asas_sn_id, lc_dir)
            if df.empty or len(df) < 10:
                if attempt == max_attempts - 1:
                    return dict(
                        trial_index=trial_index,
                        asas_sn_id=asas_sn_id,
                        detected=False,
                        error="empty_or_short_lc_max_retries",
                    )
                continue

            median_mag = float(np.nanmedian(df["mag"].values))

            # Only process stars with median magnitude between 12 and 15
            if median_mag < INJECTION_MAG_LO or median_mag > INJECTION_MAG_HI:
                if attempt == max_attempts - 1:
                     return dict(
                        trial_index=trial_index,
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
                    asas_sn_id=asas_sn_id,
                    detected=False,
                    error=str(exc),
                )
            continue

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
            **detection_result,
        )
    except Exception as exc:
        return dict(
            trial_index=trial_index,
            asas_sn_id=asas_sn_id,
            detected=False,
            error=str(exc),
        )


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
    """Run detection rate trials in parallel with checkpointing."""


    class _Writer:
        def __init__(self, path: Path):
            self.path = Path(path)
            if self.path.suffix.lower() != ".parquet":
                raise ValueError(f"Detection-rate output must be Parquet: {self.path}")
            self.columns = None
            self.parquet_writer = None

        def write_chunk(self, rows: list[dict]) -> None:
            if not rows:
                return
            df_chunk = pd.DataFrame(rows)
            if self.columns is None:
                self.columns = list(df_chunk.columns)
            df_chunk = df_chunk.reindex(columns=self.columns)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(df_chunk, preserve_index=False)
            if self.parquet_writer is None:
                self.parquet_writer = pa_parquet.ParquetWriter(self.path, table.schema, compression="zstd")
            self.parquet_writer.write_table(table)

        def close(self) -> None:
            if self.parquet_writer is not None:
                self.parquet_writer.close()

    id_col = get_id_column(manifest)
    control_ids = manifest[id_col].values
    control_dirs = manifest["lc_dir"].values if "lc_dir" in manifest.columns else manifest["path"].values

    # Resume logic
    start_index = 0
    if checkpoint_path and checkpoint_path.exists() and not no_resume:
        with open(checkpoint_path, "r") as f:
            content = f.read().strip()
            if content:
                start_index = int(content) + 1
                print(f"Resuming from trial {start_index}")

    results = []
    writer = _Writer(output_path)

    with mp.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(control_ids, control_dirs, detection_kwargs, seed),
    ) as pool:
        for trial_index in tqdm(range(start_index, total_trials), desc="Detection rate trials"):
            result = pool.apply_async(_worker_run_trial, (trial_index,))
            results.append(result)

            # Periodic checkpoint and flush
            if checkpoint_path and (trial_index + 1) % checkpoint_interval == 0:
                # Wait for pending results
                completed = [r.get() for r in results]
                results = []

                df_batch = pd.DataFrame(completed)
                writer.write_chunk(df_batch.to_dict(orient="records"))

                # Update checkpoint
                with open(checkpoint_path, "w") as f:
                    f.write(str(trial_index))

        # Final batch
        if results:
            completed = [r.get() for r in results]
            df_batch = pd.DataFrame(completed)
            writer.write_chunk(df_batch.to_dict(orient="records"))

    writer.close()

    return pd.read_parquet(output_path)


def compute_detection_summary(results_df: pd.DataFrame) -> dict:
    """Compute detection rate summary statistics."""
    total = len(results_df)
    if total == 0:
        return {"total_trials": 0, "detection_rate": np.nan}

    # Filter out errors
    has_error = results_df["error"].notna() if "error" in results_df.columns else pd.Series([False] * len(results_df))
    successful = (~has_error).sum()

    # Detection rate among successful trials
    if successful > 0:
        detected_col = "detected" if "detected" in results_df.columns else "dip_significant"
        if detected_col in results_df.columns:
            successful_mask = ~has_error
            n_detected = results_df.loc[successful_mask, detected_col].sum()
            detection_rate = n_detected / successful
        else:
            detection_rate = np.nan
            n_detected = 0
    else:
        detection_rate = np.nan
        n_detected = 0

    return {
        "total_trials": int(total),
        "successful_trials": int(successful),
        "failed_trials": int(total - successful),
        "detections": int(n_detected),
        "detection_rate": float(detection_rate) if np.isfinite(detection_rate) else None,
        "detection_rate_percent": float(detection_rate * 100) if np.isfinite(detection_rate) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run detection rate measurement (no injection).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output structure (default --output-dir output/detection_rate):
  output/detection_rate/
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

    g_io.add_argument("--manifest", type=Path, default=Path("output/lc_manifest_all.parquet"),
                        help="Manifest parquet path (default: output/lc_manifest_all.parquet)")
    g_io.add_argument("--output-dir", dest="out_dir", type=Path, default=Path("output/detection_rate"),
                        help="Base output directory (default: output/detection_rate)")
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

    args = parser.parse_args()
    apply_config(
        args,
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
