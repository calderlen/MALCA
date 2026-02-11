"""
Plot light curves for candidates that passed all post-filters.

Reads the filtered events results and plots only sources with failed_any == False.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from malca.plot import plot_bayes_results, BASELINE_FUNCTIONS
from malca.review.metadata import REVIEW_METADATA_FIELDS, normalize_vsx_df, normalize_vsx_record


def load_passing_candidates(
    filtered_path: Path | pd.DataFrame,
    *,
    require_failed_any_false: bool = True,
    require_flags: list[str] | None = None,
    exclude_flags: list[str] | None = None,
    min_lsp_power: float | None = None,
    max_lsp_bootstrap_sig: float | None = None,
    min_periodicity_score: float | None = None,
    max_plots: int | None = None,
) -> pd.DataFrame:
    """
    Load candidates that passed all post-filters.

    Parameters
    ----------
    filtered_path : Path
        Path to filtered events results (CSV/Parquet)
    max_plots : int | None
        Maximum number of candidates to return

    Returns
    -------
    pd.DataFrame
        Candidates with failed_any == False
    """
    if isinstance(filtered_path, pd.DataFrame):
        df = filtered_path.copy()
    else:
        filtered_path = Path(filtered_path)
        if filtered_path.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(filtered_path)
        else:
            df = pd.read_csv(filtered_path)

    df = normalize_vsx_df(df)

    # Filter to passing candidates
    if require_failed_any_false and "failed_any" in df.columns:
        df = df[~df["failed_any"]].copy()
    elif require_failed_any_false:
        print("Warning: 'failed_any' column not found, using all rows")

    # Include only rows where all required flags are True
    if require_flags:
        for flag_col in require_flags:
            if flag_col not in df.columns:
                print(f"Warning: required flag column '{flag_col}' not found; skipping this requirement")
                continue
            df = df[df[flag_col].fillna(False)].copy()

    # Exclude rows where any exclude flag is True
    if exclude_flags:
        for flag_col in exclude_flags:
            if flag_col not in df.columns:
                print(f"Warning: exclude flag column '{flag_col}' not found; skipping this exclusion")
                continue
            df = df[~df[flag_col].fillna(False)].copy()

    # Optional quantitative periodicity filters
    if min_lsp_power is not None:
        if "lsp_power" in df.columns:
            df = df[df["lsp_power"].fillna(-np.inf) >= float(min_lsp_power)].copy()
        else:
            print("Warning: 'lsp_power' column not found; skipping --min-lsp-power")

    if max_lsp_bootstrap_sig is not None:
        if "lsp_bootstrap_sig" in df.columns:
            df = df[df["lsp_bootstrap_sig"].fillna(np.inf) <= float(max_lsp_bootstrap_sig)].copy()
        else:
            print("Warning: 'lsp_bootstrap_sig' column not found; skipping --max-lsp-bootstrap-sig")

    if min_periodicity_score is not None:
        if "periodicity_score" in df.columns:
            df = df[df["periodicity_score"].fillna(-np.inf) >= float(min_periodicity_score)].copy()
        else:
            print("Warning: 'periodicity_score' column not found; skipping --min-periodicity-score")

    # Deduplicate by path
    if "path" in df.columns:
        df = df.drop_duplicates(subset=["path"])

    if max_plots is not None:
        df = df.head(max_plots)

    return df.reset_index(drop=True)


def _plot_single_candidate(args: tuple) -> tuple[str, str, bool, str, str]:
    """Worker function for parallel plotting."""
    (
        lc_path_str, out_path_str, baseline, baseline_kwargs,
        skip_events, plot_fits, logbf_threshold_dip, logbf_threshold_jump,
        jd_offset, clean_max_error_absolute, clean_max_error_sigma,
        detection_results_csv, annotations, metadata, run_params,
        filter_bad_cameras, bad_camera_scatter_ratio
    ) = args

    lc_path = Path(lc_path_str)
    out_path = Path(out_path_str)

    if not lc_path.exists():
        return (lc_path_str, out_path_str, False, "file not found", "")

    try:
        baseline_func = BASELINE_FUNCTIONS.get(baseline, BASELINE_FUNCTIONS["per_camera_gp"])
        filtered_cams = plot_bayes_results(
            lc_path,
            out_path=out_path,
            show=False,
            baseline_func=baseline_func,
            baseline_kwargs=baseline_kwargs or {},
            skip_events=skip_events,
            plot_fits=plot_fits,
            logbf_threshold_dip=logbf_threshold_dip,
            logbf_threshold_jump=logbf_threshold_jump,
            jd_offset=jd_offset,
            clean_max_error_absolute=clean_max_error_absolute,
            clean_max_error_sigma=clean_max_error_sigma,
            detection_results_csv=detection_results_csv,
            annotations=annotations,
            metadata=metadata,
            run_params=run_params,
            filter_bad_cameras=filter_bad_cameras,
            bad_camera_scatter_ratio=bad_camera_scatter_ratio,
            return_filtered_cameras=True,
        )
        # Format filtered cameras as comma-separated string
        filtered_str = ",".join(str(c) for c in sorted(filtered_cams)) if filtered_cams else ""
        return (lc_path_str, out_path_str, True, "", filtered_str)
    except Exception as e:
        return (lc_path_str, out_path_str, False, str(e), "")


def _as_bool(v: object) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer, float, np.floating)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


def _candidate_bucket(row: pd.Series) -> str:
    dip_sig = _as_bool(row.get("dip_significant"))
    jump_sig = _as_bool(row.get("jump_significant"))

    if not dip_sig and not jump_sig and "event_type" in row.index and pd.notna(row.get("event_type")):
        event_type = str(row.get("event_type")).lower()
        dip_sig = ("dip" in event_type) or (event_type == "either")
        jump_sig = ("jump" in event_type) or (event_type == "either")

    if dip_sig and jump_sig:
        return "both"
    if jump_sig:
        return "jump"
    return "dip"


def plot_passing_candidates(
    filtered_path: Path | pd.DataFrame,
    out_dir: Path,
    *,
    require_failed_any_false: bool = True,
    require_flags: list[str] | None = None,
    exclude_flags: list[str] | None = None,
    min_lsp_power: float | None = None,
    max_lsp_bootstrap_sig: float | None = None,
    min_periodicity_score: float | None = None,
    max_plots: int | None = None,
    baseline: str = "per_camera_gp",
    baseline_kwargs: dict | None = None,
    skip_events: bool = False,
    plot_fits: bool = False,
    format: str = "png",
    show: bool = False,
    verbose: bool = False,
    workers: int = 1,
    logbf_threshold_dip: float = 5.0,
    logbf_threshold_jump: float = 5.0,
    jd_offset: float = 2458000.0,
    clean_max_error_absolute: float = 1.0,
    clean_max_error_sigma: float = 5.0,
    detection_results_csv: Path | None = None,
    run_params: dict | None = None,
    filter_bad_cameras: bool = True,
    bad_camera_scatter_ratio: float = 2.5,
    show_tqdm: bool = True,
) -> dict[str, object]:
    """
    Plot all candidates that passed post-filters.

    Parameters
    ----------
    filtered_path : Path
        Path to filtered events results
    out_dir : Path
        Output directory for plots
    max_plots : int | None
        Maximum number of plots to generate
    baseline : str
        Baseline function name
    baseline_kwargs : dict | None
        Additional kwargs for baseline function
    skip_events : bool
        Skip event detection, just plot baseline/residuals
    plot_fits : bool
        Overlay fit curves on plots
    format : str
        Output format (png/pdf)
    show : bool
        Show plots interactively
    verbose : bool
        Print progress details
    logbf_threshold_dip : float
        Log BF threshold for dips
    logbf_threshold_jump : float
        Log BF threshold for jumps
    jd_offset : float
        JD offset for plotting
    clean_max_error_absolute : float
        Absolute error cutoff for cleaning
    clean_max_error_sigma : float
        Sigma cutoff for MAD filter
    detection_results_csv : Path | None
        Optional detection results CSV for metadata lookup

    Returns
    -------
    dict[str, object]
        Summary with plotted/failed counts and filtered camera details.
    """
    df = load_passing_candidates(
        filtered_path,
        require_failed_any_false=require_failed_any_false,
        require_flags=require_flags,
        exclude_flags=exclude_flags,
        min_lsp_power=min_lsp_power,
        max_lsp_bootstrap_sig=max_lsp_bootstrap_sig,
        min_periodicity_score=min_periodicity_score,
        max_plots=max_plots,
    )

    total_selected = len(df)

    if df.empty:
        print("No passing candidates found")
        return {
            "total_selected": 0,
            "plotted": 0,
            "failed": 0,
            "failed_paths": [],
            "filtered_camera_sources": 0,
            "filtered_cameras_by_path": {},
        }

    print(f"Found {len(df)} candidates passing all filters")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if baseline_kwargs is None:
        baseline_kwargs = {}

    bucket_dirs = {
        "dip": out_dir / "dip",
        "jump": out_dir / "jump",
        "both": out_dir / "both",
    }
    for d in bucket_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Build work items
    work_items = []
    manifest_rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        row_dict = normalize_vsx_record({k: row[k] for k in row.index})
        lc_path = Path(row["path"])
        asas_sn_id = lc_path.stem.split("-")[0]
        bucket = _candidate_bucket(row)
        out_path = bucket_dirs[bucket] / f"{asas_sn_id}_candidate.{format}"

        # Build annotations from filter results
        annotations = {}
        if "dip_bayes_factor" in row.index:
            annotations["dip_logBF"] = f"{row['dip_bayes_factor']:.1f}" if pd.notna(row["dip_bayes_factor"]) else "N/A"
        if "jump_bayes_factor" in row.index:
            annotations["jump_logBF"] = f"{row['jump_bayes_factor']:.1f}" if pd.notna(row["jump_bayes_factor"]) else "N/A"
        if "ruwe" in row.index and pd.notna(row["ruwe"]):
            annotations["RUWE"] = f"{row['ruwe']:.2f}"
        if "catalog_match" in row.index:
            annotations["periodic"] = "Yes" if row["catalog_match"] else "No"
        if "lsp_period" in row.index and pd.notna(row["lsp_period"]):
            annotations["LSP_period_d"] = f"{row['lsp_period']:.4f}"
        if "lsp_power" in row.index and pd.notna(row["lsp_power"]):
            annotations["LSP_power"] = f"{row['lsp_power']:.4f}"
        if "lsp_bootstrap_sig" in row.index and pd.notna(row["lsp_bootstrap_sig"]):
            annotations["LSP_boot_sig"] = f"{row['lsp_bootstrap_sig']:.4g}"
        if "periodicity_score" in row.index and pd.notna(row["periodicity_score"]):
            annotations["periodicity_score"] = f"{row['periodicity_score']:.3f}"

        # Add dipper/jumper scores to annotations
        if "dipper_score" in row.index and pd.notna(row["dipper_score"]):
            annotations["dipper_score"] = f"{row['dipper_score']:.2f}"
        if "jumper_score" in row.index and pd.notna(row["jumper_score"]):
            annotations["jumper_score"] = f"{row['jumper_score']:.2f}"

        # Morphology info
        if "dip_best_morph" in row.index and pd.notna(row["dip_best_morph"]):
            annotations["dip_morph"] = str(row["dip_best_morph"])
        if "jump_best_morph" in row.index and pd.notna(row["jump_best_morph"]):
            annotations["jump_morph"] = str(row["jump_best_morph"])

        # Run info
        if "dip_run_count" in row.index and pd.notna(row["dip_run_count"]):
            annotations["dip_runs"] = str(int(row["dip_run_count"]))
        if "jump_run_count" in row.index and pd.notna(row["jump_run_count"]):
            annotations["jump_runs"] = str(int(row["jump_run_count"]))

        # Coordinates
        if "ra_deg" in row.index and pd.notna(row["ra_deg"]):
            annotations["RA"] = f"{row['ra_deg']:.5f}"
        if "dec_deg" in row.index and pd.notna(row["dec_deg"]):
            annotations["Dec"] = f"{row['dec_deg']:.5f}"

        # Gaia ID
        if "gaia_id" in row.index and pd.notna(row["gaia_id"]):
            annotations["Gaia_ID"] = str(int(row["gaia_id"]))

        # Build metadata from row using shared review/plot schema
        metadata = {}
        for _, key in REVIEW_METADATA_FIELDS:
            if key in row_dict and pd.notna(row_dict[key]) and row_dict[key] != "":
                metadata[key] = row_dict[key]

        metadata["plot_bucket"] = bucket

        work_items.append((
            str(lc_path), str(out_path), baseline, baseline_kwargs,
            skip_events, plot_fits, logbf_threshold_dip, logbf_threshold_jump,
            jd_offset, clean_max_error_absolute, clean_max_error_sigma,
            str(detection_results_csv) if detection_results_csv else None,
            annotations, metadata, run_params,
            filter_bad_cameras, bad_camera_scatter_ratio
        ))
        manifest_rows.append(
            {
                "candidate_id": row_dict.get("candidate_id", asas_sn_id),
                "asas_sn_id": row_dict.get("asas_sn_id", asas_sn_id),
                "path": str(lc_path),
                "plot_bucket": bucket,
                "plot_path": str(out_path),
            }
        )

    n_plotted = 0
    n_failed = 0
    all_filtered_cameras: dict[str, str] = {}  # path -> filtered cameras string
    results: list[tuple[str, str, bool, str, str]] = []

    if workers > 1:
        from multiprocessing import Pool, cpu_count
        actual_workers = min(workers, cpu_count(), len(work_items))
        print(f"Plotting with {actual_workers} workers...")

        with Pool(processes=actual_workers, maxtasksperchild=50) as pool:
            results = list(tqdm(
                pool.imap_unordered(_plot_single_candidate, work_items),
                total=len(work_items),
                desc="Plotting candidates",
                disable=not show_tqdm,
            ))

        for lc_path, _, success, error, filtered_str in results:
            if success:
                n_plotted += 1
                if filtered_str:
                    all_filtered_cameras[lc_path] = filtered_str
            else:
                n_failed += 1
                if verbose:
                    print(f"Failed to plot {lc_path}: {error}")
    else:
        for item in tqdm(work_items, desc="Plotting candidates", disable=not show_tqdm):
            lc_path, out_path_str, success, error, filtered_str = _plot_single_candidate(item)
            results.append((lc_path, out_path_str, success, error, filtered_str))
            if success:
                n_plotted += 1
                if filtered_str:
                    all_filtered_cameras[lc_path] = filtered_str
            else:
                n_failed += 1
                if verbose:
                    print(f"Failed to plot {lc_path}: {error}")

    # Report filtered cameras
    if all_filtered_cameras:
        print(f"\nFiltered cameras summary ({len(all_filtered_cameras)} light curves had bad cameras removed):")
        for lc_path, cams in sorted(all_filtered_cameras.items())[:20]:
            print(f"  {Path(lc_path).name}: cameras {cams}")
        if len(all_filtered_cameras) > 20:
            print(f"  ... and {len(all_filtered_cameras) - 20} more")

    print(f"\nGenerated {n_plotted} plots, {n_failed} failed")
    failed_paths = [lc_path for lc_path, _, success, _, _ in results if not success]

    result_by_path = {lc_path: (success, err) for lc_path, _, success, err, _ in results}
    manifest_df = pd.DataFrame(manifest_rows)
    if not manifest_df.empty:
        manifest_df["plot_success"] = manifest_df["path"].map(lambda p: result_by_path.get(str(p), (False, "missing"))[0])
        manifest_df["plot_error"] = manifest_df["path"].map(lambda p: result_by_path.get(str(p), (False, "missing"))[1])
        for bucket in ("dip", "jump", "both"):
            bucket_manifest = manifest_df[manifest_df["plot_bucket"] == bucket].copy()
            bucket_manifest.to_csv(out_dir / f"manifest_{bucket}.csv", index=False)

    plotted_by_bucket: dict[str, int] = {"dip": 0, "jump": 0, "both": 0}
    if not manifest_df.empty:
        bucket_counts = manifest_df[manifest_df["plot_success"]]["plot_bucket"].value_counts().to_dict()
        for bucket, count in bucket_counts.items():
            plotted_by_bucket[str(bucket)] = int(count)

    return {
        "total_selected": total_selected,
        "plotted": n_plotted,
        "failed": n_failed,
        "failed_paths": failed_paths,
        "filtered_camera_sources": len(all_filtered_cameras),
        "filtered_cameras_by_path": all_filtered_cameras,
        "plotted_by_bucket": plotted_by_bucket,
        "manifest_files": {
            "dip": str(out_dir / "manifest_dip.csv"),
            "jump": str(out_dir / "manifest_jump.csv"),
            "both": str(out_dir / "manifest_both.csv"),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot light curves for candidates passing all post-filters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  malca plot --detect-run output/runs/20260128_163911
  malca plot --events results_filtered.csv --out-dir plots/
  malca plot --detect-run output/runs/20260128_163911 --max-plots 10
"""
    )

    parser.add_argument(
        "--detect-run",
        type=Path,
        default=None,
        help="Detect run directory. Reads from <detect-run>/results/*filtered* and writes to <detect-run>/plots/candidates/",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to filtered events results (CSV/Parquet). Overrides --detect-run.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots. Overrides default from --detect-run.",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=None,
        help="Maximum number of plots to generate.",
    )
    parser.add_argument(
        "--ignore-failed-any",
        action="store_true",
        help="Do not require failed_any == False before plotting.",
    )
    parser.add_argument(
        "--require-flag",
        action="append",
        default=[],
        help="Require this boolean flag column to be True (repeatable).",
    )
    parser.add_argument(
        "--exclude-flag",
        action="append",
        default=[],
        help="Exclude rows where this boolean flag column is True (repeatable).",
    )
    parser.add_argument(
        "--min-lsp-power",
        type=float,
        default=None,
        help="Require lsp_power >= this value.",
    )
    parser.add_argument(
        "--max-lsp-bootstrap-sig",
        type=float,
        default=None,
        help="Require lsp_bootstrap_sig <= this value.",
    )
    parser.add_argument(
        "--min-periodicity-score",
        type=float,
        default=None,
        help="Require periodicity_score >= this value (score = -log10 bootstrap p).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=list(BASELINE_FUNCTIONS.keys()),
        default="per_camera_gp",
        help="Baseline function to use (default: per_camera_gp)",
    )
    parser.add_argument(
        "--skip-events",
        action="store_true",
        help="Skip event detection, just plot baseline/residuals",
    )
    parser.add_argument(
        "--plot-fits",
        action="store_true",
        help="Overlay Gaussian/Paczynski fit curves",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf"),
        default="png",
        help="Output format (default: png)",
    )
    parser.add_argument(
        "--logbf-threshold-dip",
        type=float,
        default=5.0,
        help="Log BF threshold for dips (default: 5.0)",
    )
    parser.add_argument(
        "--logbf-threshold-jump",
        type=float,
        default=5.0,
        help="Log BF threshold for jumps (default: 5.0)",
    )
    parser.add_argument(
        "--jd-offset",
        type=float,
        default=2458000.0,
        help="JD offset for plotting (default: 2458000.0)",
    )
    parser.add_argument(
        "--clean-max-error-absolute",
        type=float,
        default=1.0,
        help="Absolute error cutoff for clean_lc (default: 1.0)",
    )
    parser.add_argument(
        "--clean-max-error-sigma",
        type=float,
        default=5.0,
        help="Sigma cutoff for clean_lc MAD filter (default: 5.0)",
    )
    parser.add_argument(
        "--detection-results",
        type=Path,
        default=None,
        help="Optional detection results CSV for metadata lookup",
    )
    parser.add_argument(
        "--gp-sigma", type=float, default=None, help="GP sigma parameter"
    )
    parser.add_argument(
        "--gp-rho", type=float, default=None, help="GP rho parameter"
    )
    parser.add_argument(
        "--gp-jitter", type=float, default=None, help="GP jitter term"
    )
    parser.add_argument("--gp-q", type=float, default=None, help="GP quality factor q")
    parser.add_argument("--gp-s0", type=float, default=None, help="GP SHOTerm S0")
    parser.add_argument("--gp-w0", type=float, default=None, help="GP SHOTerm w0")
    parser.add_argument("--gp-sigma-floor", type=float, default=None, help="GP sigma floor")
    parser.add_argument("--gp-floor-clip", type=float, default=None, help="GP floor clipping sigma")
    parser.add_argument("--gp-floor-iters", type=int, default=None, help="GP floor clipping iterations")
    parser.add_argument("--gp-min-floor-points", type=int, default=None, help="Min points for GP floor")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    parser.add_argument("--no-tqdm", action="store_true", help="Disable progress bars")
    # Bad camera filtering
    parser.add_argument(
        "--no-filter-bad-cameras",
        dest="filter_bad_cameras",
        action="store_false",
        help="Disable auto-filtering of bad cameras (enabled by default)",
    )
    parser.add_argument(
        "--bad-camera-scatter-ratio",
        type=float,
        default=2.5,
        help="Scatter ratio threshold for bad camera filtering (default: 2.5)",
    )
    parser.set_defaults(filter_bad_cameras=True)

    args = parser.parse_args()

    # Resolve input path
    if args.input:
        input_path = args.input.expanduser()
    elif args.detect_run:
        detect_run = args.detect_run.expanduser()
        results_dir = detect_run / "results"

        # Look for filtered results
        candidates = (
            list(results_dir.glob("*_filtered.csv")) +
            list(results_dir.glob("*_filtered.parquet")) +
            list(results_dir.glob("*filtered*.csv")) +
            list(results_dir.glob("*filtered*.parquet"))
        )

        if not candidates:
            raise FileNotFoundError(f"No filtered results found in {results_dir}")

        input_path = candidates[0]
        print(f"Using filtered results: {input_path}")
    else:
        raise ValueError("Must specify either --input or --detect-run")

    # Resolve output directory
    if args.out_dir:
        out_dir = args.out_dir.expanduser()
    elif args.detect_run:
        out_dir = args.detect_run.expanduser() / "plots" / "candidates"
    else:
        out_dir = Path("plots/candidates")

    # Build baseline kwargs
    baseline_kwargs = {}
    gp_params = {
        "sigma": args.gp_sigma,
        "rho": args.gp_rho,
        "q": args.gp_q,
        "S0": args.gp_s0,
        "w0": args.gp_w0,
        "jitter": args.gp_jitter,
        "sigma_floor": args.gp_sigma_floor,
        "floor_clip": args.gp_floor_clip,
        "floor_iters": args.gp_floor_iters,
        "min_floor_points": args.gp_min_floor_points,
    }
    gp_params = {k: v for k, v in gp_params.items() if v is not None}
    if gp_params and args.baseline.startswith("per_camera_gp"):
        baseline_kwargs.update(gp_params)

    # Load run_params.json if available from detect_run
    run_params = None
    if args.detect_run:
        import json
        run_params_path = args.detect_run.expanduser() / "run_params.json"
        if run_params_path.exists():
            try:
                with open(run_params_path) as f:
                    run_params = json.load(f)
                print(f"Loaded run params from: {run_params_path}")
            except Exception as e:
                print(f"Warning: Could not load run_params.json: {e}")

    # Plot
    summary = plot_passing_candidates(
        input_path,
        out_dir,
        require_failed_any_false=not args.ignore_failed_any,
        require_flags=args.require_flag,
        exclude_flags=args.exclude_flag,
        min_lsp_power=args.min_lsp_power,
        max_lsp_bootstrap_sig=args.max_lsp_bootstrap_sig,
        min_periodicity_score=args.min_periodicity_score,
        max_plots=args.max_plots,
        baseline=args.baseline,
        baseline_kwargs=baseline_kwargs,
        skip_events=args.skip_events,
        plot_fits=args.plot_fits,
        format=args.format,
        show=args.show,
        verbose=args.verbose,
        workers=args.workers,
        logbf_threshold_dip=args.logbf_threshold_dip,
        logbf_threshold_jump=args.logbf_threshold_jump,
        jd_offset=args.jd_offset,
        clean_max_error_absolute=args.clean_max_error_absolute,
        clean_max_error_sigma=args.clean_max_error_sigma,
        detection_results_csv=args.detection_results,
        run_params=run_params,
        filter_bad_cameras=args.filter_bad_cameras,
        bad_camera_scatter_ratio=args.bad_camera_scatter_ratio,
        show_tqdm=not args.no_tqdm,
    )

    # Write run config next to generated plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    orig_argv = getattr(sys, "orig_argv", None)
    cmd = shlex.join(orig_argv) if orig_argv else shlex.join([sys.executable] + sys.argv)
    config = {
        "timestamp": timestamp,
        "command": cmd,
        "input_file": str(input_path),
        "output_dir": str(out_dir),
        "params": {
            "require_failed_any_false": not args.ignore_failed_any,
            "require_flags": args.require_flag,
            "exclude_flags": args.exclude_flag,
            "min_lsp_power": args.min_lsp_power,
            "max_lsp_bootstrap_sig": args.max_lsp_bootstrap_sig,
            "min_periodicity_score": args.min_periodicity_score,
            "max_plots": args.max_plots,
            "baseline": args.baseline,
            "baseline_kwargs": baseline_kwargs if baseline_kwargs else None,
            "skip_events": args.skip_events,
            "plot_fits": args.plot_fits,
            "format": args.format,
            "show": args.show,
            "workers": args.workers,
            "logbf_threshold_dip": args.logbf_threshold_dip,
            "logbf_threshold_jump": args.logbf_threshold_jump,
            "jd_offset": args.jd_offset,
            "clean_max_error_absolute": args.clean_max_error_absolute,
            "clean_max_error_sigma": args.clean_max_error_sigma,
            "detection_results": str(args.detection_results) if args.detection_results else None,
            "filter_bad_cameras": args.filter_bad_cameras,
            "bad_camera_scatter_ratio": args.bad_camera_scatter_ratio,
            "show_tqdm": not args.no_tqdm,
        },
        "summary": summary,
    }
    config_path = out_dir / "plot_candidates_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"\nPlots saved to {out_dir}")
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
