"""
Plot light curves for candidates that passed all post-filters.

Reads the filtered events results and plots only sources with failed_any == False.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

from malca.plot import plot_bayes_results, BASELINE_FUNCTIONS


def load_passing_candidates(
    filtered_path: Path,
    *,
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
    filtered_path = Path(filtered_path)

    if filtered_path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(filtered_path)
    else:
        df = pd.read_csv(filtered_path)

    # Filter to passing candidates
    if "failed_any" in df.columns:
        df = df[~df["failed_any"]].copy()
    else:
        print("Warning: 'failed_any' column not found, using all rows")

    # Deduplicate by path
    if "path" in df.columns:
        df = df.drop_duplicates(subset=["path"])

    if max_plots is not None:
        df = df.head(max_plots)

    return df.reset_index(drop=True)


def _plot_single_candidate(args: tuple) -> tuple[str, bool, str]:
    """Worker function for parallel plotting."""
    (
        lc_path_str, out_path_str, baseline, baseline_kwargs,
        skip_events, plot_fits, logbf_threshold_dip, logbf_threshold_jump,
        jd_offset, clean_max_error_absolute, clean_max_error_sigma,
        detection_results_csv, annotations, metadata, run_params
    ) = args

    lc_path = Path(lc_path_str)
    out_path = Path(out_path_str)

    if not lc_path.exists():
        return (lc_path_str, False, "file not found")

    try:
        baseline_func = BASELINE_FUNCTIONS.get(baseline, BASELINE_FUNCTIONS["per_camera_gp"])
        plot_bayes_results(
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
        )
        return (lc_path_str, True, "")
    except Exception as e:
        return (lc_path_str, False, str(e))


def plot_passing_candidates(
    filtered_path: Path,
    out_dir: Path,
    *,
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
) -> int:
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
    int
        Number of plots generated
    """
    df = load_passing_candidates(filtered_path, max_plots=max_plots)

    if df.empty:
        print("No passing candidates found")
        return 0

    print(f"Found {len(df)} candidates passing all filters")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if baseline_kwargs is None:
        baseline_kwargs = {}

    # Build work items
    work_items = []
    for _, row in df.iterrows():
        lc_path = Path(row["path"])
        asas_sn_id = lc_path.stem.split("-")[0]
        out_path = out_dir / f"{asas_sn_id}_candidate.{format}"

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

        # Add dipper/jumper scores to annotations
        if "dipper_score" in row.index and pd.notna(row["dipper_score"]):
            annotations["dipper_score"] = f"{row['dipper_score']:.2f}"
        if "jumper_score" in row.index and pd.notna(row["jumper_score"]):
            annotations["jumper_score"] = f"{row['jumper_score']:.2f}"

        # Build metadata from row
        metadata = {}
        if "vsx_class" in row.index and pd.notna(row["vsx_class"]) and row["vsx_class"]:
            metadata["vsx_class"] = str(row["vsx_class"])
        if "asas_sn_id" in row.index and pd.notna(row["asas_sn_id"]):
            metadata["asas_sn_id"] = str(row["asas_sn_id"])
        if "external_id" in row.index and pd.notna(row["external_id"]):
            metadata["external_id"] = str(row["external_id"])
        if "trigger_type" in row.index and pd.notna(row["trigger_type"]):
            metadata["trigger_type"] = str(row["trigger_type"])

        work_items.append((
            str(lc_path), str(out_path), baseline, baseline_kwargs,
            skip_events, plot_fits, logbf_threshold_dip, logbf_threshold_jump,
            jd_offset, clean_max_error_absolute, clean_max_error_sigma,
            str(detection_results_csv) if detection_results_csv else None,
            annotations, metadata, run_params
        ))

    n_plotted = 0
    n_failed = 0

    if workers > 1:
        from multiprocessing import Pool, cpu_count
        actual_workers = min(workers, cpu_count(), len(work_items))
        print(f"Plotting with {actual_workers} workers...")

        with Pool(processes=actual_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_plot_single_candidate, work_items),
                total=len(work_items),
                desc="Plotting candidates"
            ))

        for lc_path, success, error in results:
            if success:
                n_plotted += 1
            else:
                n_failed += 1
                if verbose:
                    print(f"Failed to plot {lc_path}: {error}")
    else:
        for item in tqdm(work_items, desc="Plotting candidates"):
            lc_path, success, error = _plot_single_candidate(item)
            if success:
                n_plotted += 1
            else:
                n_failed += 1
                if verbose:
                    print(f"Failed to plot {lc_path}: {error}")

    print(f"\nGenerated {n_plotted} plots, {n_failed} failed")
    return n_plotted


def main():
    parser = argparse.ArgumentParser(
        description="Plot light curves for candidates passing all post-filters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m malca.plot_candidates --detect-run output/runs/20260128_163911
  python -m malca.plot_candidates --input results_filtered.csv --out-dir plots/
  python -m malca.plot_candidates --detect-run output/runs/20260128_163911 --max-plots 10
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
        "jitter": args.gp_jitter,
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
    n_plotted = plot_passing_candidates(
        input_path,
        out_dir,
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
    )

    print(f"\nPlots saved to {out_dir}")


if __name__ == "__main__":
    main()
