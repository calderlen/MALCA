"""
Postprocessing module: plot light curves that pass all post-filters.

Reads post_filter output (with failed_any column), selects passing candidates,
and generates light curve plots via plot.py into a timestamped output directory
with a config file recording all parameters.

Usage:
    malca postprocess --input results_filtered.parquet
    malca postprocess --detect-run output/runs/20250121_143052
    malca postprocess --input results_filtered.csv --baseline per_camera_gp --plot-fits
"""

import argparse
import json
import sys
import shlex
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import re

import numpy as np

from malca.plot import plot_bayes_results, BASELINE_FUNCTIONS


def _build_annotations(row: pd.Series, path: str) -> dict[str, str]:
    """Build plot annotations dict from a post-filter DataFrame row."""
    ann = {}

    # File path (truncated to last 2 components for readability)
    parts = Path(path).parts
    ann["path"] = "/".join(parts[-2:]) if len(parts) >= 2 else path

    # Score
    score = row.get("dipper_score")
    if score is not None and np.isfinite(score):
        ann["score"] = f"{score:.2f}"

    # Mag bin (parse from path)
    mag_match = re.search(r"(\d+(?:\.\d+)?_\d+(?:\.\d+)?)", path)
    if mag_match:
        ann["mag_bin"] = mag_match.group(1)

    # Morphology (best dip/jump morph)
    dip_morph = row.get("dip_best_morph")
    jump_morph = row.get("jump_best_morph")
    morph_parts = []
    if dip_morph and isinstance(dip_morph, str) and dip_morph != "none":
        morph_parts.append(f"dip:{dip_morph}")
    if jump_morph and isinstance(jump_morph, str) and jump_morph != "none":
        morph_parts.append(f"jump:{jump_morph}")
    if morph_parts:
        ann["morph"] = ", ".join(morph_parts)

    # VSX class (from pre_filter crossmatch)
    vsx_class = row.get("vsx_class")
    if vsx_class and isinstance(vsx_class, str) and vsx_class.strip():
        ann["vsx"] = vsx_class.strip()

    # Periodic catalog match
    cat_class = row.get("catalog_class")
    if cat_class and isinstance(cat_class, str) and cat_class.strip():
        cat_period = row.get("catalog_period")
        label = cat_class.strip()
        if cat_period is not None and np.isfinite(cat_period):
            label += f" (P={cat_period:.2f}d)"
        ann["catalog"] = label

    # Gaia RUWE
    ruwe = row.get("ruwe")
    if ruwe is not None and np.isfinite(ruwe):
        ann["RUWE"] = f"{ruwe:.2f}"

    # LSP periodicity
    lsp_period = row.get("lsp_period")
    lsp_sig = row.get("lsp_bootstrap_sig")
    if lsp_period is not None and np.isfinite(lsp_period):
        label = f"{lsp_period:.2f}d"
        if lsp_sig is not None and np.isfinite(lsp_sig):
            label += f" (p={lsp_sig:.3f})"
        ann["LSP"] = label

    return ann


def run_postprocess(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    baseline: str = "per_camera_gp",
    baseline_kwargs: dict | None = None,
    logbf_threshold_dip: float = 5.0,
    logbf_threshold_jump: float = 5.0,
    skip_events: bool = False,
    plot_fits: bool = False,
    plot_format: str = "png",
    jd_offset: float = 2458000.0,
    detection_results_csv: Path | None = None,
    clean_max_error_absolute: float = 1.0,
    clean_max_error_sigma: float = 5.0,
    max_plots: int | None = None,
    show_tqdm: bool = True,
) -> dict:
    """
    Plot light curves for all candidates that passed post-filters.

    Parameters
    ----------
    df : pd.DataFrame
        Post-filter output with 'path' and 'failed_any' columns.
    out_dir : Path
        Output directory for plots and config.
    baseline : str
        Baseline function name (key in BASELINE_FUNCTIONS).
    baseline_kwargs : dict | None
        Extra kwargs passed to baseline function (e.g. GP params).
    logbf_threshold_dip : float
        Log BF threshold for dip detection in plots.
    logbf_threshold_jump : float
        Log BF threshold for jump detection in plots.
    skip_events : bool
        Skip event overlay, plot baseline/residuals only.
    plot_fits : bool
        Overlay morphology fit curves on events.
    plot_format : str
        Output format ('png' or 'pdf').
    jd_offset : float
        JD offset for x-axis.
    detection_results_csv : Path | None
        Optional detection results CSV for metadata lookup.
    clean_max_error_absolute : float
        Absolute error cutoff for cleaning.
    clean_max_error_sigma : float
        Sigma cutoff for MAD-based cleaning.
    max_plots : int | None
        Limit number of plots generated.
    show_tqdm : bool
        Show progress bar.

    Returns
    -------
    dict
        Summary with counts of plotted/skipped/failed candidates.
    """
    if "failed_any" not in df.columns:
        raise KeyError("Input dataframe missing 'failed_any' column. Run post_filter first.")
    if "path" not in df.columns:
        raise KeyError("Input dataframe missing 'path' column.")

    df_passed = df[~df["failed_any"]].copy()
    # Index by path for annotation lookups (keep first row per path)
    df_passed["_path_str"] = df_passed["path"].astype(str)
    row_lookup = df_passed.drop_duplicates(subset="_path_str").set_index("_path_str")
    paths = df_passed["_path_str"].dropna().unique().tolist()

    if max_plots is not None:
        paths = paths[:max_plots]

    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_func = BASELINE_FUNCTIONS.get(baseline)
    if baseline_func is None:
        raise ValueError(f"Unknown baseline '{baseline}'. Options: {list(BASELINE_FUNCTIONS.keys())}")

    bkw = baseline_kwargs or {}

    n_plotted = 0
    n_skipped = 0
    n_failed = 0
    errors = []

    iterator = tqdm(paths, desc="postprocess plots", disable=not show_tqdm)
    for lc_path_str in iterator:
        lc_path = Path(lc_path_str)
        if not lc_path.exists():
            n_skipped += 1
            continue

        asas_sn_id = lc_path.stem.split("-")[0]
        out_path = out_dir / f"{asas_sn_id}_dips.{plot_format}"

        # Build annotations from row metadata
        annotations = None
        if lc_path_str in row_lookup.index:
            annotations = _build_annotations(row_lookup.loc[lc_path_str], lc_path_str)

        try:
            plot_bayes_results(
                lc_path,
                out_path=out_path,
                show=False,
                baseline_func=baseline_func,
                baseline_kwargs=bkw,
                logbf_threshold_dip=logbf_threshold_dip,
                logbf_threshold_jump=logbf_threshold_jump,
                skip_events=skip_events,
                plot_fits=plot_fits,
                jd_offset=jd_offset,
                detection_results_csv=detection_results_csv,
                clean_max_error_absolute=clean_max_error_absolute,
                clean_max_error_sigma=clean_max_error_sigma,
                annotations=annotations,
            )
            n_plotted += 1
        except Exception as e:
            n_failed += 1
            errors.append({"path": str(lc_path), "error": str(e)})
            iterator.set_postfix_str(f"err: {asas_sn_id}")

    return {
        "total_passed": len(paths),
        "plotted": n_plotted,
        "skipped_missing": n_skipped,
        "failed": n_failed,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot light curves that pass all post-filters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  malca postprocess --input results_filtered.parquet
  malca postprocess --detect-run output/runs/20250121_143052
  malca postprocess --input results_filtered.csv --baseline per_camera_gp --plot-fits --max-plots 50
""",
    )

    # I/O
    parser.add_argument(
        "--detect-run", type=Path, default=None,
        help="Detect run directory. Reads filtered results from <detect-run>/results/ and writes plots to a timestamped subfolder.",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Input filtered CSV/Parquet (must have 'failed_any' column). Overrides --detect-run.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory for plots. Default: timestamped folder in CWD or detect-run.",
    )

    # Plot parameters
    parser.add_argument(
        "--baseline", type=str, choices=list(BASELINE_FUNCTIONS.keys()),
        default="per_camera_gp", help="Baseline function (default: per_camera_gp)",
    )
    parser.add_argument("--logbf-threshold-dip", type=float, default=5.0)
    parser.add_argument("--logbf-threshold-jump", type=float, default=5.0)
    parser.add_argument("--skip-events", action="store_true", help="Skip event overlay")
    parser.add_argument("--plot-fits", action="store_true", help="Overlay morphology fits")
    parser.add_argument(
        "--format", choices=("png", "pdf"), default="png",
        help="Plot output format (default: png)",
    )
    parser.add_argument("--jd-offset", type=float, default=2458000.0)
    parser.add_argument("--clean-max-error-absolute", type=float, default=1.0)
    parser.add_argument("--clean-max-error-sigma", type=float, default=5.0)
    parser.add_argument("--max-plots", type=int, default=None, help="Max plots to generate")
    parser.add_argument(
        "--detection-results", type=Path, default=None,
        help="Detection results CSV for metadata lookup in plot titles",
    )

    # GP parameters (forwarded to baseline)
    parser.add_argument("--gp-sigma", type=float, default=None)
    parser.add_argument("--gp-rho", type=float, default=None)
    parser.add_argument("--gp-q", type=float, default=None)
    parser.add_argument("--gp-s0", type=float, default=None)
    parser.add_argument("--gp-w0", type=float, default=None)
    parser.add_argument("--gp-jitter", type=float, default=None)
    parser.add_argument("--gp-sigma-floor", type=float, default=None)
    parser.add_argument("--gp-floor-clip", type=float, default=None)
    parser.add_argument("--gp-floor-iters", type=int, default=None)
    parser.add_argument("--gp-min-floor-points", type=int, default=None)

    # General
    parser.add_argument("--no-tqdm", action="store_true", help="Disable progress bar")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # Resolve input path
    if args.input:
        input_path = args.input.expanduser()
    elif args.detect_run:
        detect_run = args.detect_run.expanduser()
        results_dir = detect_run / "results"
        candidates = (
            list(results_dir.glob("*filtered.parquet"))
            + list(results_dir.glob("*filtered.csv"))
        )
        if not candidates:
            raise FileNotFoundError(f"No filtered results file found in {results_dir}")
        input_path = candidates[0]
        if args.verbose:
            print(f"Using filtered results: {input_path}")
    else:
        raise ValueError("Must specify either --input or --detect-run")

    # Load data
    if input_path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    n_total = len(df)
    n_passed = int((~df["failed_any"]).sum()) if "failed_any" in df.columns else n_total
    print(f"Loaded {n_total} rows from {input_path} ({n_passed} passed all filters)")

    # Resolve output directory (timestamped)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = args.out_dir.expanduser()
    elif args.detect_run:
        out_dir = args.detect_run.expanduser() / f"postprocess_{timestamp}"
    else:
        out_dir = Path(f"postprocess_{timestamp}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Build baseline kwargs from GP args
    baseline_kwargs = {}
    gp_kwargs = {
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
    gp_kwargs = {k: v for k, v in gp_kwargs.items() if v is not None}
    if gp_kwargs:
        baseline_kwargs.update(gp_kwargs)

    # Run postprocessing
    summary = run_postprocess(
        df,
        out_dir=out_dir,
        baseline=args.baseline,
        baseline_kwargs=baseline_kwargs if baseline_kwargs else None,
        logbf_threshold_dip=args.logbf_threshold_dip,
        logbf_threshold_jump=args.logbf_threshold_jump,
        skip_events=args.skip_events,
        plot_fits=args.plot_fits,
        plot_format=args.format,
        jd_offset=args.jd_offset,
        detection_results_csv=args.detection_results,
        clean_max_error_absolute=args.clean_max_error_absolute,
        clean_max_error_sigma=args.clean_max_error_sigma,
        max_plots=args.max_plots,
        show_tqdm=not args.no_tqdm,
    )

    # Write config file
    orig_argv = getattr(sys, "orig_argv", None)
    cmd = shlex.join(orig_argv) if orig_argv else shlex.join([sys.executable] + sys.argv)

    config = {
        "timestamp": timestamp,
        "command": cmd,
        "input_file": str(input_path),
        "output_dir": str(out_dir),
        "params": {
            "baseline": args.baseline,
            "baseline_kwargs": baseline_kwargs if baseline_kwargs else None,
            "logbf_threshold_dip": args.logbf_threshold_dip,
            "logbf_threshold_jump": args.logbf_threshold_jump,
            "skip_events": args.skip_events,
            "plot_fits": args.plot_fits,
            "format": args.format,
            "jd_offset": args.jd_offset,
            "clean_max_error_absolute": args.clean_max_error_absolute,
            "clean_max_error_sigma": args.clean_max_error_sigma,
            "max_plots": args.max_plots,
            "detection_results": str(args.detection_results) if args.detection_results else None,
        },
        "summary": summary,
    }

    config_path = out_dir / "postprocess_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Print summary
    print(f"\nPlotted: {summary['plotted']}/{summary['total_passed']}")
    if summary["skipped_missing"]:
        print(f"Skipped (file not found): {summary['skipped_missing']}")
    if summary["failed"]:
        print(f"Failed: {summary['failed']}")
        if args.verbose:
            for err in summary["errors"]:
                print(f"  {err['path']}: {err['error']}")
    print(f"Config saved: {config_path}")


if __name__ == "__main__":
    main()
