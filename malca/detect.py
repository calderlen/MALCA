#!/usr/bin/env python3
"""
Wrapper script to run events.py on pre-filtered light curves.

Workflow:
1. Build/load manifest (source_id → lc_dir mapping)
2. Apply pre-filters (sparse, periodic, multi-camera)
3. Construct file paths for kept sources
4. Pass to events.py
5. [Optional] Apply post-filters (posterior strength, run robustness, etc.)
6. [Optional] Generate postprocess plots for passing candidates
7. [Optional] Run characterization (Gaia DR3 + dust extinction)
8. [Optional] Run classification (EB/CV/starspot rejection, YSO classification)
9. [Optional] Enrich passing candidates with comprehensive light curve stats

Usage:
    python -m malca detect --mag-bin 13_13.5 [options...]
    python -m malca detect --mag-bin 13_13.5 --run-post-filter --run-classify --run-enrich
"""
from __future__ import annotations

import argparse
from datetime import datetime
import shlex
import subprocess
import sys
from pathlib import Path
import pandas as pd
import tempfile

from malca.manifest import build_manifest_dataframe
from malca.pre_filter import apply_pre_filters, filter_camera_medians
from malca.post_filter import apply_post_filters
from malca.postprocess import run_postprocess
from malca.classify import compute_all_classifications
from malca.stats import compute_stats
from malca.characterize import query_gaia_by_ids, get_dust_extinction
from malca.utils import log as _log


def safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet atomically to avoid corruption on interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def default_run_dir(base_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_root / "runs" / timestamp


def clear_existing_output(path: Path | None, fmt: str) -> None:
    if path is None or (not path.exists()):
        return
    if fmt == "parquet_chunk" and path.is_dir():
        removed_any = False
        for child in path.glob("chunk_*.parquet*"):
            child.unlink()
            removed_any = True
        if removed_any:
            print(f"Overwriting existing output chunks in {path}")
        return

    path.unlink()
    print(f"Overwriting existing output file: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run events.py on pre-filtered light curves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="All other arguments are passed directly to events.py"
    )

    # Manifest/pre-filter args
    parser.add_argument("--mag-bin", required=True, nargs="+", help="Magnitude bin(s) to process")
    parser.add_argument("--index-root", type=Path, default=Path("/data/poohbah/1/assassin/rowan.90/lcsv2"),
                        help="Index root directory (contains mag_bin/index*.csv)")
    parser.add_argument("--lc-root", type=Path, default=Path("/data/poohbah/1/assassin/rowan.90/lcsv2"),
                        help="Light curve root directory (contains mag_bin/lc*_cal/)")
    parser.add_argument("--manifest-file", type=Path, default=None,
                        help="Manifest file (default: lc_manifest_{mag_bin}.parquet)")
    parser.add_argument("--filtered-file", type=Path, default=None,
                        help="Filtered manifest file (default: lc_filtered_{mag_bin}.parquet)")
    parser.add_argument("--force-manifest", action="store_true",
                        help="Force rebuild manifest even if exists")
    parser.add_argument("--force-filter", action="store_true",
                        help="Force re-run pre-filters even if filtered file exists")

    # Pre-filter args
    parser.add_argument("--min-time-span", type=float, default=100.0, help="Min time span (days)")
    parser.add_argument("--min-points-per-day", type=float, default=0.05, help="Min cadence")
    parser.add_argument("--min-cameras", type=int, default=2, help="Min cameras required")
    parser.add_argument("--skip-sparse", action="store_true", help="Skip sparse LC filter")
    parser.add_argument("--skip-multi-camera", action="store_true", help="Skip multi-camera filter")
    parser.add_argument("--skip-vsx", action="store_true", help="Skip VSX crossmatch/tagging")
    parser.add_argument("--skip-camera-median", action="store_true", help="Skip camera median filter (identifies cameras to exclude from .raw2 files)")
    parser.add_argument("--camera-median-tolerance", type=float, default=0.2, help="Tolerance beyond mag bin for camera median filter (default: 0.2 mag)")
    parser.add_argument("--vsx-max-sep", type=float, default=3.0, help="Max separation for VSX match (arcsec)")
    parser.add_argument("--vsx-mode", type=str, default="filter", choices=["tag", "filter"], help="VSX handling: tag adds sep_arcsec/class columns, filter removes matches (default: filter)")
    parser.add_argument("--vsx-crossmatch", type=Path, default=Path("input/vsx/asassn_x_vsx_matches_20250919_2252.csv"), help="Path to pre-crossmatched VSX CSV (with asas_sn_id, sep_arcsec, class)")
    parser.add_argument("--pass-all-prefilters", action="store_true", help="Pass all light curves to events.py regardless of pre-filter results (tags are still added)")
    parser.add_argument("--enforce-filters", type=str, default=None, help="Comma-separated list of pre-filters to enforce (e.g., 'sparse,multi_camera'). " "Only rows failing these filters are excluded. Default: enforce all enabled filters.")
    parser.add_argument("--workers", type=int, default=10, help="Workers for parallel processing")
    parser.add_argument("--stats-chunk-size", type=int, default=5000, help="Rows per checkpoint save during stats computation")
    parser.add_argument("--batch-size", type=int, default=2000, help="Max light curves per events.py call")

    # events.py args
    parser.add_argument("--trigger-mode", type=str, default="posterior_prob", choices=["logbf", "posterior_prob"], help="Triggering mode")
    parser.add_argument("--logbf-threshold-dip", type=float, default=5.0, help="Per-point dip trigger threshold")
    parser.add_argument("--logbf-threshold-jump", type=float, default=5.0, help="Per-point jump trigger threshold")
    parser.add_argument("--significance-threshold", type=float, default=99.99997, help="Posterior probability threshold (if trigger-mode=posterior_prob)")
    parser.add_argument("--p-points", type=int, default=12, help="Number of points in the p grid")
    parser.add_argument("--mag-points", type=int, default=12, help="Number of points in the magnitude grid")
    parser.add_argument("--run-min-points", type=int, default=2, help="Min triggered points in a run")
    parser.add_argument("--run-max-gap-points", type=int, default=5, help="Allow up to this many missing indices inside a run")
    parser.add_argument("--run-max-gap-days", type=float, default=None, help="Break runs if JD gap exceeds this")
    parser.add_argument("--run-min-duration-days", type=float, default=0.0, help="Require run duration >= this (default: 0.0 = disabled)")
    parser.add_argument("--no-event-prob", action="store_true", help="Skip LOO event responsibilities")
    parser.add_argument("--p-min-dip", type=float, default=None, help="Minimum dip fraction for p-grid")
    parser.add_argument("--p-max-dip", type=float, default=None, help="Maximum dip fraction for p-grid")
    parser.add_argument("--p-min-jump", type=float, default=None, help="Minimum jump fraction for p-grid")
    parser.add_argument("--p-max-jump", type=float, default=None, help="Maximum jump fraction for p-grid")
    parser.add_argument(
        "--baseline-func",
        type=str,
        default="gp",
        choices=["gp", "gp_masked", "global_median", "per_camera_median"],
        help="Baseline function",
    )
    # Baseline kwargs (GP kernel parameters)
    parser.add_argument("--baseline-s0", type=float, default=0.0005, help="GP kernel S0 parameter (default: 0.0005)")
    parser.add_argument("--baseline-w0", type=float, default=0.0031415926535897933, help="GP kernel w0 parameter (default: pi/1000)")
    parser.add_argument("--baseline-q", type=float, default=0.7, help="GP kernel Q parameter (default: 0.7)")
    parser.add_argument("--baseline-jitter", type=float, default=0.006, help="GP jitter term (default: 0.006)")
    parser.add_argument("--baseline-sigma-floor", type=float, default=None, help="Minimum sigma floor (default: None)")
    # Magnitude grid bounds (override auto-detection)
    parser.add_argument("--mag-min-dip", type=float, default=None, help="Min magnitude for dip grid (overrides auto)")
    parser.add_argument("--mag-max-dip", type=float, default=None, help="Max magnitude for dip grid (overrides auto)")
    parser.add_argument("--mag-min-jump", type=float, default=None, help="Min magnitude for jump grid (overrides auto)")
    parser.add_argument("--mag-max-jump", type=float, default=None, help="Max magnitude for jump grid (overrides auto)")
    parser.add_argument("--min-mag-offset", type=float, default=0.1, help="Require |event_mag - baseline_mag| > threshold")
    parser.add_argument("--output", type=str, default=None, help="Output path for results (default: <out_dir>/lc_events_results.csv)")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory for all outputs (default: output/runs/<timestamp>)")
    parser.add_argument("--output-format", type=str, default="csv", choices=["csv", "parquet", "parquet_chunk"], help="Output format")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Write results in chunks of this many rows")

    # Step 5: Post-filter args (enabled by default)
    parser.add_argument("--run-post-filter", dest="run_post_filter", action="store_true", help="Run post_filter after events.py completes (default: enabled)")
    parser.add_argument("--no-run-post-filter", dest="run_post_filter", action="store_false", help="Skip post-filter step")
    parser.add_argument("--min-bayes-factor", type=float, default=10.0, help="Min Bayes factor for post-filter (default: 10.0)")
    parser.add_argument("--post-filter-min-run-cameras", type=int, default=2, help="Min cameras for run robustness filter (default: 2)")
    parser.add_argument("--post-filter-min-run-points", type=int, default=2, help="Min points per run for robustness filter (default: 2)")

    # Step 6: Postprocess args (enabled by default)
    parser.add_argument("--run-postprocess", dest="run_postprocess", action="store_true", help="Run postprocess (generate plots) after post_filter (default: enabled)")
    parser.add_argument("--no-run-postprocess", dest="run_postprocess", action="store_false", help="Skip postprocess step")
    parser.add_argument("--max-plots", type=int, default=None, help="Limit number of plots generated (default: no limit)")
    parser.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf"], help="Output format for plots (default: png)")

    # Step 7: Characterization args (enabled by default)
    parser.add_argument("--run-characterize", dest="run_characterize", action="store_true", help="Run Gaia DR3 characterization after post_filter (default: enabled)")
    parser.add_argument("--no-run-characterize", dest="run_characterize", action="store_false", help="Skip characterization step")
    parser.add_argument("--gaia-cache", type=Path, default=None, help="Path to Gaia query cache file (parquet). Default: <out_dir>/gaia_cache/gaia_cache.parquet")
    parser.add_argument(
        "--index-file",
        type=Path,
        default=Path("output/asassn_index_masked_concat_cleaned_20250919_154524_brotli.parquet"),
        help="Path to ASAS-SN index file with gaia_id, ra_deg, dec_deg columns",
    )
    parser.add_argument("--run-dust", dest="run_dust", action="store_true", help="Run 3D dust extinction correction (default: enabled)")
    parser.add_argument("--no-run-dust", dest="run_dust", action="store_false", help="Skip dust extinction step")

    # Step 8: Classify args (enabled by default)
    parser.add_argument("--run-classify", dest="run_classify", action="store_true", help="Run classification (EB/CV/starspot rejection, YSO) (default: enabled)")
    parser.add_argument("--no-run-classify", dest="run_classify", action="store_false", help="Skip classification step")

    # Step 9: Enrich args (enabled by default)
    parser.add_argument("--run-enrich", dest="run_enrich", action="store_true", help="Enrich passing candidates with comprehensive light curve stats (default: enabled)")
    parser.add_argument("--no-run-enrich", dest="run_enrich", action="store_false", help="Skip enrichment step")
    parser.add_argument("--enrich-compute-ls", action="store_true", help="Include Lomb-Scargle periodogram in enrichment (expensive)")

    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite checkpoint log and existing output if present (start fresh).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    parser.set_defaults(
        run_post_filter=True,
        run_postprocess=True,
        run_characterize=True,
        run_dust=True,
        run_classify=True,
        run_enrich=True,
    )

    args = parser.parse_args()

    # Build events.py args from parsed arguments
    events_args = []
    if args.verbose:
        events_args.append("--verbose")
    events_args.extend(["--workers", str(args.workers)])
    events_args.extend(["--trigger-mode", args.trigger_mode])
    events_args.extend(["--logbf-threshold-dip", str(args.logbf_threshold_dip)])
    events_args.extend(["--logbf-threshold-jump", str(args.logbf_threshold_jump)])
    events_args.extend(["--significance-threshold", str(args.significance_threshold)])
    events_args.extend(["--p-points", str(args.p_points)])
    events_args.extend(["--mag-points", str(args.mag_points)])
    events_args.extend(["--run-min-points", str(args.run_min_points)])
    events_args.extend(["--run-max-gap-points", str(args.run_max_gap_points)])
    if args.run_max_gap_days is not None:
        events_args.extend(["--run-max-gap-days", str(args.run_max_gap_days)])
    if args.run_min_duration_days is not None:
        events_args.extend(["--run-min-duration-days", str(args.run_min_duration_days)])
    if args.no_event_prob:
        events_args.append("--no-event-prob")
    if args.p_min_dip is not None:
        events_args.extend(["--p-min-dip", str(args.p_min_dip)])
    if args.p_max_dip is not None:
        events_args.extend(["--p-max-dip", str(args.p_max_dip)])
    if args.p_min_jump is not None:
        events_args.extend(["--p-min-jump", str(args.p_min_jump)])
    if args.p_max_jump is not None:
        events_args.extend(["--p-max-jump", str(args.p_max_jump)])
    events_args.extend(["--baseline-func", args.baseline_func])
    # Baseline kwargs
    events_args.extend(["--baseline-s0", str(args.baseline_s0)])
    events_args.extend(["--baseline-w0", str(args.baseline_w0)])
    events_args.extend(["--baseline-q", str(args.baseline_q)])
    events_args.extend(["--baseline-jitter", str(args.baseline_jitter)])
    if args.baseline_sigma_floor is not None:
        events_args.extend(["--baseline-sigma-floor", str(args.baseline_sigma_floor)])
    # Magnitude grid bounds
    if args.mag_min_dip is not None:
        events_args.extend(["--mag-min-dip", str(args.mag_min_dip)])
    if args.mag_max_dip is not None:
        events_args.extend(["--mag-max-dip", str(args.mag_max_dip)])
    if args.mag_min_jump is not None:
        events_args.extend(["--mag-min-jump", str(args.mag_min_jump)])
    if args.mag_max_jump is not None:
        events_args.extend(["--mag-max-jump", str(args.mag_max_jump)])
    events_args.extend(["--min-mag-offset", str(args.min_mag_offset)])
    events_args.extend(["--output-format", args.output_format])
    events_args.extend(["--chunk-size", str(args.chunk_size)])

    quiet = not bool(args.verbose)

    def log(message: str) -> None:
        _log(message, quiet=quiet)

    # Determine file names
    mag_bin_tag = args.mag_bin[0] if len(args.mag_bin) == 1 else "multi"

    # IMPORTANT: never write to filesystem root (/output). Default to a writable directory.
    events_format = str(args.output_format).lower()
    base_output_root = Path("/home/lenhart.106/code/malca/output")
    if args.out_dir is not None:
        out_dir = Path(args.out_dir).expanduser()
    elif args.filtered_file is not None:
        out_dir = Path(args.filtered_file).expanduser().parent
    elif args.manifest_file is not None:
        out_dir = Path(args.manifest_file).expanduser().parent
    elif args.output is not None:
        out_dir = Path(args.output).expanduser().parent
    else:
        out_dir = default_run_dir(base_output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gaia_cache is None:
        gaia_cache_dir = out_dir / "gaia_cache"
        gaia_cache_dir.mkdir(parents=True, exist_ok=True)
        args.gaia_cache = gaia_cache_dir / "gaia_cache.parquet"

    manifests_dir = out_dir / "manifests"
    prefilter_dir = out_dir / "prefilter"
    paths_dir = out_dir / "paths"
    results_dir = out_dir / "results"
    for d in (manifests_dir, prefilter_dir, paths_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        events_output = results_dir / "lc_events_results.csv"
    else:
        events_output = Path(args.output).expanduser()
        if args.out_dir is not None and not events_output.is_absolute():
            events_output = out_dir / events_output
        elif args.out_dir is None and not events_output.is_absolute():
            events_output = out_dir / events_output

    events_args.extend(["--output", str(events_output)])

    manifest_file = Path(args.manifest_file).expanduser() if args.manifest_file else (manifests_dir / f"lc_manifest_{mag_bin_tag}.parquet")
    filtered_file = Path(args.filtered_file).expanduser() if args.filtered_file else (prefilter_dir / f"lc_filtered_{mag_bin_tag}.parquet")
    stats_checkpoint_file = prefilter_dir / f"lc_stats_checkpoint_{mag_bin_tag}.parquet"

    # Save run parameters to JSON for full reproducibility
    import json
    run_start_time = datetime.now()

    run_params_file = out_dir / "run_params.json"
    try:
        orig_argv = getattr(sys, "orig_argv", None)
        cmd = shlex.join(orig_argv) if orig_argv else shlex.join([sys.executable] + sys.argv)

        run_params = {
            "timestamp": run_start_time.isoformat(),
            "command": cmd,
            "mag_bin": args.mag_bin,
            # Pre-filter parameters
            "min_time_span": args.min_time_span,
            "min_points_per_day": args.min_points_per_day,
            "min_cameras": args.min_cameras,
            "skip_sparse": args.skip_sparse,
            "skip_multi_camera": args.skip_multi_camera,
            "skip_vsx": args.skip_vsx,
            "vsx_max_sep": args.vsx_max_sep,
            "vsx_mode": args.vsx_mode,
            "vsx_crossmatch": str(args.vsx_crossmatch),
            # Detection parameters
            "trigger_mode": args.trigger_mode,
            "logbf_threshold_dip": args.logbf_threshold_dip,
            "logbf_threshold_jump": args.logbf_threshold_jump,
            "significance_threshold": args.significance_threshold,
            "p_points": args.p_points,
            "p_min_dip": args.p_min_dip,
            "p_max_dip": args.p_max_dip,
            "p_min_jump": args.p_min_jump,
            "p_max_jump": args.p_max_jump,
            "mag_points": args.mag_points,
            "mag_min_dip": args.mag_min_dip,
            "mag_max_dip": args.mag_max_dip,
            "mag_min_jump": args.mag_min_jump,
            "mag_max_jump": args.mag_max_jump,
            # Baseline parameters
            "baseline_func": args.baseline_func,
            "baseline_s0": args.baseline_s0,
            "baseline_w0": args.baseline_w0,
            "baseline_q": args.baseline_q,
            "baseline_jitter": args.baseline_jitter,
            "baseline_sigma_floor": args.baseline_sigma_floor,
            # Run parameters
            "run_min_points": args.run_min_points,
            "run_max_gap_points": args.run_max_gap_points,
            "run_max_gap_days": args.run_max_gap_days,
            "run_min_duration_days": args.run_min_duration_days,
            "no_event_prob": args.no_event_prob,
            "min_mag_offset": args.min_mag_offset,
            # System parameters
            "workers": args.workers,
            "batch_size": args.batch_size,
            "output_format": args.output_format,
            # Pre-filter (camera median)
            "skip_camera_median": args.skip_camera_median,
            "camera_median_tolerance": args.camera_median_tolerance,
            # Step 5: Post-filter
            "run_post_filter": args.run_post_filter,
            "min_bayes_factor": args.min_bayes_factor,
            "post_filter_min_run_cameras": args.post_filter_min_run_cameras,
            "post_filter_min_run_points": args.post_filter_min_run_points,
            # Step 6: Postprocess
            "run_postprocess": args.run_postprocess,
            "max_plots": args.max_plots,
            "plot_format": args.plot_format,
            # Step 7: Characterization
            "run_characterize": args.run_characterize,
            "run_dust": args.run_dust,
            "gaia_cache": str(args.gaia_cache),
            "index_file": str(args.index_file),
            # Step 8: Classify
            "run_classify": args.run_classify,
            # Step 9: Enrich
            "run_enrich": args.run_enrich,
            "enrich_compute_ls": args.enrich_compute_ls,
            # File paths
            "index_root": str(args.index_root),
            "lc_root": str(args.lc_root),
            "out_dir": str(out_dir),
            "manifest_file": str(manifest_file),
            "filtered_file": str(filtered_file),
            "events_output": str(events_output),
        }

        with open(run_params_file, "w") as f:
            json.dump(run_params, f, indent=2, default=str)

    except Exception as e:
        if args.verbose:
            print(f"Warning: could not write run_params.json: {e}")

    # Write a simple run log with the command and key paths.
    run_log = out_dir / "run.log"
    try:
        events_cmd_preview = shlex.join([sys.executable, "-m", "malca.events", *events_args, "--", "<paths_file>"])
        run_log.write_text(
            "\n".join([
                f"timestamp: {run_start_time.isoformat()}",
                f"command: {cmd}",
                f"events_cmd: {events_cmd_preview}",
                f"out_dir: {out_dir}",
                f"run_params: {run_params_file}",
                f"manifests_dir: {manifests_dir}",
                f"prefilter_dir: {prefilter_dir}",
                f"paths_dir: {paths_dir}",
                f"results_dir: {results_dir}",
                f"results_output: {events_output}",
                f"manifest_file: {manifest_file}",
                f"filtered_file: {filtered_file}",
                f"stats_checkpoint: {stats_checkpoint_file}",
                f"rejected_pre_filter: {prefilter_dir / f'rejected_pre_filter_{mag_bin_tag}.csv'}",
            ]) + "\n"
        )
    except Exception as e:
        if args.verbose:
            print(f"Warning: could not write run log: {e}")

    # Step 1: Build or load manifest
    if args.force_manifest or not manifest_file.exists():
        log(f"Building manifest for mag_bin={args.mag_bin}...")
        df_manifest = build_manifest_dataframe(
            args.index_root,
            args.lc_root,
            mag_bins=args.mag_bin,
            id_column="asas_sn_id",
            show_progress=args.verbose
        )

        # Only keep sources where .dat2 or .csv files exist
        df_manifest = df_manifest[df_manifest["dat_exists"]].reset_index(drop=True)

        log(f"Saving manifest to {manifest_file} ({len(df_manifest)} sources)")
        safe_write_parquet(df_manifest, manifest_file)
    else:
        log(f"Loading existing manifest from {manifest_file}")
        df_manifest = pd.read_parquet(manifest_file)
        log(f"Loaded {len(df_manifest)} sources")

    # Step 2: Apply pre-filters

    if args.force_filter or not filtered_file.exists():
        log(f"\nApplying pre-filters with {args.workers} workers...")

        # Use lc_dir as the directory path for pre_filter compatibility (path/<id>.dat2)
        df_to_filter = df_manifest.rename(columns={"lc_dir": "path"}).copy()

        df_filtered = apply_pre_filters(
            df_to_filter,
            apply_sparse=not args.skip_sparse,
            min_time_span=args.min_time_span,
            min_points_per_day=args.min_points_per_day,
            apply_vsx=not args.skip_vsx,
            vsx_max_sep_arcsec=args.vsx_max_sep,
            vsx_mode=args.vsx_mode,
            vsx_crossmatch_csv=args.vsx_crossmatch,
            apply_multi_camera=not args.skip_multi_camera,
            min_cameras=args.min_cameras,
            n_workers=args.workers,
            show_tqdm=args.verbose,
            rejected_log_csv=str(prefilter_dir / f"rejected_pre_filter_{mag_bin_tag}.csv"),
            stats_checkpoint=str(stats_checkpoint_file),
            stats_chunk_size=args.stats_chunk_size,
        )

        # Exclude rows based on pre-filter results
        if not args.pass_all_prefilters:
            failed_cols = [c for c in df_filtered.columns if c.startswith("failed_") and c != "failed_any"]

            if args.enforce_filters:
                # Only enforce specified filters
                enforce_set = {f"failed_{f.strip()}" for f in args.enforce_filters.split(",")}
                enforce_cols = [c for c in failed_cols if c in enforce_set]
            else:
                enforce_cols = failed_cols

            if enforce_cols:
                exclude_mask = df_filtered[enforce_cols].any(axis=1)
                df_filtered = df_filtered[~exclude_mask].reset_index(drop=True)

        log(f"\nKept {len(df_filtered)}/{len(df_manifest)} sources after pre-filtering")
        log(f"Saving filtered manifest to {filtered_file}")
        safe_write_parquet(df_filtered, filtered_file)
    else:
        log(f"\nLoading existing filtered manifest from {filtered_file}")
        df_filtered = pd.read_parquet(filtered_file)
        log(f"Loaded {len(df_filtered)} filtered sources")

    # Step 2.5: Apply camera median filter to identify cameras to exclude
    if not args.skip_camera_median and "mag_bin" in df_filtered.columns:
        log(f"\nApplying camera median filter (tolerance={args.camera_median_tolerance} mag)...")
        # Ensure 'path' column exists for filter_camera_medians
        if "path" not in df_filtered.columns and "dat_path" in df_filtered.columns:
            df_filtered = df_filtered.rename(columns={"dat_path": "path"})
        df_filtered = filter_camera_medians(
            df_filtered,
            mag_tolerance=args.camera_median_tolerance,
            show_tqdm=args.verbose,
        )
        n_with_exclusions = (df_filtered["excluded_cameras"].fillna("") != "").sum()
        log(f"Found {n_with_exclusions}/{len(df_filtered)} sources with excluded cameras")

    # Step 3: Construct file paths (use full dat_path for events.py input)
    file_col = "dat_path" if "dat_path" in df_filtered.columns else "path"

    # Build metadata CSV with VSX tags and excluded_cameras
    metadata_file = None
    meta_cols = [file_col]
    if not args.skip_vsx and "sep_arcsec" in df_filtered.columns and "class" in df_filtered.columns:
        meta_cols.extend(["sep_arcsec", "class"])
    if "excluded_cameras" in df_filtered.columns:
        meta_cols.append("excluded_cameras")

    if len(meta_cols) > 1:  # More than just file_col
        metadata_dir = prefilter_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = metadata_dir / f"metadata_{mag_bin_tag}.csv"
        meta_df = df_filtered[meta_cols].rename(columns={file_col: "path"})
        meta_df.to_csv(metadata_file, index=False)
        events_args.extend(["--metadata-csv", str(metadata_file)])
        log(f"Wrote metadata CSV with columns: {', '.join(meta_cols[1:])}")

    file_paths = df_filtered[file_col].tolist()

    if not file_paths:
        log("\nNo sources to process after filtering!")
        return

    # Step 4: Call events.py with the filtered paths in batches, with resume support
    log(f"\nPreparing to run events.py on {len(file_paths)} light curves...")

    # Write paths to temp file for events.py to consume
    paths_file = paths_dir / f"filtered_paths_{mag_bin_tag}.txt"
    with open(paths_file, "w") as f:
        for path in file_paths:
            f.write(f"{path}\n")
    if run_log.exists():
        try:
            with run_log.open("a") as f:
                f.write(f"paths_file: {paths_file}\n")
        except Exception as e:
            if args.verbose:
                print(f"Warning: could not update run log with paths_file: {e}")

    # Resume logic: skip paths already recorded in events checkpoint log if present
    base_output = events_output or (results_dir / "lc_events_results.csv")
    suffix_map = {"csv": ".csv", "parquet": ".parquet", "parquet_chunk": None}
    ext = suffix_map.get(events_format)
    if ext and base_output.suffix.lower() != ext:
        base_output = base_output.with_suffix(ext)
    checkpoint_log = base_output.with_name(f"{base_output.stem}_PROCESSED.txt")
    processed_paths: set[str] = set()
    if checkpoint_log.exists() and args.overwrite:
        try:
            with open(checkpoint_log, "w"):
                pass
            log(f"Overwriting checkpoint log: {checkpoint_log}")
        except Exception as e:
            log(f"Warning: could not overwrite checkpoint log {checkpoint_log}: {e}")

    if args.overwrite:
        clear_existing_output(base_output, events_format)

    if checkpoint_log.exists() and not args.overwrite:
        try:
            with open(checkpoint_log, "r") as f:
                processed_paths = {line.strip() for line in f if line.strip()}
            log(f"Checkpoint detected, skipping {len(processed_paths)} already-processed paths")
        except Exception as e:
            log(f"Warning: could not read checkpoint log {checkpoint_log}: {e}")

    remaining = [p for p in file_paths if str(p) not in processed_paths]
    if not remaining:
        log("All paths already processed according to checkpoint. Exiting.")
        return

    # Batch and run
    batch_size = max(1, args.batch_size)
    total_batches = (len(remaining) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(len(remaining), start + batch_size)
        batch_paths = remaining[start:end]

        log(f"\nRunning batch {batch_idx + 1}/{total_batches} ({len(batch_paths)} LCs)...")

        events_cmd = [
            sys.executable, "-m", "malca.events",
            *events_args,
            "--",
            *batch_paths,
        ]

        # Execute
        try:
            result = subprocess.run(events_cmd, check=False)
            if result.returncode != 0:
                print(f"events.py returned non-zero exit ({result.returncode}); stopping.")
                sys.exit(result.returncode)
        except Exception as e:
            print(f"\nError running events.py: {e}")
            print(f"\nFiltered paths saved to: {paths_file}")
            print(f"You can manually run events.py with these paths")
            sys.exit(1)

        # Append processed paths to checkpoint log safely
        checkpoint_log.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_log, "a") as f:
            for p in batch_paths:
                f.write(f"{p}\n")

    log("\nAll batches completed.")

    # Generate run summary with results statistics
    run_end_time = datetime.now()
    run_summary_file = out_dir / "run_summary.json"
    try:
        summary = {
            "run_info": {
                "start_time": run_start_time.isoformat(),
                "end_time": run_end_time.isoformat(),
                "duration_seconds": (run_end_time - run_start_time).total_seconds(),
            },
            "manifest_stats": {
                "total_sources": len(df_manifest),
                "filtered_sources": len(df_filtered),
                "kept_fraction": len(df_filtered) / len(df_manifest) if len(df_manifest) > 0 else 0.0,
            },
        }

        # Pre-filter rejection breakdown
        rejected_log = prefilter_dir / f"rejected_pre_filter_{mag_bin_tag}.csv"
        if rejected_log.exists():
            try:
                df_rejected = pd.read_csv(rejected_log)
                if "reason" in df_rejected.columns:
                    rejection_counts = df_rejected["reason"].value_counts().to_dict()
                    summary["pre_filter_rejections"] = {
                        "total_rejected": len(df_rejected),
                        "by_reason": rejection_counts,
                    }
            except Exception as e:
                if args.verbose:
                    print(f"Warning: could not parse rejection log: {e}")

        # Detection results statistics
        results_files = []
        if events_format == "csv":
            if base_output.exists():
                results_files = [base_output]
        elif events_format == "parquet":
            if base_output.exists():
                results_files = [base_output]
        elif events_format == "parquet_chunk":
            chunk_dir = base_output.parent if base_output.suffix else base_output
            results_files = sorted(chunk_dir.glob("chunk_*.parquet"))

        if results_files:
            try:
                if events_format == "csv":
                    df_results = pd.read_csv(results_files[0])
                else:  # parquet or parquet_chunk
                    df_results = pd.concat([pd.read_parquet(f) for f in results_files], ignore_index=True)

                detection_stats = {
                    "total_detections": len(df_results),
                    "unique_sources": df_results["path"].nunique() if "path" in df_results.columns else None,
                }

                # Count significant detections
                if "dip_significant" in df_results.columns:
                    detection_stats["dip_significant"] = int(df_results["dip_significant"].sum())
                if "jump_significant" in df_results.columns:
                    detection_stats["jump_significant"] = int(df_results["jump_significant"].sum())

                # Event type counts
                if "event_type" in df_results.columns:
                    detection_stats["by_event_type"] = df_results["event_type"].value_counts().to_dict()

                summary["detection_stats"] = detection_stats

            except Exception as e:
                if args.verbose:
                    print(f"Warning: could not parse detection results: {e}")

        # Write summary (will be updated again if post-filter/postprocess run)
        with open(run_summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        log(f"\nRun summary saved to {run_summary_file}")

    except Exception as e:
        if args.verbose:
            print(f"Warning: could not write run summary: {e}")

    # Step 5: Apply post-filters (optional)
    if args.run_post_filter and results_files:
        log("\n=== Step 5: Applying post-filters ===")
        try:
            # Load events results
            if events_format == "csv":
                df_events = pd.read_csv(results_files[0])
            else:
                df_events = pd.concat([pd.read_parquet(f) for f in results_files], ignore_index=True)

            # Apply post-filters
            df_post_filtered = apply_post_filters(
                df_events,
                apply_posterior_strength=True,
                min_bayes_factor=args.min_bayes_factor,
                apply_run_robustness=True,
                min_run_cameras=args.post_filter_min_run_cameras,
                min_run_points=args.post_filter_min_run_points,
                show_tqdm=args.verbose,
                verbose=args.verbose,
            )

            # Save filtered results
            post_filter_output = results_dir / "lc_events_filtered.csv"
            df_post_filtered.to_csv(post_filter_output, index=False)
            log(f"Post-filtered results saved to {post_filter_output}")

            # Update summary with post-filter stats
            n_passed = int((~df_post_filtered["failed_any"]).sum()) if "failed_any" in df_post_filtered.columns else len(df_post_filtered)
            n_failed = int(df_post_filtered["failed_any"].sum()) if "failed_any" in df_post_filtered.columns else 0
            summary["post_filter_stats"] = {
                "total_input": len(df_events),
                "passed": n_passed,
                "failed": n_failed,
                "pass_rate": n_passed / len(df_events) if len(df_events) > 0 else 0.0,
            }

            # Overwrite summary with updated stats
            with open(run_summary_file, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            log(f"Post-filter: {n_passed}/{len(df_events)} passed")

        except Exception as e:
            print(f"Error in post-filter step: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Step 6: Generate postprocess plots (optional)
    if args.run_postprocess:
        if not args.run_post_filter:
            print("Warning: --run-postprocess requires --run-post-filter. Skipping postprocess.")
        else:
            log("\n=== Step 6: Generating postprocess plots ===")
            try:
                post_filter_output = results_dir / "lc_events_filtered.csv"
                if post_filter_output.exists():
                    df_post_filtered = pd.read_csv(post_filter_output)
                    
                    postprocess_dir = out_dir / "plots"
                    postprocess_dir.mkdir(parents=True, exist_ok=True)

                    postprocess_summary = run_postprocess(
                        df_post_filtered,
                        out_dir=postprocess_dir,
                        baseline=args.baseline_func,
                        logbf_threshold_dip=args.logbf_threshold_dip,
                        logbf_threshold_jump=args.logbf_threshold_jump,
                        plot_format=args.plot_format,
                        max_plots=args.max_plots,
                        show_tqdm=args.verbose,
                    )

                    # Update summary with postprocess stats
                    summary["postprocess_stats"] = postprocess_summary

                    # Overwrite summary with updated stats
                    with open(run_summary_file, "w") as f:
                        json.dump(summary, f, indent=2, default=str)

                    log(f"Postprocess: {postprocess_summary.get('plotted', 0)} plots generated in {postprocess_dir}")
                else:
                    print(f"Warning: post-filter output not found at {post_filter_output}")

            except Exception as e:
                print(f"Error in postprocess step: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

    # Step 7: Characterization + dust (optional)
    if args.run_characterize or args.run_dust:
        if not args.run_post_filter:
            print("Warning: --run-characterize/--run-dust requires --run-post-filter. Skipping characterization.")
        else:
            log("\n=== Step 7: Characterizing candidates ===")
            try:
                post_filter_output = results_dir / "lc_events_filtered.csv"
                if not post_filter_output.exists():
                    print(f"Warning: post-filter output not found at {post_filter_output}")
                else:
                    df_char = pd.read_csv(post_filter_output)

                    if "failed_any" in df_char.columns:
                        df_char = df_char[~df_char["failed_any"]].copy()

                    if "path" in df_char.columns and "asas_sn_id" not in df_char.columns:
                        def _extract_id(path_str: str) -> str:
                            name = Path(path_str).name
                            return Path(name).stem.split("-")[0]

                        df_char["asas_sn_id"] = df_char["path"].astype(str).map(_extract_id)

                    index_path = args.index_file.expanduser() if args.index_file else None
                    if index_path and index_path.exists():
                        try:
                            index_cols = ["asas_sn_id", "gaia_id", "ra_deg", "dec_deg"]
                            if str(index_path).endswith(".parquet"):
                                index_df = pd.read_parquet(index_path, columns=index_cols)
                            else:
                                index_df = pd.read_csv(index_path, usecols=index_cols)
                            index_df["asas_sn_id"] = index_df["asas_sn_id"].astype(str)
                            if "asas_sn_id" in df_char.columns:
                                df_char["asas_sn_id"] = df_char["asas_sn_id"].astype(str)
                                df_char = df_char.merge(index_df, on="asas_sn_id", how="left", suffixes=("", "_idx"))
                        except Exception as e:
                            print(f"Warning: could not load index file {index_path}: {e}")
                    elif args.run_characterize or args.run_dust:
                        print(f"Warning: index file not found: {index_path}")

                    if args.run_characterize:
                        gaia_ids = []
                        if "gaia_id" in df_char.columns:
                            gaia_ids = df_char["gaia_id"].dropna().astype(str).unique().tolist()
                        if not gaia_ids:
                            print("Warning: no Gaia IDs found for characterization. Provide --index-file with gaia_id.")
                        else:
                            gaia_df = query_gaia_by_ids(
                                gaia_ids,
                                cache_file=str(args.gaia_cache) if args.gaia_cache else None,
                            )
                            if not gaia_df.empty:
                                df_char["gaia_id"] = df_char["gaia_id"].astype(str)
                                gaia_df["source_id"] = gaia_df["source_id"].astype(str)
                                df_char = df_char.merge(
                                    gaia_df,
                                    left_on="gaia_id",
                                    right_on="source_id",
                                    how="left",
                                    suffixes=("", "_gaia"),
                                )

                    if args.run_dust:
                        df_char = get_dust_extinction(df_char)

                    characterize_output = results_dir / "lc_events_characterized.csv"
                    df_char.to_csv(characterize_output, index=False)
                    log(f"Characterization results saved to {characterize_output}")

            except Exception as e:
                print(f"Error in characterization step: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

    # Step 8: Run classification (optional)
    if args.run_classify:
        if not args.run_post_filter:
            print("Warning: --run-classify requires --run-post-filter. Skipping classification.")
        else:
            log("\n=== Step 8: Running classification ===")
            try:
                characterize_output = results_dir / "lc_events_characterized.csv"
                post_filter_output = results_dir / "lc_events_filtered.csv"

                if characterize_output.exists():
                    df_post_filtered = pd.read_csv(characterize_output)
                elif post_filter_output.exists():
                    df_post_filtered = pd.read_csv(post_filter_output)
                else:
                    df_post_filtered = None
                    print(f"Warning: post-filter output not found at {post_filter_output}")

                if df_post_filtered is not None:
                    # Run classification on passing candidates
                    df_passed = df_post_filtered[~df_post_filtered["failed_any"]].copy() if "failed_any" in df_post_filtered.columns else df_post_filtered.copy()
                    
                    if len(df_passed) > 0:
                        df_classified = compute_all_classifications(df_passed)
                        
                        # Save classified results
                        classify_output = results_dir / "lc_events_classified.csv"
                        df_classified.to_csv(classify_output, index=False)
                        log(f"Classification results saved to {classify_output}")
                        
                        # Update summary with classification stats
                        class_counts = df_classified["final_class"].value_counts().to_dict() if "final_class" in df_classified.columns else {}
                        summary["classification_stats"] = {
                            "total_classified": len(df_classified),
                            "by_class": class_counts,
                        }
                        
                        # Overwrite summary with updated stats
                        with open(run_summary_file, "w") as f:
                            json.dump(summary, f, indent=2, default=str)
                        
                        log(f"Classification: {len(df_classified)} candidates classified")
                    else:
                        log("No passing candidates to classify.")

            except Exception as e:
                print(f"Error in classification step: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

    # Step 9: Enrich with compute_stats (optional)
    if args.run_enrich:
        if not args.run_post_filter:
            print("Warning: --run-enrich requires --run-post-filter. Skipping enrichment.")
        else:
            log("\n=== Step 9: Enriching with light curve stats ===")
            try:
                # Load final results (classified if available, otherwise post-filtered)
                classify_output = results_dir / "lc_events_classified.csv"
                characterize_output = results_dir / "lc_events_characterized.csv"
                post_filter_output = results_dir / "lc_events_filtered.csv"
                
                if classify_output.exists():
                    df_to_enrich = pd.read_csv(classify_output)
                elif characterize_output.exists():
                    df_to_enrich = pd.read_csv(characterize_output)
                elif post_filter_output.exists():
                    df_to_enrich = pd.read_csv(post_filter_output)
                else:
                    print(f"Warning: No post-filter or classified output found")
                    df_to_enrich = None
                
                if df_to_enrich is not None:
                    # Filter to passing candidates only
                    if "failed_any" in df_to_enrich.columns:
                        df_passed = df_to_enrich[~df_to_enrich["failed_any"]].copy()
                    else:
                        df_passed = df_to_enrich.copy()
                    
                    if len(df_passed) > 0:
                        log(f"Enriching {len(df_passed)} candidates with compute_stats...")
                        
                        enriched_rows = []
                        from tqdm.auto import tqdm
                        
                        for idx, row in tqdm(df_passed.iterrows(), total=len(df_passed), 
                                            desc="compute_stats", disable=not args.verbose):
                            lc_path = Path(row["path"])
                            if not lc_path.exists():
                                enriched_rows.append(row.to_dict())
                                continue
                            
                            try:
                                # Extract asassn_id from path
                                asassn_id = lc_path.stem.split("-")[0]
                                dir_path = str(lc_path.parent)
                                
                                # Run compute_stats
                                _, stats_dict = compute_stats(
                                    asassn_id, 
                                    dir_path,
                                    use_only_good=True,
                                    compute_ls=args.enrich_compute_ls,
                                )
                                
                                # Merge stats into row
                                merged = row.to_dict()
                                for k, v in stats_dict.items():
                                    if k not in merged:  # Don't overwrite existing columns
                                        merged[f"stats_{k}"] = v
                                enriched_rows.append(merged)
                                
                            except Exception as e:
                                if args.verbose:
                                    print(f"Warning: compute_stats failed for {lc_path}: {e}")
                                enriched_rows.append(row.to_dict())
                        
                        df_enriched = pd.DataFrame(enriched_rows)
                        
                        # Save enriched results
                        enrich_output = results_dir / "lc_events_enriched.csv"
                        df_enriched.to_csv(enrich_output, index=False)
                        log(f"Enriched results saved to {enrich_output}")
                        
                        # Update summary
                        n_stats_cols = len([c for c in df_enriched.columns if c.startswith("stats_")])
                        summary["enrichment_stats"] = {
                            "total_enriched": len(df_enriched),
                            "stats_columns_added": n_stats_cols,
                        }
                        
                        with open(run_summary_file, "w") as f:
                            json.dump(summary, f, indent=2, default=str)
                        
                        log(f"Enrichment: {len(df_enriched)} candidates, {n_stats_cols} stats columns added")
                    else:
                        log("No passing candidates to enrich.")

            except Exception as e:
                print(f"Error in enrichment step: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()


if __name__ == "__main__":
    main()
