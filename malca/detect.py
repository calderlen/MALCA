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
    malca detect --mag-bin 13_13.5 [options...]
    malca detect --mag-bin 13_13.5 --run-post-filter --run-classify --run-enrich
"""
from __future__ import annotations

import os
# Set threading environment variables before importing numpy/pandas/numba
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import argparse
import shutil
from datetime import datetime
import json
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any
import pandas as pd
import tempfile
from tqdm.auto import tqdm

from malca.manifest import build_manifest_dataframe
from malca.pre_filter import apply_pre_filters, filter_camera_medians
from malca.post_filter import apply_post_filters
from malca.plot import plot_passing_candidates
from malca.classify import compute_all_classifications
from malca.stats import compute_stats
from malca.characterize import characterize_candidates_df
from malca.gaia_fetch import _extract_gaia_ids, fetch_gaia_catalog
from malca.enrich.neighbor import run_neighbor_enrichment
from malca.enrich.spectra import run_spectra_availability
from malca.config.config_io import PARQUET_OUTPUT_COMPRESSION, PARQUET_CACHE_COMPRESSION
from malca.config.config_paths import ASASSN_INDEX_PATH, LCV2_ROOT, VSX_CROSSMATCH_PATH, GAIA_LOCAL_CATALOG
from malca.config.config_pipeline import (
    WORKERS, BATCH_SIZE, TRIGGER_MODE, P_POINTS, MAG_POINTS,
    LOGBF_THRESHOLD_DIP, LOGBF_THRESHOLD_JUMP, SIGNIFICANCE_THRESHOLD,
    MIN_MAG_OFFSET, RUN_MIN_POINTS, RUN_MAX_GAP_POINTS,
    BASELINE_FUNC, BASELINE_S0, BASELINE_W0, BASELINE_Q, BASELINE_JITTER,
)
from malca.config.config_io import OUTPUT_FORMAT, EVENTS_OUTPUT_CHUNK_SIZE
from malca.config.config_filters import (
    MIN_TIME_SPAN, MIN_POINTS_PER_DAY, MIN_CAMERAS,
    VSX_MAX_SEP_ARCSEC, VSX_MODE, CAMERA_MEDIAN_TOLERANCE, STATS_CHUNK_SIZE,
    MIN_BAYES_FACTOR, POST_FILTER_MIN_RUN_CAMERAS, POST_FILTER_MIN_RUN_POINTS,
)
from malca.config.config_characterize import (
    GAIA_CHUNK_SIZE, NEIGHBOR_RADIUS_ARCSEC, NEIGHBOR_CHUNK_SIZE,
    SPECTRA_RADIUS_ARCSEC, SPECTRA_CHUNK_SIZE,
)
from malca.utils import log as _log


def safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet atomically to avoid corruption on interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def load_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        safe_write_parquet(df, path)
    else:
        df.to_csv(path, index=False)


def default_run_dir(base_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_root / "runs" / timestamp


def find_latest_run_dir(base_root: Path, mag_bin: list[str]) -> Path | None:
    """Find the most recent run directory matching the given mag bin(s)."""
    runs_dir = base_root / "runs"
    if not runs_dir.is_dir():
        return None
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        params_file = d / "run_params.json"
        if not params_file.exists():
            continue
        with open(params_file) as f:
            params = json.load(f)
        if params.get("mag_bin") == mag_bin:
            return d  # sorted reverse by timestamp, first match is latest
    return None


def _normalize_mag_bins(raw_value: Any) -> list[str] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        value = raw_value.strip()
        return [value] if value else None
    if isinstance(raw_value, (list, tuple)):
        values = [str(v).strip() for v in raw_value if str(v).strip()]
        return values or None
    return None


def _read_mag_bins_from_params_file(params_file: Path) -> list[str] | None:
    if not params_file.exists():
        return None
    try:
        with params_file.open() as f:
            params = json.load(f)
    except Exception:
        return None
    return _normalize_mag_bins(params.get("mag_bin"))


def _read_mag_bins_from_bundle(bundle_zip: Path) -> list[str] | None:
    bundle_zip = Path(bundle_zip).expanduser()
    if not bundle_zip.exists() or (not zipfile.is_zipfile(bundle_zip)):
        return None
    try:
        with zipfile.ZipFile(bundle_zip, "r") as zf:
            with zf.open("run_params.json") as f:
                params = json.load(f)
    except Exception:
        return None
    return _normalize_mag_bins(params.get("mag_bin"))


def _assert_mag_bin_match(expected: list[str], observed: list[str], source: str) -> None:
    if expected != observed:
        raise SystemExit(
            f"Provided --mag-bin ({observed}) does not match {source} ({expected})."
        )


def get_out_dir_from_bundle(bundle_path: Path, base_root: Path) -> Path:
    """Extract run directory name from bundle filename."""
    bundle_name = bundle_path.stem  # e.g., "20260209_162336_bundle" -> "20260209_162336_bundle"
    base_name = bundle_name.removesuffix("_bundle")  # Always strip _bundle

    runs_dir = base_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    candidate = runs_dir / base_name
    if not candidate.exists():
        return candidate

    # If exists, append _home
    return runs_dir / f"{base_name}_home"


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


def import_bundle_zip(bundle_zip: Path, out_dir: Path) -> None:
    """Extract a pipeline transfer bundle into out_dir."""
    bundle_zip = Path(bundle_zip).expanduser()
    if not bundle_zip.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_zip}")
    if not zipfile.is_zipfile(bundle_zip):
        raise ValueError(f"Bundle is not a valid zip file: {bundle_zip}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_zip, "r") as zf:
        zf.extractall(out_dir)


def _collect_bundle_lightcurve_files(out_dir: Path) -> list[tuple[Path, str]]:
    """Collect candidate .dat2/.raw2 files to include in bundle assets.

    Source files are read directly from their original location and are never
    modified in place.
    """
    filtered_candidates = out_dir / "results" / "lc_events_filtered.parquet"
    if not filtered_candidates.exists():
        return []

    try:
        df_candidates = pd.read_parquet(filtered_candidates)
    except Exception as exc:
        print(f"Warning: could not read {filtered_candidates} for light curve bundling: {exc}")
        return []

    if "path" not in df_candidates.columns:
        return []

    files_to_bundle: list[tuple[Path, str]] = []
    seen_files: set[Path] = set()

    for raw_path in df_candidates["path"].dropna().astype(str).unique().tolist():
        dat_path = Path(raw_path).expanduser()
        if dat_path.suffix.lower() != ".dat2":
            continue

        for source_file in (dat_path, dat_path.with_suffix(".raw2")):
            if not source_file.exists() or (not source_file.is_file()):
                continue

            resolved = source_file.resolve()
            if resolved in seen_files:
                continue

            seen_files.add(resolved)
            arcname = f"bundle_assets/lightcurves/{resolved.name}"
            files_to_bundle.append((resolved, arcname))

    return files_to_bundle


def export_bundle_zip(bundle_zip: Path, out_dir: Path, include_all: bool = False) -> list[str]:
    """Create transfer bundle zip from a pipeline out_dir."""
    bundle_zip = Path(bundle_zip).expanduser()
    bundle_zip.parent.mkdir(parents=True, exist_ok=True)

    include_rel_paths = [
        "run_params.json",
        "run_summary.json",
        "run.log",
        "results/lc_events_filtered.parquet",
        "results/lc_events_enriched.parquet",
        "results/lc_events_characterized.parquet",
        "results/lc_events_classified.parquet",
        "results/lc_events_neighbors.parquet",
        "results/lc_events_spectra.parquet",
    ]
    if include_all:
        include_rel_paths.append("bundle_assets/asassn_index_full.parquet")
    include_globs = [
        "results/lc_events_results.*",
    ]
    include_dirs = [
        "results",
        "plots",
    ]
    if include_all:
        include_dirs.extend(["manifests", "prefilter", "paths", "gaia_cache"])

    files_to_add: set[Path] = set()
    for rel in include_rel_paths:
        p = out_dir / rel
        if p.exists() and p.is_file():
            files_to_add.add(p)

    for pattern in include_globs:
        for p in out_dir.glob(pattern):
            if p.exists() and p.is_file():
                files_to_add.add(p)

    for rel_dir in include_dirs:
        d = out_dir / rel_dir
        if d.exists() and d.is_dir():
            for p in d.rglob("*"):
                if p.is_file():
                    files_to_add.add(p)

    # Prevent accidental self-inclusion if bundle path is inside out_dir.
    files_to_add.discard(bundle_zip)

    lightcurve_files = _collect_bundle_lightcurve_files(out_dir)

    if not files_to_add and not lightcurve_files:
        raise FileNotFoundError(f"No bundle files found under {out_dir}")

    ordered_files = sorted(files_to_add, key=lambda p: str(p.relative_to(out_dir)))
    ordered_lightcurve_files = sorted(lightcurve_files, key=lambda item: item[1])

    total_files = len(ordered_files) + len(ordered_lightcurve_files)
    print(f"Bundling {total_files} files with ZIP_DEFLATED compression...")

    bundled_paths: list[str] = []
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in ordered_files:
            arcname = str(p.relative_to(out_dir))
            zf.write(p, arcname=arcname)
            bundled_paths.append(arcname)
        for source_file, arcname in ordered_lightcurve_files:
            zf.write(source_file, arcname=arcname)
            bundled_paths.append(arcname)

    return bundled_paths


def _build_post_filter_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build apply_post_filters kwargs from detect CLI arguments."""
    return {
        # Core filters
        "apply_evidence_strength": not args.skip_evidence_strength,
        "min_bayes_factor": args.min_bayes_factor,
        "require_finite_local_bf": not args.allow_infinite_local_bf,
        "apply_run_robustness": not args.skip_run_robustness,
        "min_run_count": args.min_run_count,
        "min_run_points": args.post_filter_min_run_points,
        "min_run_cameras": args.post_filter_min_run_cameras,
        # Optional filters
        "apply_morphology": args.apply_morphology,
        "dip_morphology": args.dip_morphology,
        "jump_morphology": args.jump_morphology,
        "min_delta_bic": args.min_delta_bic,
        "apply_score": not args.skip_score_filter,
        "min_score": args.min_score,
        # Validation filters
        "apply_periodicity_validation": args.apply_periodicity_validation,
        "periodicity_n_bootstrap": args.periodicity_n_bootstrap,
        "periodicity_significance": args.periodicity_significance,
        "periodicity_exclude_aliases": not args.periodicity_no_exclude_aliases,
        "periodicity_flag_only": not args.periodicity_reject,
        "periodicity_workers": args.periodicity_workers,
        "periodicity_checkpoint_dir": args.periodicity_checkpoint_dir,
        "phase_plot_max_sig": args.phase_plot_max_sig,
        "phase_plot_min_power": args.phase_plot_min_power,
        "phase_plot_allow_alias": args.phase_plot_allow_alias,
        "apply_gaia_ruwe_validation": not args.skip_gaia_ruwe_validation,
        "gaia_max_ruwe": args.gaia_max_ruwe,
        "gaia_flag_only": not args.gaia_reject,
        "apply_periodic_catalog_validation": not args.skip_periodic_catalog_validation,
        "periodic_catalog_max_sep": args.periodic_catalog_max_sep,
        "periodic_catalog_flag_only": not args.periodic_catalog_reject,
        # Progress/logging
        "show_tqdm": args.verbose,
        "verbose": args.verbose,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run events.py on pre-filtered light curves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="All other arguments are passed directly to events.py"
    )

    # Manifest/pre-filter args
    parser.add_argument("--mag-bin", nargs="+", help="Magnitude bin(s) to process")
    parser.add_argument("--index-root", type=Path, default=LCV2_ROOT,
                        help="Index root directory (contains mag_bin/index*.csv)")
    parser.add_argument("--lc-root", type=Path, default=LCV2_ROOT,
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
    parser.add_argument("--min-time-span", type=float, default=MIN_TIME_SPAN, help="Min time span (days)")
    parser.add_argument("--min-points-per-day", type=float, default=MIN_POINTS_PER_DAY, help="Min cadence")
    parser.add_argument("--min-cameras", type=int, default=MIN_CAMERAS, help="Min cameras required")
    parser.add_argument("--skip-sparse", action="store_true", help="Skip sparse LC filter")
    parser.add_argument("--skip-multi-camera", action="store_true", help="Skip multi-camera filter")
    parser.add_argument("--skip-vsx", action="store_true", help="Skip VSX crossmatch/tagging")
    parser.add_argument("--skip-camera-median", action="store_true", help="Skip camera median filter (identifies cameras to exclude from .raw2 files)")
    parser.add_argument("--camera-median-tolerance", type=float, default=CAMERA_MEDIAN_TOLERANCE, help="Tolerance beyond mag bin for camera median filter (default: 0.2 mag)")
    parser.add_argument("--vsx-max-sep", type=float, default=VSX_MAX_SEP_ARCSEC, help="Max separation for VSX match (arcsec)")
    parser.add_argument("--vsx-mode", type=str, default=VSX_MODE, choices=["tag", "filter"], help="VSX handling: tag adds vsx_sep_arcsec/vsx_class columns, filter removes matches (default: tag)")
    parser.add_argument("--vsx-crossmatch", type=Path, default=VSX_CROSSMATCH_PATH, help="Path to pre-crossmatched VSX CSV (with asas_sn_id, vsx_sep_arcsec, vsx_class)")
    parser.add_argument("--pass-all-prefilters", action="store_true", help="Pass all light curves to events.py regardless of pre-filter results (tags are still added)")
    parser.add_argument("--enforce-filters", type=str, default=None, help="Comma-separated list of pre-filters to enforce (e.g., 'sparse,multi_camera'). " "Only rows failing these filters are excluded. Default: enforce all enabled filters.")
    parser.add_argument("--workers", type=int, default=WORKERS, help="Workers for parallel processing")
    parser.add_argument("--stats-chunk-size", type=int, default=STATS_CHUNK_SIZE, help="Rows per checkpoint save during stats computation")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Max light curves per events.py call")

    # events.py args
    parser.add_argument("--trigger-mode", type=str, default=TRIGGER_MODE, choices=["logbf", "posterior_prob"], help="Triggering mode")
    parser.add_argument("--logbf-threshold-dip", type=float, default=LOGBF_THRESHOLD_DIP, help="Per-point dip trigger threshold")
    parser.add_argument("--logbf-threshold-jump", type=float, default=LOGBF_THRESHOLD_JUMP, help="Per-point jump trigger threshold")
    parser.add_argument("--significance-threshold", type=float, default=SIGNIFICANCE_THRESHOLD, help="Posterior probability threshold (if trigger-mode=posterior_prob)")
    parser.add_argument("--p-points", type=int, default=P_POINTS, help="Number of points in the p grid")
    parser.add_argument("--mag-points", type=int, default=MAG_POINTS, help="Number of points in the magnitude grid")
    parser.add_argument("--run-min-points", type=int, default=RUN_MIN_POINTS, help="Min triggered points in a run")
    parser.add_argument("--run-max-gap-points", type=int, default=RUN_MAX_GAP_POINTS, help="Allow up to this many missing indices inside a run")
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
        default=BASELINE_FUNC,
        choices=["gp", "gp_masked", "global_median", "per_camera_median"],
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
    parser.add_argument("--min-mag-offset", type=float, default=MIN_MAG_OFFSET, help="Require |event_mag - baseline_mag| > threshold")
    parser.add_argument("--output", type=str, default=None, help="Output path for results (default: <out_dir>/lc_events_results.parquet)")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory for all outputs (default: output/runs/<timestamp>)")
    parser.add_argument("--output-format", type=str, default=OUTPUT_FORMAT, choices=["csv", "parquet", "parquet_chunk"], help="Output format")
    parser.add_argument("--chunk-size", type=int, default=EVENTS_OUTPUT_CHUNK_SIZE, help="Write results in chunks of this many rows")
    parser.add_argument(
        "--stage",
        type=str,
        default="full",
        choices=["full", "cluster", "home"],
        help="Pipeline stage: full=all steps, cluster=raw-dependent upstream, home=downstream only",
    )
    parser.add_argument("--import-bundle", type=Path, default=None, help="Zip bundle produced by --export-bundle (for home stage)")
    parser.add_argument("--export-bundle", type=Path, default=None, help="Write transferable zip bundle at end of run")
    parser.add_argument("--no-export-bundle", dest="export_bundle_enabled", action="store_false",
                        help="Skip export bundle creation at end of run")
    parser.add_argument("--full-bundle", action="store_true", default=False, help="Include all large assets in export bundle (index, gaia cache, manifests, prefilter, paths)")

    # Step 5: Post-filter args (enabled by default)
    parser.add_argument("--run-post-filter", dest="run_post_filter", action="store_true", help="Run post_filter after events.py completes (default: enabled)")
    parser.add_argument("--no-run-post-filter", dest="run_post_filter", action="store_false", help="Skip post-filter step")
    parser.add_argument("--skip-evidence-strength", action="store_true", help="Skip evidence-strength filter")
    parser.add_argument("--min-bayes-factor", type=float, default=MIN_BAYES_FACTOR, help="Min Bayes factor for post-filter (default: 10.0)")
    parser.add_argument("--allow-infinite-local-bf", action="store_true", help="Allow infinite local Bayes factors (default: require finite)")
    parser.add_argument("--skip-run-robustness", action="store_true", help="Skip run-robustness filter")
    parser.add_argument("--min-run-count", type=int, default=1, help="Minimum run count for run-robustness filter (default: 1)")
    parser.add_argument("--post-filter-min-run-cameras", type=int, default=POST_FILTER_MIN_RUN_CAMERAS, help="Min cameras for run robustness filter (default: 2)")
    parser.add_argument("--post-filter-min-run-points", type=int, default=POST_FILTER_MIN_RUN_POINTS, help="Min points per run for robustness filter (default: 2)")
    parser.add_argument("--apply-morphology", action="store_true", help="Apply morphology filter in post-filter stage")
    parser.add_argument("--dip-morphology", type=str, default="gaussian", choices=["gaussian", "paczynski"], help="Required morphology for dip events (default: gaussian)")
    parser.add_argument("--jump-morphology", type=str, default="paczynski", choices=["gaussian", "paczynski"], help="Required morphology for jump events (default: paczynski)")
    parser.add_argument("--min-delta-bic", type=float, default=10.0, help="Minimum delta BIC for morphology filter (default: 10.0)")
    parser.add_argument("--skip-score-filter", action="store_true", help="Skip score filter (enabled by default)")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum score threshold for score filter (default: 0.0)")
    parser.add_argument("--apply-periodicity-validation", action="store_true", help="Enable bootstrap periodicity validation")
    parser.add_argument("--periodicity-n-bootstrap", type=int, default=1000, help="Bootstrap iterations for periodicity validation (default: 1000)")
    parser.add_argument("--periodicity-significance", type=float, default=0.01, help="Significance threshold for periodicity validation (default: 0.01)")
    parser.add_argument("--periodicity-no-exclude-aliases", action="store_true", help="Do not exclude alias periods during periodicity validation")
    parser.add_argument("--periodicity-reject", action="store_true", help="Reject periodicity matches instead of flagging only")
    parser.add_argument("--periodicity-workers", type=int, default=WORKERS, help="Workers for periodicity validation (default: WORKERS)")
    parser.add_argument("--periodicity-checkpoint-dir", type=Path, default=None, help="Checkpoint directory for periodicity validation")
    parser.add_argument("--phase-plot-max-sig", type=float, default=0.01, help="Require lsp_bootstrap_sig <= this for phase plotting (default: 0.01)")
    parser.add_argument("--phase-plot-min-power", type=float, default=0.3, help="Require lsp_power >= this for phase plotting (default: 0.3)")
    parser.add_argument("--phase-plot-allow-alias", action="store_true", help="Allow alias periods for phase plotting")
    parser.add_argument("--skip-gaia-ruwe-validation", action="store_true", help="Skip Gaia RUWE validation")
    parser.add_argument("--gaia-max-ruwe", type=float, default=1.4, help="Maximum RUWE threshold (default: 1.4)")
    parser.add_argument("--gaia-reject", action="store_true", help="Reject high-RUWE sources instead of flagging only")
    parser.add_argument("--skip-periodic-catalog-validation", action="store_true", help="Skip periodic-catalog crossmatch validation")
    parser.add_argument("--periodic-catalog-max-sep", type=float, default=3.0, help="Maximum separation for periodic-catalog matching in arcsec (default: 3.0)")
    parser.add_argument("--periodic-catalog-reject", action="store_true", help="Reject periodic-catalog matches instead of flagging only")

    # Step 7: Postprocess args (enabled by default)
    parser.add_argument("--run-postprocess", dest="run_postprocess", action="store_true", help="Run postprocess (generate plots) after post_filter (default: enabled)")
    parser.add_argument("--no-run-postprocess", dest="run_postprocess", action="store_false", help="Skip postprocess step")
    parser.add_argument("--max-plots", type=int, default=None, help="Limit number of plots generated (default: no limit)")
    parser.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf"], help="Output format for plots (default: png)")

    # Step 8: Characterization args (enabled by default)
    parser.add_argument("--run-characterize", dest="run_characterize", action="store_true", help="Run Gaia DR3 characterization after post_filter (default: enabled)")
    parser.add_argument("--no-run-characterize", dest="run_characterize", action="store_false", help="Skip characterization step")
    parser.add_argument("--gaia-cache", type=Path, default=None, help="Path to Gaia query cache file (parquet). Default: <out_dir>/gaia_cache/gaia_cache.parquet")
    parser.add_argument("--characterize-crossmatch", type=Path, default=VSX_CROSSMATCH_PATH, help="ASAS-SN x VSX crossmatch file for characterize step")
    parser.add_argument("--characterize-chunk-size", type=int, default=GAIA_CHUNK_SIZE, help="Gaia query chunk size for characterize step")
    parser.add_argument("--characterize-starhorse", type=str, default="tap", help="StarHorse mode/path for characterize step (default: tap)")
    parser.add_argument("--no-characterize-banyan", dest="characterize_banyan", action="store_false", help="Disable BANYAN Sigma enrichment in characterize step")
    parser.add_argument("--no-characterize-iphas", dest="characterize_iphas", action="store_false", help="Disable IPHAS enrichment in characterize step")
    parser.add_argument("--no-characterize-sfr", dest="characterize_sfr", action="store_false", help="Disable star-forming-region enrichment in characterize step")
    parser.add_argument("--no-characterize-clusters", dest="characterize_clusters", action="store_false", help="Disable open-cluster enrichment in characterize step")
    parser.add_argument("--no-characterize-unwise", dest="characterize_unwise", action="store_false", help="Disable unWISE enrichment in characterize step")
    parser.add_argument("--run-dust", dest="run_dust", action="store_true", help="Run 3D dust extinction correction (default: enabled)")
    parser.add_argument("--no-run-dust", dest="run_dust", action="store_false", help="Skip dust extinction step")

    # Step 9: Classify args (enabled by default)
    parser.add_argument("--run-classify", dest="run_classify", action="store_true", help="Run classification (EB/CV/starspot rejection, YSO) (default: enabled)")
    parser.add_argument("--no-run-classify", dest="run_classify", action="store_false", help="Skip classification step")

    # Step 6: Enrich args (enabled by default)
    parser.add_argument("--run-enrich", dest="run_enrich", action="store_true", help="Enrich passing candidates with comprehensive light curve stats (default: enabled)")
    parser.add_argument("--no-run-enrich", dest="run_enrich", action="store_false", help="Skip enrichment step")
    parser.add_argument("--enrich-compute-ls", action="store_true", help="Include Lomb-Scargle periodogram in enrichment (expensive)")

    # Step 10: Neighbor enrichment args (enabled by default)
    parser.add_argument("--run-neighbor-enrich", dest="run_neighbor_enrich", action="store_true", help="Bulk neighbor enrichment for passing candidates (default: enabled)")
    parser.add_argument("--no-run-neighbor-enrich", dest="run_neighbor_enrich", action="store_false", help="Skip neighbor enrichment step")
    parser.add_argument("--neighbor-radius-arcsec", type=float, default=NEIGHBOR_RADIUS_ARCSEC, help="Neighbor search radius in arcsec (default: 15)")
    parser.add_argument("--neighbor-chunk-size", type=int, default=NEIGHBOR_CHUNK_SIZE, help="Bulk chunk size for neighbor lookups")
    parser.add_argument("--neighbor-cache", type=Path, default=None, help="Optional cache parquet path for neighbor lookups")

    # Step 11: Spectra availability args (enabled by default)
    parser.add_argument("--run-spectra-enrich", dest="run_spectra_enrich", action="store_true", help="Bulk spectra-availability enrichment for passing candidates (default: enabled)")
    parser.add_argument("--no-run-spectra-enrich", dest="run_spectra_enrich", action="store_false", help="Skip spectra enrichment step")
    parser.add_argument("--spectra-radius-arcsec", type=float, default=SPECTRA_RADIUS_ARCSEC, help="Spectra crossmatch radius in arcsec (default: 3)")
    parser.add_argument("--spectra-chunk-size", type=int, default=SPECTRA_CHUNK_SIZE, help="Bulk chunk size for spectra lookups")
    parser.add_argument("--spectra-cache", type=Path, default=None, help="Optional cache parquet path for spectra lookups")

    parser.add_argument("--test-run", action="store_true", help="Limit the number of light curves processed (for quick end-to-end validation)")
    parser.add_argument("--test-run-n", type=int, default=10000, help="Number of light curves to sample in test-run mode (default: 10000)")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite checkpoint log and existing output if present (start fresh).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    parser.set_defaults(
        run_post_filter=True,
        run_postprocess=True,
        run_characterize=True,
        run_dust=True,
        characterize_banyan=True,
        characterize_iphas=True,
        characterize_sfr=True,
        characterize_clusters=True,
        characterize_unwise=True,
        run_classify=True,
        run_enrich=True,
        run_neighbor_enrich=True,
        run_spectra_enrich=True,
        export_bundle_enabled=True,
    )

    args = parser.parse_args()

    stage = str(args.stage)
    run_upstream = stage in {"full", "cluster"}
    run_downstream = stage in {"full", "home"}

    if stage != "home" and not args.mag_bin:
        parser.error("--mag-bin is required unless --stage home is used.")

    if stage == "home":
        if args.mag_bin:
            if args.import_bundle is not None:
                bundle_mag_bins = _read_mag_bins_from_bundle(args.import_bundle)
                if bundle_mag_bins is not None:
                    _assert_mag_bin_match(bundle_mag_bins, args.mag_bin, f"{Path(args.import_bundle).expanduser()}/run_params.json")
            if args.out_dir is not None:
                out_dir_params = Path(args.out_dir).expanduser() / "run_params.json"
                out_dir_mag_bins = _read_mag_bins_from_params_file(out_dir_params)
                if out_dir_mag_bins is not None:
                    _assert_mag_bin_match(out_dir_mag_bins, args.mag_bin, str(out_dir_params))
        else:
            if args.import_bundle is None and args.out_dir is None:
                parser.error("--stage home without --mag-bin requires --import-bundle or --out-dir.")

            detected_mag_bins = None
            detected_source = None

            if args.import_bundle is not None:
                detected_mag_bins = _read_mag_bins_from_bundle(args.import_bundle)
                detected_source = f"{Path(args.import_bundle).expanduser()}/run_params.json"

            if detected_mag_bins is None and args.out_dir is not None:
                out_dir_params = Path(args.out_dir).expanduser() / "run_params.json"
                detected_mag_bins = _read_mag_bins_from_params_file(out_dir_params)
                detected_source = str(out_dir_params)

            if not detected_mag_bins:
                parser.error(
                    "Could not auto-detect --mag-bin for --stage home. "
                    "Expected mag_bin in run_params.json from --import-bundle or --out-dir."
                )

            args.mag_bin = detected_mag_bins
            print(f"Info: auto-detected --mag-bin={args.mag_bin} from {detected_source}")

    if stage == "cluster" and (args.run_characterize or args.run_dust or args.run_classify or args.run_neighbor_enrich or args.run_spectra_enrich):
        print("Info: --stage cluster runs upstream only (steps 1-6 plus enrich). Downstream steps are skipped.")
    if stage == "home" and (args.force_manifest or args.force_filter):
        print("Info: --stage home skips manifest/pre-filter/events regardless of force flags.")

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
    base_output_root = Path("output").resolve()
    if args.out_dir is not None:
        out_dir = Path(args.out_dir).expanduser()
    elif args.import_bundle is not None:
        # Auto-derive out_dir from bundle name
        bundle_path = Path(args.import_bundle).expanduser()
        out_dir = get_out_dir_from_bundle(bundle_path, base_output_root)
        log(f"Using output directory from bundle name: {out_dir}")
    elif args.filtered_file is not None:
        out_dir = Path(args.filtered_file).expanduser().parent
    elif args.manifest_file is not None:
        out_dir = Path(args.manifest_file).expanduser().parent
    elif args.output is not None:
        out_dir = Path(args.output).expanduser().parent
    else:
        out_dir = find_latest_run_dir(base_output_root, args.mag_bin)
        if out_dir is not None:
            log(f"Reusing existing run directory: {out_dir}")
        else:
            out_dir = default_run_dir(base_output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.import_bundle is not None:
        import_bundle_zip(args.import_bundle, out_dir)

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

    if stage == "home":
        required_filtered = results_dir / "lc_events_filtered.parquet"
        if not required_filtered.exists():
            source_hint = f" from bundle {args.import_bundle}" if args.import_bundle else ""
            raise FileNotFoundError(
                f"Home stage requires {required_filtered}{source_hint}. "
                "Run cluster/full stage first and transfer results/lc_events_filtered.parquet."
            )

    if args.output is None:
        events_output = results_dir / "lc_events_results.parquet"
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
    run_start_time = datetime.now()

    # Build a compact fingerprint of filtering/characterization behavior.
    if args.pass_all_prefilters:
        enforced_prefilters = []
    elif args.enforce_filters:
        enforced_prefilters = [f.strip() for f in args.enforce_filters.split(",") if f.strip()]
    else:
        enforced_prefilters = []
        if not args.skip_sparse:
            enforced_prefilters.append("sparse")
        if not args.skip_multi_camera:
            enforced_prefilters.append("multi_camera")
        if (not args.skip_vsx) and args.vsx_mode == "filter":
            enforced_prefilters.append("vsx")

    config_fingerprint = {
        "vsx_mode": args.vsx_mode,
        "skip_vsx": args.skip_vsx,
        "pass_all_prefilters": args.pass_all_prefilters,
        "enforced_prefilters": enforced_prefilters,
        "post_filter": {
            "apply_evidence_strength": not args.skip_evidence_strength,
            "min_bayes_factor": args.min_bayes_factor,
            "require_finite_local_bf": not args.allow_infinite_local_bf,
            "apply_run_robustness": not args.skip_run_robustness,
            "min_run_count": args.min_run_count,
            "min_run_cameras": args.post_filter_min_run_cameras,
            "min_run_points": args.post_filter_min_run_points,
            "apply_morphology": args.apply_morphology,
            "dip_morphology": args.dip_morphology,
            "jump_morphology": args.jump_morphology,
            "min_delta_bic": args.min_delta_bic,
            "apply_score": not args.skip_score_filter,
            "min_score": args.min_score,
            "apply_periodicity_validation": args.apply_periodicity_validation,
            "periodicity_n_bootstrap": args.periodicity_n_bootstrap,
            "periodicity_significance": args.periodicity_significance,
            "periodicity_exclude_aliases": not args.periodicity_no_exclude_aliases,
            "periodicity_flag_only": not args.periodicity_reject,
            "periodicity_workers": args.periodicity_workers,
            "periodicity_checkpoint_dir": str(args.periodicity_checkpoint_dir) if args.periodicity_checkpoint_dir else None,
            "phase_plot_max_sig": args.phase_plot_max_sig,
            "phase_plot_min_power": args.phase_plot_min_power,
            "phase_plot_allow_alias": args.phase_plot_allow_alias,
            "apply_gaia_ruwe_validation": not args.skip_gaia_ruwe_validation,
            "gaia_max_ruwe": args.gaia_max_ruwe,
            "gaia_flag_only": not args.gaia_reject,
            "apply_periodic_catalog_validation": not args.skip_periodic_catalog_validation,
            "periodic_catalog_max_sep": args.periodic_catalog_max_sep,
            "periodic_catalog_flag_only": not args.periodic_catalog_reject,
        },
        "characterize": {
            "run_characterize": args.run_characterize,
            "run_dust": args.run_dust,
            "starhorse": args.characterize_starhorse,
            "banyan": args.characterize_banyan,
            "iphas": args.characterize_iphas,
            "sfr": args.characterize_sfr,
            "clusters": args.characterize_clusters,
            "unwise": args.characterize_unwise,
        },
        "downstream_pass_logic": "characterize/classify/enrich run on post-filter passers (failed_any == False)",
    }

    run_params_file = out_dir / "run_params.json"
    run_summary_file = out_dir / "run_summary.json"
    summary: dict[str, object] = {
        "run_info": {
            "start_time": run_start_time.isoformat(),
            "stage": stage,
        }
    }
    if run_summary_file.exists():
        try:
            with open(run_summary_file) as f:
                existing_summary = json.load(f)
            # Preserve existing stats, update run_info
            existing_summary.update(summary)
            summary = existing_summary
        except Exception:
            pass

    results_files: list[Path] = []
    cmd = shlex.join(getattr(sys, "orig_argv", None) or ([sys.executable] + sys.argv))
    try:
        run_params = {
            "timestamp": run_start_time.isoformat(),
            "command": cmd,
            "stage": stage,
            "import_bundle": str(args.import_bundle) if args.import_bundle else None,
            "export_bundle": str(args.export_bundle) if args.export_bundle else None,
            "export_bundle_enabled": args.export_bundle_enabled,
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
            "skip_evidence_strength": args.skip_evidence_strength,
            "min_bayes_factor": args.min_bayes_factor,
            "allow_infinite_local_bf": args.allow_infinite_local_bf,
            "skip_run_robustness": args.skip_run_robustness,
            "min_run_count": args.min_run_count,
            "post_filter_min_run_cameras": args.post_filter_min_run_cameras,
            "post_filter_min_run_points": args.post_filter_min_run_points,
            "apply_morphology": args.apply_morphology,
            "dip_morphology": args.dip_morphology,
            "jump_morphology": args.jump_morphology,
            "min_delta_bic": args.min_delta_bic,
            "skip_score_filter": args.skip_score_filter,
            "min_score": args.min_score,
            "apply_periodicity_validation": args.apply_periodicity_validation,
            "periodicity_n_bootstrap": args.periodicity_n_bootstrap,
            "periodicity_significance": args.periodicity_significance,
            "periodicity_no_exclude_aliases": args.periodicity_no_exclude_aliases,
            "periodicity_reject": args.periodicity_reject,
            "periodicity_workers": args.periodicity_workers,
            "periodicity_checkpoint_dir": str(args.periodicity_checkpoint_dir) if args.periodicity_checkpoint_dir else None,
            "phase_plot_max_sig": args.phase_plot_max_sig,
            "phase_plot_min_power": args.phase_plot_min_power,
            "phase_plot_allow_alias": args.phase_plot_allow_alias,
            "skip_gaia_ruwe_validation": args.skip_gaia_ruwe_validation,
            "gaia_max_ruwe": args.gaia_max_ruwe,
            "gaia_reject": args.gaia_reject,
            "skip_periodic_catalog_validation": args.skip_periodic_catalog_validation,
            "periodic_catalog_max_sep": args.periodic_catalog_max_sep,
            "periodic_catalog_reject": args.periodic_catalog_reject,
            # Step 7: Postprocess
            "run_postprocess": args.run_postprocess,
            "max_plots": args.max_plots,
            "plot_format": args.plot_format,
            # Step 8: Characterization
            "run_characterize": args.run_characterize,
            "run_dust": args.run_dust,
            "gaia_cache": str(args.gaia_cache),
            "characterize_crossmatch": str(args.characterize_crossmatch),
            "characterize_chunk_size": args.characterize_chunk_size,
            "characterize_starhorse": args.characterize_starhorse,
            "characterize_banyan": args.characterize_banyan,
            "characterize_iphas": args.characterize_iphas,
            "characterize_sfr": args.characterize_sfr,
            "characterize_clusters": args.characterize_clusters,
            "characterize_unwise": args.characterize_unwise,
            # Step 9: Classify
            "run_classify": args.run_classify,
            # Step 6: Enrich
            "run_enrich": args.run_enrich,
            "enrich_compute_ls": args.enrich_compute_ls,
            # Step 10: Neighbor enrichment
            "run_neighbor_enrich": args.run_neighbor_enrich,
            "neighbor_radius_arcsec": args.neighbor_radius_arcsec,
            "neighbor_chunk_size": args.neighbor_chunk_size,
            "neighbor_cache": str(args.neighbor_cache) if args.neighbor_cache else None,
            # Step 11: Spectra enrichment
            "run_spectra_enrich": args.run_spectra_enrich,
            "spectra_radius_arcsec": args.spectra_radius_arcsec,
            "spectra_chunk_size": args.spectra_chunk_size,
            "spectra_cache": str(args.spectra_cache) if args.spectra_cache else None,
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

    df_manifest = pd.DataFrame()
    df_filtered = pd.DataFrame()

    # Step 1: Build or load manifest
    if run_upstream:
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

            # Use lc_dir as the directory path for pre_filter input (path/<id>.dat2)
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

    # Test-run sampling: cap sources to limit expensive downstream steps
    if run_upstream and args.test_run and len(df_filtered) > args.test_run_n:
        log(f"\n[TEST RUN] Sampling {args.test_run_n}/{len(df_filtered)} sources")
        df_filtered = df_filtered.sample(n=args.test_run_n, random_state=42).reset_index(drop=True)

    # Step 2.5: Apply camera median filter to identify cameras to exclude
    camera_median_file = prefilter_dir / f"camera_medians_{mag_bin_tag}.parquet"
    if run_upstream and (not args.skip_camera_median) and ("mag_bin" in df_filtered.columns):
        if args.force_filter or not camera_median_file.exists():
            log(f"\nApplying camera median filter (tolerance={args.camera_median_tolerance} mag)...")
            # Camera median validation needs per-source file paths (.dat2 -> .raw2).
            # Keep the original path column unchanged for downstream code.
            camera_median_df = df_filtered.copy()
            if "dat_path" in camera_median_df.columns:
                camera_median_df["path"] = camera_median_df["dat_path"]
            camera_median_checkpoint = prefilter_dir / f"camera_medians_{mag_bin_tag}_CHECKPOINT.parquet"
            df_camera = filter_camera_medians(
                camera_median_df,
                mag_tolerance=args.camera_median_tolerance,
                show_tqdm=args.verbose,
                n_workers=args.workers,
                checkpoint_path=str(camera_median_checkpoint),
            )
            df_filtered["excluded_cameras"] = df_camera["excluded_cameras"]
            safe_write_parquet(df_filtered[["source_id", "excluded_cameras"]], camera_median_file)
        else:
            log(f"\nLoading cached camera median results from {camera_median_file}")
            cam_cache = pd.read_parquet(camera_median_file)
            df_filtered = df_filtered.merge(cam_cache, on="source_id", how="left")
        n_with_exclusions = (df_filtered["excluded_cameras"].fillna("") != "").sum()
        log(f"Found {n_with_exclusions}/{len(df_filtered)} sources with excluded cameras")

    # Step 3: Construct file paths (use full dat_path for events.py input)
    file_col = "dat_path" if "dat_path" in df_filtered.columns else "path"

    # Build metadata CSV with VSX tags and excluded_cameras
    metadata_file = None
    meta_cols = [file_col]
    if not args.skip_vsx and "vsx_sep_arcsec" in df_filtered.columns and "vsx_class" in df_filtered.columns:
        meta_cols.extend(["vsx_sep_arcsec", "vsx_class"])
    if "excluded_cameras" in df_filtered.columns:
        meta_cols.append("excluded_cameras")

    if run_upstream and len(meta_cols) > 1:  # More than just file_col
        metadata_dir = prefilter_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = metadata_dir / f"metadata_{mag_bin_tag}.csv"
        meta_df = df_filtered[meta_cols].rename(columns={file_col: "path"})
        meta_df.to_csv(metadata_file, index=False)
        events_args.extend(["--metadata-csv", str(metadata_file)])
        log(f"Wrote metadata CSV with columns: {', '.join(meta_cols[1:])}")

    file_paths = df_filtered[file_col].tolist() if run_upstream else []

    if run_upstream and (not file_paths):
        log("\nNo sources to process after filtering!")
        return

    # Step 4: Call events.py with the filtered paths in batches, with resume support
    if run_upstream:
        log(f"\nPreparing to run events.py on {len(file_paths)} light curves...")

    # Write paths to temp file for events.py to consume
    paths_file = paths_dir / f"filtered_paths_{mag_bin_tag}.txt"
    if run_upstream:
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
    base_output = events_output or (results_dir / "lc_events_results.parquet")
    suffix_map = {"csv": ".csv", "parquet": ".parquet", "parquet_chunk": None}
    ext = suffix_map.get(events_format)
    if ext and base_output.suffix.lower() != ext:
        base_output = base_output.with_suffix(ext)
    checkpoint_log = base_output.with_name(f"{base_output.stem}_PROCESSED.txt")
    processed_paths: set[str] = set()
    if run_upstream and checkpoint_log.exists() and args.overwrite:
        try:
            with open(checkpoint_log, "w"):
                pass
            log(f"Overwriting checkpoint log: {checkpoint_log}")
        except Exception as e:
            log(f"Warning: could not overwrite checkpoint log {checkpoint_log}: {e}")

    if run_upstream and args.overwrite:
        clear_existing_output(base_output, events_format)

    if run_upstream and checkpoint_log.exists() and not args.overwrite:
        try:
            with open(checkpoint_log, "r") as f:
                processed_paths = {line.strip() for line in f if line.strip()}
            log(f"Checkpoint detected, skipping {len(processed_paths)} already-processed paths")
        except Exception as e:
            log(f"Warning: could not read checkpoint log {checkpoint_log}: {e}")

    remaining = [p for p in file_paths if str(p) not in processed_paths]
    if run_upstream and (not remaining):
        log("All paths already processed according to checkpoint.")

    # Batch and run
    batch_size = max(1, args.batch_size)
    total_batches = (len(remaining) + batch_size - 1) // batch_size if run_upstream else 0
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(len(remaining), start + batch_size)
        batch_paths = remaining[start:end]

        log(f"\nRunning batch {batch_idx + 1}/{total_batches} ({len(batch_paths)} LCs)...")

        events_cmd = [
            sys.executable, "-m", "malca.events",
            *events_args,
            "--input",
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

    if run_upstream:
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
            "config_fingerprint": config_fingerprint,
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
    if run_upstream and args.run_post_filter and results_files:
        log("\n=== Step 5: Applying post-filters ===")
        try:
            # Load events results
            if events_format == "parquet_chunk":
                df_events = pd.concat([pd.read_parquet(f) for f in results_files], ignore_index=True)
            else:
                df_events = load_table(results_files[0])

            # Apply post-filters
            post_filter_kwargs = _build_post_filter_kwargs(args)
            if stage == "cluster":
                # Cluster stage must avoid internet catalog lookups.
                post_filter_kwargs["apply_gaia_ruwe_validation"] = False
                post_filter_kwargs["apply_periodic_catalog_validation"] = False
            df_post_filtered = apply_post_filters(df_events, **post_filter_kwargs)

            # Save filtered results
            post_filter_output = results_dir / "lc_events_filtered.parquet"
            save_table(df_post_filtered, post_filter_output)
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

    # Step 6: Enrich with compute_stats (optional, runs immediately after post-filter)
    if run_upstream and args.run_enrich:
        if not args.run_post_filter:
            print("Warning: --run-enrich requires --run-post-filter. Skipping enrichment.")
        else:
            log("\n=== Step 6: Enriching with light curve stats ===")
            try:
                # Enrichment now runs directly from post-filter output
                post_filter_output = results_dir / "lc_events_filtered.parquet"

                if post_filter_output.exists():
                    df_to_enrich = load_table(post_filter_output)
                else:
                    print(f"Warning: No post-filter output found at {post_filter_output}")
                    df_to_enrich = None

                if df_to_enrich is not None:
                    # Filter to passing candidates only
                    if "failed_any" in df_to_enrich.columns:
                        df_passed = df_to_enrich[~df_to_enrich["failed_any"]].copy()
                    else:
                        df_passed = df_to_enrich.copy()

                    if len(df_passed) > 0:
                        log(f"Enriching {len(df_passed)} candidates with compute_stats...")

                        # Checkpoint support
                        enrich_checkpoint = results_dir / "lc_events_enriched_CHECKPOINT.parquet"
                        if args.overwrite and enrich_checkpoint.exists():
                            enrich_checkpoint.unlink()

                        already_enriched: set[str] = set()
                        enriched_rows: list[dict] = []
                        if enrich_checkpoint.exists():
                            try:
                                df_ckpt = pd.read_parquet(enrich_checkpoint)
                                enriched_rows = df_ckpt.to_dict("records")
                                already_enriched = set(df_ckpt["path"].astype(str))
                                log(f"Loaded enrichment checkpoint: {len(already_enriched)} already enriched")
                            except Exception as e:
                                log(f"Warning: could not load enrichment checkpoint: {e}")

                        ENRICH_SAVE_INTERVAL = 10000
                        new_count = 0
                        for idx, row in tqdm(df_passed.iterrows(), total=len(df_passed),
                                            desc="compute_stats", disable=not args.verbose):
                            lc_path = Path(row["path"])
                            if str(lc_path) in already_enriched:
                                continue
                            if not lc_path.exists():
                                enriched_rows.append(row.to_dict())
                                new_count += 1
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
                                    if isinstance(v, dict):
                                        for sub_k, sub_v in v.items():
                                            col = f"stats_{k}_{sub_k}"
                                            if col not in merged:
                                                merged[col] = sub_v
                                    elif isinstance(v, (pd.DataFrame, pd.Series)):
                                        continue
                                    elif f"stats_{k}" not in merged:
                                        merged[f"stats_{k}"] = v
                                enriched_rows.append(merged)

                            except Exception as e:
                                if args.verbose:
                                    print(f"Warning: compute_stats failed for {lc_path}: {e}")
                                enriched_rows.append(row.to_dict())

                            new_count += 1
                            if new_count % ENRICH_SAVE_INTERVAL == 0:
                                pd.DataFrame(enriched_rows).to_parquet(
                                    enrich_checkpoint, index=False,
                                    compression=PARQUET_CACHE_COMPRESSION,
                                )

                        df_enriched = pd.DataFrame(enriched_rows)
                        if not df_enriched.empty:
                            df_enriched = df_enriched.drop_duplicates(subset=["path"], keep="last")

                        # Save enriched results
                        enrich_output = results_dir / "lc_events_enriched.parquet"
                        save_table(df_enriched, enrich_output)
                        log(f"Enriched results saved to {enrich_output}")

                        # Clean up checkpoint
                        if enrich_checkpoint.exists():
                            enrich_checkpoint.unlink()

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

    # Step 7: Generate review plots (optional)
    if run_upstream and args.run_postprocess:
        if not args.run_post_filter:
            print("Warning: --run-postprocess requires --run-post-filter. Skipping postprocess plots.")
        else:
            log("\n=== Step 7: Generating candidate plots ===")
            try:
                post_filter_output = results_dir / "lc_events_filtered.parquet"
                if not post_filter_output.exists():
                    print(f"Warning: No post-filter output found at {post_filter_output}; skipping postprocess plots.")
                else:
                    plots_out = out_dir / "plots" / "candidates"
                    baseline_for_plots = {
                        "gp": "per_camera_gp",
                        "gp_masked": "per_camera_gp",
                        "per_camera_median": "per_camera_median",
                        "global_median": "global_median",
                    }.get(str(args.baseline_func), "per_camera_gp")

                    plot_summary = plot_passing_candidates(
                        post_filter_output,
                        plots_out,
                        require_failed_any_false=True,
                        max_plots=args.max_plots,
                        baseline=baseline_for_plots,
                        baseline_kwargs={},
                        skip_events=False,
                        plot_fits=False,
                        format=args.plot_format,
                        show=False,
                        verbose=args.verbose,
                        workers=max(1, int(args.workers)),
                        logbf_threshold_dip=float(args.logbf_threshold_dip),
                        logbf_threshold_jump=float(args.logbf_threshold_jump),
                        jd_offset=2458000.0,
                        clean_max_error_absolute=1.0,
                        clean_max_error_sigma=5.0,
                        run_params=run_params if 'run_params' in locals() else None,
                        filter_bad_cameras=True,
                        bad_camera_scatter_ratio=2.5,
                        show_tqdm=args.verbose,
                    )

                    summary["postprocess_stats"] = {
                        "output_dir": str(plots_out),
                        "total_selected": int(plot_summary.get("total_selected", 0)),
                        "plotted": int(plot_summary.get("plotted", 0)),
                        "failed": int(plot_summary.get("failed", 0)),
                        "phase_plotted": int(plot_summary.get("phase_plotted", 0)),
                    }
                    with open(run_summary_file, "w") as f:
                        json.dump(summary, f, indent=2, default=str)
                    log(f"Postprocess plots written to {plots_out}")
            except Exception as e:
                print(f"Error in postprocess plotting step: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()



    post_filter_output = results_dir / "lc_events_filtered.parquet"
    has_post_filter_output = post_filter_output.exists()

    # Home-only external catalog validations (Gaia RUWE + periodic catalog)
    if stage == "home" and args.run_post_filter and has_post_filter_output:
        log("\n=== Home External Validation: Gaia RUWE + periodic catalog ===")
        try:
            bundled_index = out_dir / "bundle_assets" / "asassn_index_full.parquet"
            if bundled_index.exists():
                index_file = bundled_index
            else:
                index_file = ASASSN_INDEX_PATH.expanduser()

            if not index_file.exists():
                raise FileNotFoundError(
                    f"Index file not found for home external validation: {index_file}. "
                    "Expected bundle_assets/asassn_index_full.parquet from export bundle."
                )

            external_validation_cmd = [
                sys.executable,
                "-m",
                "malca.post_filter",
                "--input",
                str(post_filter_output),
                "--output",
                str(post_filter_output),
                "--index-file",
                str(index_file),
                "--skip-evidence-strength",
                "--skip-run-robustness",
                "--gaia-max-ruwe",
                str(args.gaia_max_ruwe),
                "--periodic-catalog-max-sep",
                str(args.periodic_catalog_max_sep),
            ]
            if args.gaia_reject:
                external_validation_cmd.append("--gaia-reject")
            if args.periodic_catalog_reject:
                external_validation_cmd.append("--periodic-catalog-reject")
            if args.skip_gaia_ruwe_validation:
                external_validation_cmd.append("--skip-gaia-ruwe-validation")
            if args.skip_periodic_catalog_validation:
                external_validation_cmd.append("--skip-periodic-catalog-validation")
            if not args.verbose:
                external_validation_cmd.append("--no-tqdm")
            if args.verbose:
                external_validation_cmd.append("--verbose")

            result = subprocess.run(external_validation_cmd, check=False)
            if result.returncode != 0:
                print(f"Home external validation failed with exit code {result.returncode}")
                sys.exit(result.returncode)

            has_post_filter_output = post_filter_output.exists()
            log(f"Home external validation wrote updated filtered results to {post_filter_output}")
        except Exception as e:
            print(f"Error in home external validation step: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)



    if run_downstream and (args.run_characterize or args.run_dust) and (not has_post_filter_output):
        print(f"Warning: downstream stage requires filtered results at {post_filter_output}. Skipping characterization.")

    # Step 7b: Auto-fetch Gaia data for characterization (incremental)
    if run_downstream and (args.run_characterize or args.run_dust) and has_post_filter_output:
        log("\n=== Ensuring local Gaia catalog is up to date ===")
        try:
            gaia_catalog_path = args.gaia_cache.expanduser() if args.gaia_cache else GAIA_LOCAL_CATALOG
            gaia_ids = _extract_gaia_ids(
                post_filter_output,
                args.characterize_crossmatch.expanduser(),
            )
            if gaia_ids:
                fetch_gaia_catalog(gaia_ids, output_path=gaia_catalog_path)
            else:
                log("No Gaia IDs found; skipping Gaia fetch.")
        except Exception as e:
            print(f"Warning: Gaia auto-fetch failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Step 8: Characterization + dust (optional)
    if run_downstream and (args.run_characterize or args.run_dust) and has_post_filter_output:
        log("\n=== Step 8: Characterizing candidates ===")
        try:
            df_char = load_table(post_filter_output)

            if "failed_any" in df_char.columns:
                df_char = df_char[~df_char["failed_any"]].copy()

            if "path" in df_char.columns and "asas_sn_id" not in df_char.columns:
                def _extract_id(path_str: str) -> str:
                    name = Path(path_str).name
                    return Path(name).stem.split("-")[0]

                df_char["asas_sn_id"] = df_char["path"].astype(str).map(_extract_id)

            # Use full characterize pipeline (single source of truth)
            char_checkpoint = results_dir / "lc_events_characterized_CHECKPOINT.parquet"
            if args.overwrite and char_checkpoint.exists():
                char_checkpoint.unlink()

            starhorse_arg = args.characterize_starhorse if args.run_characterize else None
            df_char = characterize_candidates_df(
                df_char,
                crossmatch=args.characterize_crossmatch.expanduser(),
                chunk_size=args.characterize_chunk_size,
                cache=args.gaia_cache.expanduser() if args.gaia_cache else (out_dir / "gaia_cache" / "gaia_cache.parquet"),
                dust=args.run_dust,
                starhorse=starhorse_arg,
                run_banyan=args.run_characterize and args.characterize_banyan,
                run_iphas=args.run_characterize and args.characterize_iphas,
                run_sfr=args.run_characterize and args.characterize_sfr,
                run_clusters=args.run_characterize and args.characterize_clusters,
                run_unwise=args.run_characterize and args.characterize_unwise,
                checkpoint_path=char_checkpoint,
            )

            characterize_output = results_dir / "lc_events_characterized.parquet"
            save_table(df_char, characterize_output)
            log(f"Characterization results saved to {characterize_output}")

        except Exception as e:
            print(f"Error in characterization step: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Step 9: Run classification (optional)
    if run_downstream and args.run_classify:
        if not has_post_filter_output:
            print(f"Warning: downstream stage requires filtered results at {post_filter_output}. Skipping classification.")
        else:
            classify_output = results_dir / "lc_events_classified.parquet"
            if classify_output.exists() and not args.overwrite:
                log(f"\n=== Step 9: Classification output exists, skipping: {classify_output} ===")
            else:
                log("\n=== Step 9: Running classification ===")
                try:
                    characterize_output = results_dir / "lc_events_characterized.parquet"
                    post_filter_output = results_dir / "lc_events_filtered.parquet"

                    if characterize_output.exists():
                        df_post_filtered = load_table(characterize_output)
                    elif post_filter_output.exists():
                        df_post_filtered = load_table(post_filter_output)
                    else:
                        df_post_filtered = None
                        print(f"Warning: post-filter output not found at {post_filter_output}")

                    if df_post_filtered is not None:
                        # Run classification on passing candidates
                        df_passed = df_post_filtered[~df_post_filtered["failed_any"]].copy() if "failed_any" in df_post_filtered.columns else df_post_filtered.copy()

                        if len(df_passed) > 0:
                            df_classified = compute_all_classifications(df_passed)

                            # Save classified results
                            save_table(df_classified, classify_output)
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

    # Step 10: Neighbor enrichment (optional)
    if run_downstream and args.run_neighbor_enrich:
        if not has_post_filter_output:
            print(f"Warning: downstream stage requires filtered results at {post_filter_output}. Skipping neighbor enrichment.")
        else:
            log("\n=== Step 10: Bulk neighbor enrichment ===")
            try:
                enrich_output = results_dir / "lc_events_enriched.parquet"
                classify_output = results_dir / "lc_events_classified.parquet"
                characterize_output = results_dir / "lc_events_characterized.parquet"
                post_filter_output = results_dir / "lc_events_filtered.parquet"

                if classify_output.exists():
                    df_neighbors_in = load_table(classify_output)
                elif characterize_output.exists():
                    df_neighbors_in = load_table(characterize_output)
                elif enrich_output.exists():
                    df_neighbors_in = load_table(enrich_output)
                elif post_filter_output.exists():
                    df_neighbors_in = load_table(post_filter_output)
                else:
                    df_neighbors_in = None

                if df_neighbors_in is not None:
                    if "failed_any" in df_neighbors_in.columns:
                        df_neighbors_in = df_neighbors_in[~df_neighbors_in["failed_any"]].copy()

                    neighbor_dir = results_dir / "neighbor_enrichment"
                    neighbor_cache = args.neighbor_cache.expanduser() if args.neighbor_cache else (neighbor_dir / "neighbors_cache.parquet")
                    neighbor_checkpoint = neighbor_dir / "neighbors_CHECKPOINT.parquet"
                    if args.overwrite and neighbor_checkpoint.exists():
                        neighbor_checkpoint.unlink()
                    _, df_neighbor_summary = run_neighbor_enrichment(
                        df_neighbors_in,
                        out_dir=neighbor_dir,
                        radius_arcsec=args.neighbor_radius_arcsec,
                        chunk_size=args.neighbor_chunk_size,
                        cache_file=neighbor_cache,
                        checkpoint_path=neighbor_checkpoint,
                    )

                    if not df_neighbor_summary.empty:
                        key_col = "candidate_id" if "candidate_id" in df_neighbors_in.columns else "asas_sn_id"
                        left = df_neighbors_in.copy()
                        if key_col not in left.columns and "path" in left.columns:
                            left[key_col] = left["path"].astype(str).map(lambda p: Path(p).stem.split("-")[0])
                        left[key_col] = left[key_col].astype(str)
                        right = df_neighbor_summary.copy()
                        right["candidate_id"] = right["candidate_id"].astype(str)
                        merged = left.merge(right, left_on=key_col, right_on="candidate_id", how="left")
                        save_table(merged, results_dir / "lc_events_neighbors.parquet")

                    summary["neighbor_enrichment_stats"] = {
                        "rows_input": int(len(df_neighbors_in)),
                        "radius_arcsec": float(args.neighbor_radius_arcsec),
                        "chunk_size": int(args.neighbor_chunk_size),
                        "output_dir": str(neighbor_dir),
                    }
                    with open(run_summary_file, "w") as f:
                        json.dump(summary, f, indent=2, default=str)
                    log(f"Neighbor enrichment outputs written to {neighbor_dir}")

            except Exception as e:
                print(f"Error in neighbor enrichment step: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

    # Step 11: Spectra availability enrichment (optional)
    if run_downstream and args.run_spectra_enrich:
        if not has_post_filter_output:
            print(f"Warning: downstream stage requires filtered results at {post_filter_output}. Skipping spectra enrichment.")
        else:
            log("\n=== Step 11: Spectra availability enrichment ===")
            try:
                neighbor_output = results_dir / "lc_events_neighbors.parquet"
                enrich_output = results_dir / "lc_events_enriched.parquet"
                classify_output = results_dir / "lc_events_classified.parquet"
                characterize_output = results_dir / "lc_events_characterized.parquet"
                post_filter_output = results_dir / "lc_events_filtered.parquet"

                if neighbor_output.exists():
                    df_spectra_in = load_table(neighbor_output)
                elif enrich_output.exists():
                    df_spectra_in = load_table(enrich_output)
                elif classify_output.exists():
                    df_spectra_in = load_table(classify_output)
                elif characterize_output.exists():
                    df_spectra_in = load_table(characterize_output)
                elif post_filter_output.exists():
                    df_spectra_in = load_table(post_filter_output)
                else:
                    df_spectra_in = None

                if df_spectra_in is not None:
                    if "failed_any" in df_spectra_in.columns:
                        df_spectra_in = df_spectra_in[~df_spectra_in["failed_any"]].copy()

                    spectra_dir = results_dir / "spectra_enrichment"
                    spectra_cache = args.spectra_cache.expanduser() if args.spectra_cache else (spectra_dir / "spectra_cache.parquet")
                    spectra_checkpoint = spectra_dir / "spectra_CHECKPOINT.parquet"
                    if args.overwrite and spectra_checkpoint.exists():
                        spectra_checkpoint.unlink()
                    _, spectra_summary = run_spectra_availability(
                        df_spectra_in,
                        out_dir=spectra_dir,
                        radius_arcsec=args.spectra_radius_arcsec,
                        chunk_size=args.spectra_chunk_size,
                        cache_file=spectra_cache,
                        checkpoint_path=spectra_checkpoint,
                    )

                    if not spectra_summary.empty:
                        key_col = "candidate_id" if "candidate_id" in df_spectra_in.columns else "asas_sn_id"
                        left = df_spectra_in.copy()
                        if key_col not in left.columns and "path" in left.columns:
                            left[key_col] = left["path"].astype(str).map(lambda p: Path(p).stem.split("-")[0])
                        left[key_col] = left[key_col].astype(str)
                        right = spectra_summary.copy()
                        right["candidate_id"] = right["candidate_id"].astype(str)
                        merged = left.merge(right, left_on=key_col, right_on="candidate_id", how="left")
                        save_table(merged, results_dir / "lc_events_spectra.parquet")

                    summary["spectra_enrichment_stats"] = {
                        "rows_input": int(len(df_spectra_in)),
                        "radius_arcsec": float(args.spectra_radius_arcsec),
                        "chunk_size": int(args.spectra_chunk_size),
                        "output_dir": str(spectra_dir),
                    }
                    with open(run_summary_file, "w") as f:
                        json.dump(summary, f, indent=2, default=str)
                    log(f"Spectra enrichment outputs written to {spectra_dir}")

            except Exception as e:
                print(f"Error in spectra enrichment step: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

    if args.export_bundle_enabled:
        export_bundle_path = args.export_bundle if args.export_bundle is not None else out_dir / f"{out_dir.name}_bundle.zip"
        log(f"\n=== Exporting bundle to {export_bundle_path} ===")
        try:
            if args.full_bundle:
                source_index_file = ASASSN_INDEX_PATH.expanduser()
                if source_index_file.exists():
                    bundle_assets_dir = out_dir / "bundle_assets"
                    bundle_assets_dir.mkdir(parents=True, exist_ok=True)
                    bundle_index_file = bundle_assets_dir / "asassn_index_full.parquet"
                    if (not bundle_index_file.exists()) or (bundle_index_file.stat().st_size != source_index_file.stat().st_size):
                        log(f"Copying full index into bundle assets: {source_index_file} -> {bundle_index_file}")
                        shutil.copy2(source_index_file, bundle_index_file)
                else:
                    raise FileNotFoundError(
                        f"Required index file not found for bundle export: {source_index_file}"
                    )

            bundled = export_bundle_zip(export_bundle_path, out_dir, include_all=args.full_bundle)
            log(f"Exported bundle to {export_bundle_path.expanduser()} with {len(bundled)} files")
        except Exception as e:
            print(f"Error creating export bundle: {e}")


if __name__ == "__main__":
    main()
