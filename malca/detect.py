#!/usr/bin/env python3
"""
Wrapper script to run events.py on tagged light curves.

Workflow:
1. Build/load manifest (source_id → lc_dir mapping)
2. Apply tags (sparse, periodic, multi-camera)
3. Construct file paths for kept sources
4. Pass to events.py
5. [Optional] Apply filters (posterior strength, run robustness, etc.)
6. [Optional] Generate postprocess plots for passing candidates
7. [Optional] Run characterization (Gaia DR3 + dust extinction)
8. [Optional] Run classification (EB/CV/starspot rejection, YSO classification)
9. [Optional] Enrich passing candidates with comprehensive light curve stats

Usage:
    malca detect --mag-bin 13_13.5 [options...]
    malca detect --mag-bin 13_13.5 14_14.5 [options...]  # Process multiple bins together
    malca detect --mag-bin all [options...]  # Process all 6 magnitude bins together
    malca detect --mag-bin 13_13.5 --run-filter --run-classify --run-enrich
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

from tqdm.auto import tqdm
import pandas as pd

from malca.characterize import characterize_candidates_df
from malca.classify import compute_all_classifications
from malca.config import (
    GAIA_CHUNK_SIZE, NEIGHBOR_RADIUS_ARCSEC, NEIGHBOR_CHUNK_SIZE,
    SPECTRA_RADIUS_ARCSEC, SPECTRA_CHUNK_SIZE,
    UNWISE_CHECKPOINT_EVERY,
)
from malca.config import (
    MIN_TIME_SPAN, MIN_POINTS_PER_DAY, MIN_CAMERAS,
    VSX_MAX_SEP_ARCSEC, VSX_MODE, CAMERA_MEDIAN_TOLERANCE, STATS_CHUNK_SIZE,
    MIN_BAYES_FACTOR, POST_FILTER_MIN_RUN_CAMERAS, POST_FILTER_MIN_RUN_POINTS,
    CLEAN_LC_MAX_ERROR_ABSOLUTE, CLEAN_LC_MAX_ERROR_SIGMA,
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    PRE_PERIODICITY_CE_SNR_THRESHOLD, PRE_PERIODICITY_MAX_PERIOD,
    PRE_PERIODICITY_MIN_POINTS, PRE_PERIODICITY_MIN_PERIOD,
    PRE_PERIODICITY_N_PERIODS, PRE_PERIODICITY_SCATTER_RATIO_MAX,
    POST_FILTER_PDM_METHOD,
)
from malca.config import OUTPUT_FORMAT, EVENTS_OUTPUT_CHUNK_SIZE
from malca.config import PARQUET_OUTPUT_COMPRESSION, PARQUET_CACHE_COMPRESSION
from malca.config import ASASSN_INDEX_PATH, LCV2_ROOT, VSX_CROSSMATCH_PATH, GAIA_LOCAL_CATALOG
from malca.config import (
    WORKERS, BATCH_SIZE, TRIGGER_MODE, P_POINTS, MAG_POINTS,
    LOGBF_THRESHOLD_DIP, LOGBF_THRESHOLD_JUMP, SIGNIFICANCE_THRESHOLD,
    MIN_MAG_OFFSET, RUN_MIN_POINTS, RUN_MAX_GAP_POINTS,
    BASELINE_FUNC, BASELINE_S0, BASELINE_W0, BASELINE_Q, BASELINE_JITTER,
    JD_OFFSET, MAG_BINS,
)
from malca.config import PDM_METHOD_CHOICES
from malca.enrich.neighbor import run_neighbor_enrichment
from malca.enrich.spectra import run_spectra_availability
from malca.gaia_fetch import _extract_gaia_ids, fetch_gaia_catalog
from malca.manifest import build_manifest
from malca.periodicity_gate import apply_pre_periodicity_gate, PREGATE_ROUTER_MODE
from malca.plot import plot_passing_candidates
from malca.filter import apply_filters
from malca.review.sync import auto_export_review_bundle
from malca.run_metadata import build_run_summary, load_summary_state, preserve_imported_run_snapshots
from malca.review.store import db_connect, import_candidates
from concurrent.futures import ProcessPoolExecutor
from malca.stats import compute_stats, _enrich_row_worker
from malca.tag import apply_tags, filter_camera_medians
from malca.utils import log as _log
from malca.vetting import vet_candidates



# Set threading environment variables before importing numpy/pandas/numba
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")




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


def _select_passing_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows with failed_any == False when that column exists."""
    if "failed_any" not in df.columns:
        return df.copy()

    failed = df["failed_any"]
    if pd.api.types.is_bool_dtype(failed):
        keep = ~failed.fillna(False).astype(bool)
    elif pd.api.types.is_numeric_dtype(failed):
        keep = failed.fillna(0).astype(float) == 0.0
    else:
        lowered = failed.fillna("").astype(str).str.strip().str.lower()
        keep = ~lowered.isin({"1", "true", "t", "yes", "y"})

    return df.loc[keep].copy()


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        try:
            key = str(path.resolve(strict=False))
        except Exception:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique



def _candidate_asassn_index_paths(out_dir: Path, index_override: Path | None = None) -> list[Path]:
    out_dir = Path(out_dir).expanduser()
    default_index = ASASSN_INDEX_PATH.expanduser()
    output_root = out_dir.parents[1] if len(out_dir.parents) >= 2 else out_dir.parent

    candidates: list[Path] = []
    if index_override is not None:
        candidates.append(Path(index_override).expanduser())

    candidates.extend([
        out_dir / "bundle_assets" / "asassn_index_full.parquet",
        out_dir / "bundle_assets" / default_index.name,
        out_dir / default_index.name,
        out_dir / "input" / default_index.name,
        output_root / default_index.name,
        default_index,
    ])

    search_dirs = [
        out_dir / "bundle_assets",
        out_dir / "input",
        out_dir,
        output_root,
        Path("input"),
        Path("output"),
    ]
    for search_dir in _unique_paths(search_dirs):
        if not search_dir.exists() or (not search_dir.is_dir()):
            continue
        for pattern in ("asassn_index*.parquet", "asassn_index*.pq", "asassn_index*.csv"):
            candidates.extend(sorted(search_dir.glob(pattern)))

    return _unique_paths(candidates)


def _resolve_asassn_index_path(out_dir: Path, index_override: Path | None = None) -> tuple[Path | None, list[Path]]:
    candidates = _candidate_asassn_index_paths(out_dir, index_override=index_override)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate, candidates
    return None, candidates


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


_BUNDLE_MAG_BIN_RE = re.compile(
    r"^results/lc_events_(?:results|filtered|enriched)_([0-9.]+_[0-9.]+)\.parquet$"
)


def _infer_mag_bins_from_bundle_contents(zf: zipfile.ZipFile) -> list[str] | None:
    tags: set[str] = set()
    for name in zf.namelist():
        m = _BUNDLE_MAG_BIN_RE.match(str(name))
        if not m:
            continue
        tag = str(m.group(1))
        if tag and tag != "multi":
            tags.add(tag)
    if not tags:
        return None
    return sorted(tags)


def _read_mag_bins_from_bundle(bundle_zip: Path) -> list[str] | None:
    bundle_zip = Path(bundle_zip).expanduser()
    if not bundle_zip.exists() or (not zipfile.is_zipfile(bundle_zip)):
        return None
    try:
        with zipfile.ZipFile(bundle_zip, "r") as zf:
            inferred = _infer_mag_bins_from_bundle_contents(zf)
            params = None
            try:
                with zf.open("run_params.json") as f:
                    params = json.load(f)
            except Exception:
                params = None
    except Exception:
        return None

    from_params = _normalize_mag_bins(params.get("mag_bin")) if isinstance(params, dict) else None

    # Prefer inference from bundled per-mag-bin result filenames, since run_params.json can be
    # overwritten when multiple mag bins share one out_dir (concurrent runs).
    if inferred:
        return inferred
    return from_params


def _assert_mag_bin_match(expected: list[str], observed: list[str], source: str) -> None:
    if expected != observed:
        raise SystemExit(
            f"Provided --mag-bin ({observed}) does not match {source} ({expected})."
        )


def get_out_dir_from_bundle(bundle_path: Path, base_root: Path, *, overwrite: bool = False) -> Path:
    """Extract run directory name from bundle filename.

    If the derived run directory already exists:
      - return it directly when ``overwrite`` is True
      - otherwise return a ``_home``-suffixed directory for safety
    """
    bundle_name = bundle_path.stem  # e.g., "20260209_162336_bundle" -> "20260209_162336_bundle"
    base_name = bundle_name.removesuffix("_bundle")  # Always strip _bundle

    runs_dir = base_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    candidate = runs_dir / base_name
    if not candidate.exists():
        return candidate

    if overwrite:
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


def import_bundle_zip(bundle_zip: Path, out_dir: Path, *, show_progress: bool = False) -> None:
    """Extract a pipeline transfer bundle into out_dir."""
    bundle_zip = Path(bundle_zip).expanduser()
    if not bundle_zip.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_zip}")
    if not zipfile.is_zipfile(bundle_zip):
        raise ValueError(f"Bundle is not a valid zip file: {bundle_zip}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_zip, "r") as zf:
        members = zf.infolist()
        files = [m for m in members if not m.is_dir()]
        total_bytes = sum(m.file_size for m in files)

        if show_progress:
            with tqdm(total=total_bytes, desc="Import bundle", unit="B", unit_scale=True) as pbar:
                for member in members:
                    zf.extract(member, out_dir)
                    if not member.is_dir():
                        pbar.update(member.file_size)
        else:
            zf.extractall(out_dir)


def _collect_bundle_lightcurve_files(out_dir: Path, mag_bin_tag: str | None = None, include_all: bool = False) -> list[tuple[Path, str]]:
    """Collect candidate .dat2/.raw2 files to include in bundle assets.

    By default only includes light curves for candidates that passed all
    filters (failed_any=False). Pass include_all=True to bundle every
    candidate regardless of filter outcome. Source files are read directly from
    their original location and are never modified in place.
    """
    if mag_bin_tag:
        filtered_candidates = out_dir / "results" / f"lc_events_filtered_{mag_bin_tag}.parquet"
    else:
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

    if not include_all and "failed_any" in df_candidates.columns:
        n_before = len(df_candidates)
        df_candidates = df_candidates[~df_candidates["failed_any"].astype(bool)]
        print(f"Bundling light curves for {len(df_candidates)}/{n_before} passing candidates (failed_any=False)")

    files_to_bundle: list[tuple[Path, str]] = []
    seen_files: set[Path] = set()

    path_series = pd.Series(df_candidates["path"])
    for raw_path in path_series.dropna().astype(str).unique().tolist():
        dat_path = Path(raw_path).expanduser()
        # Accept any dat extension (.dat, .dat2, .dat3, etc.)
        if not dat_path.suffix.lower().startswith(".dat"):
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


def export_bundle_zip(bundle_zip: Path, out_dir: Path, include_all: bool = False, mag_bin_tag: str | None = None) -> list[str]:
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
    if mag_bin_tag:
        include_rel_paths.extend([
            f"run_params_{mag_bin_tag}.json",
            f"run_{mag_bin_tag}.log",
            f"results/lc_events_results_{mag_bin_tag}.parquet",
            f"results/lc_events_filtered_{mag_bin_tag}.parquet",
            f"results/lc_events_enriched_{mag_bin_tag}.parquet",
        ])
    if include_all:
        include_rel_paths.append("bundle_assets/asassn_index_full.parquet")
    include_globs = [
        f"results/lc_events_results_{mag_bin_tag}*" if mag_bin_tag else "results/lc_events_results*",
    ]
    include_dirs = [
        "plots",
    ]
    if include_all:
        include_dirs.extend(["manifests", "tags", "paths", "gaia_cache"])

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

    lightcurve_files = _collect_bundle_lightcurve_files(out_dir, mag_bin_tag=mag_bin_tag, include_all=include_all)

    if not files_to_add and not lightcurve_files:
        raise FileNotFoundError(f"No bundle files found under {out_dir}")

    ordered_files = sorted(files_to_add, key=lambda p: str(p.relative_to(out_dir)))
    ordered_lightcurve_files = sorted(lightcurve_files, key=lambda item: item[1])

    total_files = len(ordered_files) + len(ordered_lightcurve_files)
    print(f"Bundling {total_files} files with ZIP_DEFLATED compression...")

    bundled_paths: list[str] = []
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for p in ordered_files:
            arcname = str(p.relative_to(out_dir))
            zf.write(p, arcname=arcname)
            bundled_paths.append(arcname)
        for source_file, arcname in ordered_lightcurve_files:
            zf.write(source_file, arcname=arcname)
            bundled_paths.append(arcname)

    return bundled_paths


def _add_gaia_ids_from_index(df_events: pd.DataFrame, index_path) -> pd.DataFrame:
    """
    Merge gaia_id and asas_sn_id from the ASASSN index into the events DataFrame.

    The ASASSN index covers all ~17M ASAS-SN sources and carries a Gaia ID for
    each one, so almost every candidate should receive a gaia_id after this merge.
    The VSX crossmatch only covers ~99K known variables and must not be used here.

    Parameters
    ----------
    df_events : pd.DataFrame
        Events DataFrame from events.py (must have 'path' column).
    index_path : Path or str
        Path to the ASASSN index parquet (or CSV) file.

    Returns
    -------
    pd.DataFrame
        Events DataFrame with gaia_id and asas_sn_id columns added
        (NaN for the rare unmatched sources).
    """
    if "path" not in df_events.columns:
        _log("Warning: Cannot add gaia_id - 'path' column not found")
        return df_events

    if not Path(index_path).exists():
        _log(f"Warning: ASASSN index not found at {index_path}")
        return df_events

    try:
        df = df_events.copy()

        # Derive asas_sn_id from the LC filename stem (e.g. "498216332934.dat3" → 498216332934)
        def _extract_id(path_str):
            if pd.isna(path_str):
                return None
            try:
                return int(Path(str(path_str)).stem.split(".")[0])
            except Exception:
                return None

        df["asas_sn_id"] = df["path"].apply(_extract_id)

        # Load only the columns we need from the index
        index_path = Path(index_path)
        _log(f"Loading ASASSN index from {index_path.name}...")
        if index_path.suffix in (".parquet", ".pq"):
            df_index = pd.read_parquet(index_path, columns=["asas_sn_id", "gaia_id"])
        else:
            df_index = pd.read_csv(index_path, usecols=["asas_sn_id", "gaia_id"], low_memory=False)

        df_index["asas_sn_id"] = pd.to_numeric(df_index["asas_sn_id"], errors="coerce")
        df_index = df_index.dropna(subset=["asas_sn_id"])
        df_index["asas_sn_id"] = df_index["asas_sn_id"].astype("int64")

        df_merged = df.merge(
            df_index[["asas_sn_id", "gaia_id"]].drop_duplicates(subset=["asas_sn_id"]),
            on="asas_sn_id",
            how="left",
        )

        n_with_gaia = df_merged["gaia_id"].notna().sum()
        n_total = len(df_merged)
        pct = 100.0 * n_with_gaia / n_total if n_total > 0 else 0.0
        _log(f"[gaia_id merge] Added gaia_id for {n_with_gaia}/{n_total} events ({pct:.2f}%)")

        return df_merged
    except Exception as e:
        _log(f"Warning: Failed to merge gaia_id from index: {e}")
        return df_events


def _build_filter_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build apply_filters kwargs from detect CLI arguments."""
    return {
        # Core filters
        "apply_evidence_strength": not args.skip_evidence_strength,
        "min_bayes_factor": args.min_bayes_factor,
        "require_finite_local_bf": not args.allow_infinite_local_bf,
        "apply_significant_detection": not args.skip_significant_detection,
        "significant_require_flag": not args.significant_no_require_flag,
        "significant_min_peak_count": args.significant_min_peak_count,
        "significant_min_run_count": args.significant_min_run_count,
        "apply_run_robustness": not args.skip_run_robustness,
        "min_run_count": args.min_run_count,
        "max_run_count": args.max_run_count,
        "min_run_points": args.filter_min_run_points,
        "min_run_cameras": args.filter_min_run_cameras,
        # Optional filters
        "apply_morphology": args.apply_morphology,
        "dip_morphology": args.dip_morphology,
        "jump_morphology": args.jump_morphology,
        "min_delta_bic": args.min_delta_bic,
        "apply_score": not args.skip_score_filter,
        "min_dip_score": args.min_dip_score,
        "min_jump_score": args.min_jump_score,
        "min_score": args.min_score,
        # Validation filters
        "apply_periodicity_validation": args.apply_periodicity_validation,
        "periodicity_n_bootstrap": args.periodicity_n_bootstrap,
        "periodicity_significance": args.periodicity_significance,
        "periodicity_pdm_method": args.periodicity_pdm_method,
        "periodicity_exclude_aliases": not args.periodicity_no_exclude_aliases,
        "periodicity_flag_only": not args.periodicity_reject,
        "periodicity_workers": args.periodicity_workers,
        "periodicity_checkpoint_dir": args.periodicity_checkpoint_dir,
        "periodicity_all_candidates": args.periodicity_all_candidates,
        "phase_plot_max_sig": args.phase_plot_max_sig,
        "phase_plot_min_power": args.phase_plot_min_power,
        "phase_plot_allow_alias": args.phase_plot_allow_alias,
        "apply_gaia_ruwe_validation": not args.skip_gaia_ruwe_validation,
        "gaia_max_ruwe": args.gaia_max_ruwe,
        "gaia_flag_only": not args.gaia_reject,
        "apply_gaia_pm_validation": not args.skip_gaia_pm_validation,
        "gaia_max_pm": args.gaia_max_pm,
        "gaia_pm_flag_only": not args.gaia_pm_reject,
        "apply_periodic_catalog_validation": not args.skip_periodic_catalog_validation,
        "periodic_catalog_max_sep": args.periodic_catalog_max_sep,
        "periodic_catalog_flag_only": not args.periodic_catalog_reject,
        # Progress/logging
        "show_tqdm": args.verbose,
        "verbose": args.verbose,
    }


def _build_home_external_validation_cmd(
    args: argparse.Namespace,
    *,
    post_filter_output: Path,
    index_file: Path,
) -> list[str]:
    """Build the home-stage external validation subprocess command."""
    cmd = [
        sys.executable,
        "-m",
        "malca.filter",
        "--input",
        str(post_filter_output),
        "--output",
        str(post_filter_output),
        "--index-file",
        str(index_file),
        "--home-passers-only",
        "--skip-evidence-strength",
        "--skip-significant-detection",
        "--skip-run-robustness",
        "--gaia-max-ruwe",
        str(args.gaia_max_ruwe),
        "--gaia-max-pm",
        str(args.gaia_max_pm),
        "--periodic-catalog-max-sep",
        str(args.periodic_catalog_max_sep),
    ]

    if args.apply_periodicity_validation:
        cmd.extend(
            [
                "--apply-periodicity-validation",
                "--periodicity-n-bootstrap",
                str(args.periodicity_n_bootstrap),
                "--periodicity-significance",
                str(args.periodicity_significance),
                "--periodicity-pdm-method",
                str(args.periodicity_pdm_method),
                "--workers",
                str(args.periodicity_workers),
                "--phase-plot-max-sig",
                str(args.phase_plot_max_sig),
                "--phase-plot-min-power",
                str(args.phase_plot_min_power),
            ]
        )
        if args.periodicity_no_exclude_aliases:
            cmd.append("--periodicity-no-exclude-aliases")
        if args.periodicity_reject:
            cmd.append("--periodicity-reject")
        if args.periodicity_all_candidates:
            cmd.append("--periodicity-all-candidates")
        if args.phase_plot_allow_alias:
            cmd.append("--phase-plot-allow-alias")
        if args.periodicity_checkpoint_dir:
            cmd.extend(["--checkpoint-dir", str(args.periodicity_checkpoint_dir)])

    if args.gaia_reject:
        cmd.append("--gaia-reject")
    if args.gaia_pm_reject:
        cmd.append("--gaia-pm-reject")
    if args.periodic_catalog_reject:
        cmd.append("--periodic-catalog-reject")
    if args.skip_gaia_ruwe_validation:
        cmd.append("--skip-gaia-ruwe-validation")
    if args.skip_gaia_pm_validation:
        cmd.append("--skip-gaia-pm-validation")
    if args.skip_periodic_catalog_validation:
        cmd.append("--skip-periodic-catalog-validation")
    if not args.verbose:
        cmd.append("--no-tqdm")
    if args.verbose:
        cmd.append("--verbose")
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Run events.py on tagged light curves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="All other arguments are passed directly to events.py",
    )
    g_manifest = parser.add_argument_group("Manifest & index")
    g_tag = parser.add_argument_group("Tag")
    g_pregate = parser.add_argument_group("Pre-periodicity gate")
    g_events = parser.add_argument_group("Event detection")
    g_output = parser.add_argument_group("Output & bundle")
    g_filter = parser.add_argument_group("Filter")
    g_postprocess = parser.add_argument_group("Postprocess")
    g_characterize = parser.add_argument_group("Characterize")
    g_classify = parser.add_argument_group("Classify")
    g_enrich = parser.add_argument_group("Enrich")
    g_neighbor = parser.add_argument_group("Neighbor enrichment")
    g_spectra = parser.add_argument_group("Spectra enrichment")
    g_vetting = parser.add_argument_group("Vetting")
    g_general = parser.add_argument_group("General")

    g_manifest.add_argument("--mag-bin", nargs="+", help="Magnitude bin(s) to process. Use 'all' to process all bins automatically.")
    g_manifest.add_argument("--index-root", type=Path, default=LCV2_ROOT,
                        help="Index root directory (contains mag_bin/index*.csv)")
    g_manifest.add_argument("--lc-root", type=Path, default=LCV2_ROOT,
                        help="Light curve root directory (contains mag_bin/lc*_cal/)")
    g_manifest.add_argument("--flat-lc-dir", type=Path, default=None,
                        help="Flat directory of <source_id>.<extension> light curves, such as bundle_assets/lightcurves")
    g_manifest.add_argument("--index-file", type=Path, default=None,
                        help="Optional ASAS-SN index/metadata file. Used by home external validation, --full-bundle export, and flat light-curve manifests")
    g_manifest.add_argument("--manifest-file", type=Path, default=None,
                        help="Manifest file (default: lc_manifest_{mag_bin}.parquet)")
    g_manifest.add_argument("--filtered-file", type=Path, default=None,
                        help="Filtered manifest file (default: lc_filtered_{mag_bin}.parquet)")
    g_manifest.add_argument("--force-manifest", action="store_true",
                        help="Force rebuild manifest even if exists")
    g_manifest.add_argument("--force-tag", action="store_true",
                        help="Force re-run tagging even if tagged file exists")
    g_manifest.add_argument("--extension", "-e", type=str, default=None,
                        help="Light curve file extension (e.g., dat, dat2, dat3). Default: dat3 (from config)")

    g_tag.add_argument("--min-time-span", type=float, default=MIN_TIME_SPAN, help="Min time span (days)")
    g_tag.add_argument("--min-points-per-day", type=float, default=MIN_POINTS_PER_DAY, help="Min cadence")
    g_tag.add_argument("--min-cameras", type=int, default=MIN_CAMERAS, help="Min cameras required")
    g_tag.add_argument("--mag-lo", type=float, default=10.0, help="Min baseline magnitude for mag range filter (default: 10)")
    g_tag.add_argument("--mag-hi", type=float, default=18.0, help="Max baseline magnitude for mag range filter (default: 18)")
    g_tag.add_argument("--skip-sparse", action="store_true", help="Skip sparse LC filter")
    g_tag.add_argument("--skip-multi-camera", action="store_true", help="Skip multi-camera filter")
    g_tag.add_argument("--skip-mag-range", action="store_true", help="Skip magnitude range filter")
    g_tag.add_argument("--skip-vsx", action="store_true", help="Skip VSX crossmatch/tagging")
    g_tag.add_argument("--skip-camera-median", action="store_true", help="Skip camera median filter (identifies cameras to exclude from .raw2 files)")
    g_tag.add_argument("--camera-median-tolerance", type=float, default=CAMERA_MEDIAN_TOLERANCE, help="Tolerance beyond mag bin for camera median filter (default: 0.2 mag)")
    g_tag.add_argument("--vsx-max-sep", type=float, default=VSX_MAX_SEP_ARCSEC, help="Max separation for VSX match (arcsec)")
    g_tag.add_argument(
        "--vsx-mode",
        type=str,
        default=VSX_MODE,
        choices=["tag"],
        help="VSX handling mode. Only 'tag' is supported.",
    )
    g_tag.add_argument("--vsx-crossmatch", type=Path, default=VSX_CROSSMATCH_PATH, help="Path to pre-crossmatched VSX CSV (with asas_sn_id, vsx_sep_arcsec, vsx_class)")
    g_tag.add_argument("--pass-all-tags", action="store_true", help="Pass all light curves to events.py regardless of tag results (failure tags are still added)")
    g_tag.add_argument("--enforce-tags", type=str, default=None, help="Comma-separated list of tag checks to enforce (e.g., 'sparse,multi_camera'). " "Only rows failing these checks are excluded. Default: enforce all enabled checks.")
    g_tag.add_argument("--workers", type=int, default=WORKERS, help="Workers for parallel processing")
    g_tag.add_argument("--stats-chunk-size", type=int, default=STATS_CHUNK_SIZE, help="Rows per checkpoint save during stats computation")
    g_tag.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Max light curves per events.py call")

    g_pregate.add_argument("--apply-pre-periodicity-gate", action="store_true", help="Run the CE-only periodicity gate before events.py and split confident periodic candidates out of the stochastic branch")
    g_pregate.add_argument("--pre-periodicity-min-period", type=float, default=PRE_PERIODICITY_MIN_PERIOD, help="Minimum trial period in days for the pre-events gate")
    g_pregate.add_argument("--pre-periodicity-max-period", type=float, default=PRE_PERIODICITY_MAX_PERIOD, help="Maximum trial period in days for the pre-events gate")
    g_pregate.add_argument("--pre-periodicity-n-periods", type=int, default=PRE_PERIODICITY_N_PERIODS, help="Number of CE trial periods for the pre-events gate")
    g_pregate.add_argument("--pre-periodicity-ce-snr-threshold", type=float, default=PRE_PERIODICITY_CE_SNR_THRESHOLD, help="Minimum CE SNR for periodic routing")
    g_pregate.add_argument("--pre-periodicity-min-points", type=int, default=PRE_PERIODICITY_MIN_POINTS, help="Minimum cleaned LC points required for the pre-events gate")
    g_pregate.add_argument("--pre-periodicity-scatter-ratio-max", type=float, default=PRE_PERIODICITY_SCATTER_RATIO_MAX, help="Maximum folded/raw scatter ratio for confident periodic routing")
    g_pregate.add_argument("--pre-periodicity-checkpoint", type=Path, default=None, help="Checkpoint parquet path for the pre-events periodicity gate")
    g_pregate.add_argument("--pre-periodicity-workers", type=int, default=WORKERS, help="Workers for the pre-events periodicity gate")

    g_events.add_argument("--trigger-mode", type=str, default=TRIGGER_MODE, choices=["logbf", "posterior_prob"], help="Triggering mode")
    g_events.add_argument("--logbf-threshold-dip", type=float, default=LOGBF_THRESHOLD_DIP, help="Per-point dip trigger threshold")
    g_events.add_argument("--logbf-threshold-jump", type=float, default=LOGBF_THRESHOLD_JUMP, help="Per-point jump trigger threshold")
    g_events.add_argument("--significance-threshold", type=float, default=SIGNIFICANCE_THRESHOLD, help="Posterior probability threshold (if trigger-mode=posterior_prob)")
    g_events.add_argument("--p-points", type=int, default=P_POINTS, help="Number of points in the p grid")
    g_events.add_argument("--mag-points", type=int, default=MAG_POINTS, help="Number of points in the magnitude grid")
    g_events.add_argument("--run-min-points", type=int, default=RUN_MIN_POINTS, help="Min triggered points in a run")
    g_events.add_argument("--run-max-gap-points", type=int, default=RUN_MAX_GAP_POINTS, help="Allow up to this many missing indices inside a run")
    g_events.add_argument("--run-max-gap-days", type=float, default=None, help="Break runs if JD gap exceeds this")
    g_events.add_argument("--run-min-duration-days", type=float, default=0.0, help="Require run duration >= this (default: 0.0 = disabled)")
    g_events.add_argument("--no-event-prob", action="store_true", help="Skip LOO event responsibilities")
    g_events.add_argument("--p-min-dip", type=float, default=None, help="Minimum dip fraction for p-grid")
    g_events.add_argument("--p-max-dip", type=float, default=None, help="Maximum dip fraction for p-grid")
    g_events.add_argument("--p-min-jump", type=float, default=None, help="Minimum jump fraction for p-grid")
    g_events.add_argument("--p-max-jump", type=float, default=None, help="Maximum jump fraction for p-grid")
    g_events.add_argument(
        "--baseline-func",
        type=str,
        default=BASELINE_FUNC,
        choices=["gp", "gp_masked", "global_median", "per_camera_median", "phase_template"],
        help="Baseline function",
    )
    g_events.add_argument("--baseline-s0", type=float, default=BASELINE_S0, help="GP kernel S0 parameter (default: 0.0005)")
    g_events.add_argument("--baseline-w0", type=float, default=BASELINE_W0, help="GP kernel w0 parameter (default: pi/1000)")
    g_events.add_argument("--baseline-q", type=float, default=BASELINE_Q, help="GP kernel Q parameter (default: 0.7)")
    g_events.add_argument("--baseline-jitter", type=float, default=BASELINE_JITTER, help="GP jitter term (default: 0.006)")
    g_events.add_argument("--baseline-sigma-floor", type=float, default=None, help="Minimum sigma floor (default: None)")
    g_events.add_argument("--mag-min-dip", type=float, default=None, help="Min magnitude for dip grid (overrides auto)")
    g_events.add_argument("--mag-max-dip", type=float, default=None, help="Max magnitude for dip grid (overrides auto)")
    g_events.add_argument("--mag-min-jump", type=float, default=None, help="Min magnitude for jump grid (overrides auto)")
    g_events.add_argument("--mag-max-jump", type=float, default=None, help="Max magnitude for jump grid (overrides auto)")
    g_events.add_argument("--min-mag-offset", type=float, default=MIN_MAG_OFFSET, help="Require |event_mag - baseline_mag| > threshold")
    g_output.add_argument("--output", type=str, default=None, help="Output path for results (default: <out_dir>/lc_events_results.parquet)")
    g_output.add_argument("--out-dir", type=str, default=None, help="Directory for all outputs (default: output/runs/<timestamp>)")
    g_output.add_argument("--output-format", type=str, default=OUTPUT_FORMAT, choices=["csv", "parquet", "parquet_chunk"], help="Output format")
    g_output.add_argument("--chunk-size", type=int, default=EVENTS_OUTPUT_CHUNK_SIZE, help="Write results in chunks of this many rows")
    g_output.add_argument(
        "--stage",
        type=str,
        default="full",
        choices=["full", "cluster", "home"],
        help="Pipeline stage: full=all steps, cluster=raw-dependent upstream, home=downstream only",
    )
    g_output.add_argument("--import-bundle", type=Path, default=None, help="Zip bundle produced by --export-bundle (for home stage)")
    g_output.add_argument("--export-bundle", type=Path, default=None, help="Write transferable zip bundle at end of run")
    g_output.add_argument("--no-export-bundle", dest="export_bundle_enabled", action="store_false",
                        help="Skip export bundle creation at end of run")
    g_output.add_argument("--full-bundle", action="store_true", default=False, help="Include all large assets in export bundle (index, gaia cache, manifests, tags, paths)")
    g_output.add_argument("--no-review-sync", dest="review_sync_enabled", action="store_false",
                        help="Skip automatic reviews/*.jsonl export after review DB import")
    g_output.add_argument("--review-sync-dir", type=Path, default=Path("reviews"),
                        help="Directory for automatic Git-trackable review export (default: reviews)")
    g_output.add_argument("--review-sync-hash-assets", action="store_true",
                        help="Include SHA-256 hashes for resolved assets in automatic review export")

    g_filter.add_argument("--run-filter", dest="run_filter", action="store_true", help="Run filter after events.py completes (default: enabled)")
    g_filter.add_argument("--no-run-filter", dest="run_filter", action="store_false", help="Skip filter step")
    g_filter.add_argument("--skip-evidence-strength", action="store_true", help="Skip evidence-strength filter")
    g_filter.add_argument("--min-bayes-factor", type=float, default=MIN_BAYES_FACTOR, help="Min Bayes factor for filter stage (default: 10.0)")
    g_filter.add_argument("--allow-infinite-local-bf", action="store_true", help="Allow infinite local Bayes factors (default: require finite)")
    g_filter.add_argument("--skip-significant-detection", action="store_true", help="Skip explicit significant run/peak gate")
    g_filter.add_argument("--significant-no-require-flag", action="store_true", help="Do not require dip/jump significant flags in significant detection gate")
    g_filter.add_argument("--significant-min-peak-count", type=int, default=1, help="Minimum dip_count/jump_count for significant detection gate (default: 1)")
    g_filter.add_argument("--significant-min-run-count", type=int, default=1, help="Minimum dip_run_count/jump_run_count for significant detection gate (default: 1)")
    g_filter.add_argument("--skip-run-robustness", action="store_true", help="Skip run-robustness filter")
    g_filter.add_argument("--min-run-count", type=int, default=1, help="Minimum run count for run-robustness filter (default: 1)")
    g_filter.add_argument("--max-run-count", type=int, default=None, help="Maximum run count for run-robustness filter (default: disabled)")
    g_filter.add_argument("--filter-min-run-cameras", dest="filter_min_run_cameras", type=int, default=POST_FILTER_MIN_RUN_CAMERAS, help="Min cameras for run robustness filter (default: 2)")
    g_filter.add_argument("--filter-min-run-points", dest="filter_min_run_points", type=int, default=POST_FILTER_MIN_RUN_POINTS, help="Min points per run for robustness filter (default: 2)")
    g_filter.add_argument("--apply-morphology", action="store_true", help="Apply morphology filter in filter stage")
    g_filter.add_argument("--dip-morphology", type=str, default="gaussian", choices=["gaussian", "paczynski"], help="Required morphology for dip events (default: gaussian)")
    g_filter.add_argument("--jump-morphology", type=str, default="paczynski", choices=["gaussian", "paczynski"], help="Required morphology for jump events (default: paczynski)")
    g_filter.add_argument("--min-delta-bic", type=float, default=10.0, help="Minimum delta BIC for morphology filter (default: 10.0)")
    g_filter.add_argument("--skip-score-filter", action="store_true", help="Skip score filter (enabled by default)")
    g_filter.add_argument("--min-score", type=float, default=0.0, help="Legacy minimum score threshold applied to dipper/jumper score filters (default: 0.0)")
    g_filter.add_argument("--min-dip-score", type=float, default=None, help="Minimum dipper_score threshold (overrides --min-score for dips)")
    g_filter.add_argument("--min-jump-score", type=float, default=None, help="Minimum jumper_score threshold (overrides --min-score for jumps)")
    g_filter.add_argument("--apply-periodicity-validation", action="store_true", help="Enable bootstrap periodicity validation")
    g_filter.add_argument("--periodicity-n-bootstrap", type=int, default=1000, help="Bootstrap iterations for periodicity validation (default: 1000)")
    g_filter.add_argument("--periodicity-significance", type=float, default=0.01, help="Significance threshold for periodicity validation (default: 0.01)")
    g_filter.add_argument("--periodicity-pdm-method", type=str, default=POST_FILTER_PDM_METHOD, choices=list(PDM_METHOD_CHOICES), help="PDM implementation for periodicity validation")
    g_filter.add_argument("--periodicity-no-exclude-aliases", action="store_true", help="Do not exclude alias periods during periodicity validation")
    g_filter.add_argument("--periodicity-reject", action="store_true", help="Reject periodicity matches instead of flagging only")
    g_filter.add_argument("--periodicity-all-candidates", action="store_true", help="Run periodicity validation on all queued candidates instead of only prerequisite passers")
    g_filter.add_argument("--periodicity-workers", type=int, default=WORKERS, help="Workers for periodicity validation (default: WORKERS)")
    g_filter.add_argument("--periodicity-checkpoint-dir", type=Path, default=None, help="Checkpoint directory for periodicity validation")
    g_filter.add_argument("--phase-plot-max-sig", type=float, default=0.01, help="Require lsp_bootstrap_sig <= this for phase plotting (default: 0.01)")
    g_filter.add_argument("--phase-plot-min-power", type=float, default=0.3, help="Require lsp_power >= this for phase plotting (default: 0.3)")
    g_filter.add_argument("--phase-plot-allow-alias", action="store_true", help="Allow alias periods for phase plotting")
    g_filter.add_argument("--skip-gaia-ruwe-validation", action="store_true", help="Skip Gaia RUWE validation")
    g_filter.add_argument("--gaia-max-ruwe", type=float, default=1.4, help="Maximum RUWE threshold (default: 1.4)")
    g_filter.add_argument("--gaia-reject", action="store_true", help="Reject high-RUWE sources instead of flagging only")
    g_filter.add_argument("--skip-gaia-pm-validation", action="store_true", help="Skip Gaia proper-motion validation")
    g_filter.add_argument("--gaia-max-pm", type=float, default=100.0, help="Maximum proper-motion threshold in mas/yr (default: 100.0)")
    g_filter.add_argument("--gaia-pm-reject", action="store_true", help="Reject high proper-motion sources instead of flagging only")
    g_filter.add_argument("--skip-periodic-catalog-validation", action="store_true", help="Skip periodic-catalog crossmatch validation")
    g_filter.add_argument("--periodic-catalog-max-sep", type=float, default=3.0, help="Maximum separation for periodic-catalog matching in arcsec (default: 3.0)")
    g_filter.add_argument("--periodic-catalog-reject", action="store_true", help="Reject periodic-catalog matches instead of flagging only")

    g_postprocess.add_argument("--run-postprocess", dest="run_postprocess", action="store_true", help="Run postprocess (generate plots) after filtering")
    g_postprocess.add_argument("--no-run-postprocess", dest="run_postprocess", action="store_false", help="Skip postprocess step (default)")
    g_postprocess.add_argument("--max-plots", type=int, default=None, help="Limit number of plots generated (default: no limit)")
    g_postprocess.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf"], help="Output format for plots (default: png)")

    g_characterize.add_argument("--run-characterize", dest="run_characterize", action="store_true", help="Run Gaia DR3 characterization after filtering (default: enabled)")
    g_characterize.add_argument("--no-run-characterize", dest="run_characterize", action="store_false", help="Skip characterization step")
    g_characterize.add_argument("--gaia-cache", type=Path, default=None, help="Path to Gaia query cache file (parquet). Default: <out_dir>/gaia_cache/gaia_cache.parquet")
    g_characterize.add_argument("--gaia-fetch-chunk-size", type=int, default=GAIA_CHUNK_SIZE, help="Gaia fetch chunk size for pre-characterization local catalog sync (default: 1000)")
    g_characterize.add_argument("--characterize-crossmatch", type=Path, default=VSX_CROSSMATCH_PATH, help="ASAS-SN x VSX crossmatch file for characterize step")
    g_characterize.add_argument("--characterize-chunk-size", type=int, default=GAIA_CHUNK_SIZE, help="Gaia query chunk size for characterize step")
    g_characterize.add_argument("--characterize-starhorse", type=str, default="tap", help="StarHorse mode/path for characterize step (default: tap)")
    g_characterize.add_argument("--characterize-starhorse-cache", type=Path, default=None, help="Optional StarHorse TAP cache parquet path (default: output/cache/catalogs/starhorse_tap_cache.parquet)")
    g_characterize.add_argument("--characterize-unwise-checkpoint-every", type=int, default=UNWISE_CHECKPOINT_EVERY, help="Persist unWISE checkpoint every N completed candidates")
    g_characterize.add_argument("--no-characterize-banyan", dest="characterize_banyan", action="store_false", help="Disable BANYAN Sigma enrichment in characterize step")
    g_characterize.add_argument("--no-characterize-iphas", dest="characterize_iphas", action="store_false", help="Disable IPHAS enrichment in characterize step")
    g_characterize.add_argument("--no-characterize-sfr", dest="characterize_sfr", action="store_false", help="Disable star-forming-region enrichment in characterize step")
    g_characterize.add_argument("--no-characterize-clusters", dest="characterize_clusters", action="store_false", help="Disable open-cluster enrichment in characterize step")
    g_characterize.add_argument("--characterize-unwise", dest="characterize_unwise", action="store_true", help="Enable unWISE/unTimely variability enrichment in characterize step (default: disabled)")
    g_characterize.add_argument("--no-characterize-unwise", dest="characterize_unwise", action="store_false", help="Disable unWISE enrichment in characterize step")
    g_characterize.add_argument("--run-dust", dest="run_dust", action="store_true", help="Run 3D dust extinction correction (default: enabled)")
    g_characterize.add_argument("--no-run-dust", dest="run_dust", action="store_false", help="Skip dust extinction step")

    g_classify.add_argument("--run-classify", dest="run_classify", action="store_true", help="Run classification (EB/CV/starspot rejection, YSO) (default: enabled)")
    g_classify.add_argument("--no-run-classify", dest="run_classify", action="store_false", help="Skip classification step")

    g_enrich.add_argument("--run-enrich", dest="run_enrich", action="store_true", help="Enrich passing candidates with comprehensive light curve stats (default: enabled)")
    g_enrich.add_argument("--no-run-enrich", dest="run_enrich", action="store_false", help="Skip enrichment step")
    g_enrich.add_argument("--enrich-compute-ls", action="store_true", help="Include Lomb-Scargle periodogram in enrichment (expensive)")

    g_neighbor.add_argument("--run-neighbor-enrich", dest="run_neighbor_enrich", action="store_true", help="Bulk neighbor enrichment for passing candidates (default: enabled)")
    g_neighbor.add_argument("--no-run-neighbor-enrich", dest="run_neighbor_enrich", action="store_false", help="Skip neighbor enrichment step")
    g_neighbor.add_argument("--neighbor-radius-arcsec", type=float, default=NEIGHBOR_RADIUS_ARCSEC, help="Neighbor search radius in arcsec (default: 15)")
    g_neighbor.add_argument("--neighbor-chunk-size", type=int, default=NEIGHBOR_CHUNK_SIZE, help="Bulk chunk size for neighbor lookups")
    g_neighbor.add_argument("--neighbor-cache", type=Path, default=None, help="Optional cache parquet path for neighbor lookups")

    g_spectra.add_argument("--run-spectra-enrich", dest="run_spectra_enrich", action="store_true", help="Bulk spectra-availability enrichment for passing candidates (default: enabled)")
    g_spectra.add_argument("--no-run-spectra-enrich", dest="run_spectra_enrich", action="store_false", help="Skip spectra enrichment step")
    g_spectra.add_argument("--spectra-radius-arcsec", type=float, default=SPECTRA_RADIUS_ARCSEC, help="Spectra crossmatch radius in arcsec (default: 3)")
    g_spectra.add_argument("--spectra-chunk-size", type=int, default=SPECTRA_CHUNK_SIZE, help="Bulk chunk size for spectra lookups")
    g_spectra.add_argument("--spectra-cache", type=Path, default=None, help="Optional cache parquet path for spectra lookups")

    g_vetting.add_argument("--run-vetting", dest="run_vetting", action="store_true", help="Run post-review vetting (SIMBAD, Gaia variability, ASAS-SN variables) (default: enabled)")
    g_vetting.add_argument("--no-run-vetting", dest="run_vetting", action="store_false", help="Skip vetting step")
    g_vetting.add_argument("--vetting-min-score", type=float, default=None, help="Only vet candidates with interest_score >= this value")
    g_vetting.add_argument("--vetting-simbad-radius", type=float, default=5.0, help="SIMBAD search radius in arcsec (default: 5)")
    g_vetting.add_argument("--vetting-asassn-radius", type=float, default=5.0, help="ASAS-SN crossmatch radius in arcsec (default: 5)")
    g_vetting.add_argument("--no-vetting-simbad", action="store_true", help="Skip SIMBAD query in vetting")
    g_vetting.add_argument("--no-vetting-gaia-var", action="store_true", help="Skip Gaia DR3 variability query in vetting")
    g_vetting.add_argument("--no-vetting-gaia-epoch", action="store_true", help="Skip Gaia epoch photometry check in vetting")
    g_vetting.add_argument("--no-vetting-asassn-var", action="store_true", help="Skip ASAS-SN variable catalog crossmatch in vetting")
    g_vetting.add_argument("--no-vetting-alerce", action="store_true", help="Skip ALeRCE ZTF query in vetting")
    g_vetting.add_argument("--no-vetting-erosita", action="store_true", help="Skip eROSITA X-ray crossmatch in vetting")
    g_vetting.add_argument("--no-vetting-pm-check", action="store_true", help="Skip proper motion consistency check in vetting")
    g_vetting.add_argument("--vetting-atlas", dest="vetting_atlas", action="store_true", help="Run ATLAS forced photometry in vetting (default: disabled)")
    g_vetting.add_argument("--no-vetting-atlas", dest="vetting_atlas", action="store_false", help="Skip ATLAS forced photometry in vetting")
    g_vetting.add_argument("--vetting-atlas-token", type=str, default=None, help="ATLAS forced photometry API token")
    g_vetting.add_argument("--vetting-neowise-lc", dest="vetting_neowise_lc", action="store_true", help="Fetch full NEOWISE light curves in vetting (default: disabled)")
    g_vetting.add_argument("--no-vetting-neowise-lc", dest="vetting_neowise_lc", action="store_false", help="Skip full NEOWISE light curves in vetting")
    g_vetting.add_argument("--vetting-input", type=Path, default=None, help="Explicit input file for vetting (default: latest enriched/characterized output)")

    g_general.add_argument("--test-run", action="store_true", help="Limit the number of light curves processed (for quick end-to-end validation)")
    g_general.add_argument("--test-run-n", type=int, default=10000, help="Number of light curves to sample in test-run mode (default: 10000)")
    g_general.add_argument("-o", "--overwrite", action="store_true", help="Overwrite checkpoint log and existing output if present (start fresh).")
    g_general.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    parser.set_defaults(
        run_filter=True,
        run_postprocess=False,
        run_characterize=True,
        run_dust=True,
        characterize_banyan=True,
        characterize_iphas=True,
        characterize_sfr=True,
        characterize_clusters=True,
        characterize_unwise=False,
        run_classify=True,
        run_enrich=True,
        run_neighbor_enrich=True,
        run_spectra_enrich=True,
        run_vetting=True,
        vetting_atlas=False,
        vetting_neowise_lc=False,
        export_bundle_enabled=True,
        review_sync_enabled=True,
    )

    args = parser.parse_args()

    # Handle --mag-bin all: expand to all bins in reverse order
    is_auto_all_mode = False
    if args.mag_bin and "all" in args.mag_bin:
        if len(args.mag_bin) > 1:
            parser.error("Cannot mix 'all' with specific magnitude bins. Use '--mag-bin all' alone or specify individual bins.")
        
        # Expand "all" to full list of magnitude bins in reverse order
        is_auto_all_mode = True
        args.mag_bin = list(reversed(MAG_BINS))

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
    if stage == "home" and (args.force_manifest or args.force_tag):
        print("Info: --stage home skips manifest/tag/events regardless of force flags.")

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
    mag_bin_tag = "all" if is_auto_all_mode else (args.mag_bin[0] if len(args.mag_bin) == 1 else "multi")

    # IMPORTANT: never write to filesystem root (/output). Default to a writable directory.
    events_format = str(args.output_format).lower()
    base_output_root = Path("output").resolve()
    if args.out_dir is not None:
        out_dir = Path(args.out_dir).expanduser()
    elif args.import_bundle is not None:
        # Auto-derive out_dir from bundle name
        bundle_path = Path(args.import_bundle).expanduser()
        out_dir = get_out_dir_from_bundle(bundle_path, base_output_root, overwrite=bool(args.overwrite))
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
    if args.import_bundle is not None and args.overwrite and out_dir.exists():
        log(f"Overwriting existing imported run directory: {out_dir}")
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        else:
            out_dir.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.import_bundle is not None:
        import_started = time.perf_counter()
        log(f"Importing bundle from {Path(args.import_bundle).expanduser()} to {out_dir}...")
        import_bundle_zip(args.import_bundle, out_dir, show_progress=args.verbose)
        log(f"Bundle import completed in {time.perf_counter() - import_started:.1f}s")
        imported_flat_lc_dir = out_dir / "bundle_assets" / "lightcurves"
        if args.flat_lc_dir is None and imported_flat_lc_dir.is_dir():
            args.flat_lc_dir = imported_flat_lc_dir
            log(f"Using imported flat light-curve directory: {args.flat_lc_dir}")

    if args.gaia_cache is None:
        gaia_cache_dir = out_dir / "gaia_cache"
        gaia_cache_dir.mkdir(parents=True, exist_ok=True)
        args.gaia_cache = gaia_cache_dir / "gaia_cache.parquet"

    manifests_dir = out_dir / "manifests"
    tags_dir = out_dir / "tags"
    paths_dir = out_dir / "paths"
    results_dir = out_dir / "results"
    for d in (manifests_dir, tags_dir, paths_dir, results_dir):
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
        events_output = results_dir / f"lc_events_results_{mag_bin_tag}.parquet"
    else:
        events_output = Path(args.output).expanduser()
        if args.out_dir is not None and not events_output.is_absolute():
            events_output = out_dir / events_output
        elif args.out_dir is None and not events_output.is_absolute():
            events_output = out_dir / events_output

    manifest_file = Path(args.manifest_file).expanduser() if args.manifest_file else (manifests_dir / f"lc_manifest_{mag_bin_tag}.parquet")
    filtered_file = Path(args.filtered_file).expanduser() if args.filtered_file else (tags_dir / f"lc_filtered_{mag_bin_tag}.parquet")
    stats_checkpoint_file = tags_dir / f"lc_stats_checkpoint_{mag_bin_tag}.parquet"
    pre_periodicity_file = tags_dir / f"pre_periodicity_{mag_bin_tag}.parquet"
    periodic_branch_file = manifests_dir / f"lc_periodic_branch_{mag_bin_tag}.parquet"
    stochastic_branch_file = manifests_dir / f"lc_stochastic_branch_{mag_bin_tag}.parquet"
    periodic_paths_file = paths_dir / f"periodic_paths_{mag_bin_tag}.txt"
    branch_cache_dir = results_dir / "_branch_events"
    branch_cache_dir.mkdir(parents=True, exist_ok=True)
    periodic_branch_events_output = branch_cache_dir / f"lc_events_periodic_branch_{mag_bin_tag}.parquet"
    stochastic_branch_events_output = branch_cache_dir / f"lc_events_stochastic_branch_{mag_bin_tag}.parquet"
    if args.pre_periodicity_checkpoint is None:
        pre_periodicity_checkpoint = tags_dir / f"pre_periodicity_checkpoint_{mag_bin_tag}.parquet"
    else:
        pre_periodicity_checkpoint = Path(args.pre_periodicity_checkpoint).expanduser()

    # Save run parameters to JSON for full reproducibility
    run_start_time = datetime.now()

    # Build a compact fingerprint of filtering/characterization behavior.
    if args.pass_all_tags:
        enforced_tags = []
    elif args.enforce_tags:
        enforced_tags = [f.strip() for f in args.enforce_tags.split(",") if f.strip()]
    else:
        enforced_tags = []
        if not args.skip_sparse:
            enforced_tags.append("sparse")
        if not args.skip_multi_camera:
            enforced_tags.append("multi_camera")
        if not args.skip_mag_range:
            enforced_tags.append("mag_range")

    config_fingerprint = {
        "vsx_mode": args.vsx_mode,
        "skip_vsx": args.skip_vsx,
        "pass_all_tags": args.pass_all_tags,
        "enforced_tags": enforced_tags,
        "pre_periodicity_gate": {
            "enabled": args.apply_pre_periodicity_gate,
            "router_mode": PREGATE_ROUTER_MODE,
            "min_period": args.pre_periodicity_min_period,
            "max_period": args.pre_periodicity_max_period,
            "n_periods": args.pre_periodicity_n_periods,
            "ce_snr_threshold": args.pre_periodicity_ce_snr_threshold,
            "min_points": args.pre_periodicity_min_points,
            "scatter_ratio_max": args.pre_periodicity_scatter_ratio_max,
            "workers": args.pre_periodicity_workers,
            "checkpoint": str(pre_periodicity_checkpoint),
        },
        "periodic_branch": {
            "enabled": args.apply_pre_periodicity_gate,
            "mode": "events_phase_template_residual",
            "baseline_func": "phase_template",
            "workers": args.workers,
            "cache_dir": str(branch_cache_dir),
        },
        "filter": {
            "apply_evidence_strength": not args.skip_evidence_strength,
            "min_bayes_factor": args.min_bayes_factor,
            "require_finite_local_bf": not args.allow_infinite_local_bf,
            "apply_significant_detection": not args.skip_significant_detection,
            "significant_require_flag": not args.significant_no_require_flag,
            "significant_min_peak_count": args.significant_min_peak_count,
            "significant_min_run_count": args.significant_min_run_count,
            "apply_run_robustness": not args.skip_run_robustness,
            "min_run_count": args.min_run_count,
            "max_run_count": args.max_run_count,
            "min_run_cameras": args.filter_min_run_cameras,
            "min_run_points": args.filter_min_run_points,
            "apply_morphology": args.apply_morphology,
            "dip_morphology": args.dip_morphology,
            "jump_morphology": args.jump_morphology,
            "min_delta_bic": args.min_delta_bic,
            "apply_score": not args.skip_score_filter,
            "min_dip_score": args.min_dip_score,
            "min_jump_score": args.min_jump_score,
            "min_score": args.min_score,
            "apply_periodicity_validation": args.apply_periodicity_validation,
            "periodicity_n_bootstrap": args.periodicity_n_bootstrap,
            "periodicity_significance": args.periodicity_significance,
            "periodicity_pdm_method": args.periodicity_pdm_method,
            "periodicity_exclude_aliases": not args.periodicity_no_exclude_aliases,
            "periodicity_flag_only": not args.periodicity_reject,
            "periodicity_workers": args.periodicity_workers,
            "periodicity_checkpoint_dir": str(args.periodicity_checkpoint_dir) if args.periodicity_checkpoint_dir else None,
            "periodicity_all_candidates": args.periodicity_all_candidates,
            "phase_plot_max_sig": args.phase_plot_max_sig,
            "phase_plot_min_power": args.phase_plot_min_power,
            "phase_plot_allow_alias": args.phase_plot_allow_alias,
            "apply_gaia_ruwe_validation": not args.skip_gaia_ruwe_validation,
            "gaia_max_ruwe": args.gaia_max_ruwe,
            "gaia_flag_only": not args.gaia_reject,
            "apply_gaia_pm_validation": not args.skip_gaia_pm_validation,
            "gaia_max_pm": args.gaia_max_pm,
            "gaia_pm_flag_only": not args.gaia_pm_reject,
            "apply_periodic_catalog_validation": not args.skip_periodic_catalog_validation,
            "periodic_catalog_max_sep": args.periodic_catalog_max_sep,
            "periodic_catalog_flag_only": not args.periodic_catalog_reject,
        },
        "characterize": {
            "run_characterize": args.run_characterize,
            "run_dust": args.run_dust,
            "starhorse": args.characterize_starhorse,
            "starhorse_cache": str(args.characterize_starhorse_cache) if args.characterize_starhorse_cache else None,
            "unwise_checkpoint_every": args.characterize_unwise_checkpoint_every,
            "banyan": args.characterize_banyan,
            "iphas": args.characterize_iphas,
            "sfr": args.characterize_sfr,
            "clusters": args.characterize_clusters,
            "unwise": args.characterize_unwise,
        },
        "downstream_pass_logic": "characterize/classify/enrich run on filter passers (failed_any == False)",
    }

    run_params_file = out_dir / "run_params.json"
    run_params_tagged_file = out_dir / f"run_params_{mag_bin_tag}.json"
    run_summary_file = out_dir / "run_summary.json"
    imported_run_params_snapshot, imported_run_summary_snapshot = preserve_imported_run_snapshots(
        stage=stage,
        import_bundle=args.import_bundle,
        out_dir=out_dir,
        run_params_file=run_params_file,
        run_summary_file=run_summary_file,
    )

    bundle_lightcurve_dir = out_dir / "bundle_assets" / "lightcurves"
    bundle_lightcurve_count = (
        sum(1 for p in bundle_lightcurve_dir.iterdir() if p.is_file())
        if bundle_lightcurve_dir.is_dir()
        else 0
    )
    manifests_file_count = sum(1 for p in manifests_dir.rglob("*") if p.is_file()) if manifests_dir.exists() else 0
    tags_file_count = sum(1 for p in tags_dir.rglob("*") if p.is_file()) if tags_dir.exists() else 0
    paths_file_count = sum(1 for p in paths_dir.rglob("*") if p.is_file()) if paths_dir.exists() else 0

    summary_state = load_summary_state(
        run_summary_file=run_summary_file,
        run_start_time=run_start_time,
        stage=stage,
    )

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
            "review_sync_enabled": args.review_sync_enabled,
            "review_sync_dir": str(args.review_sync_dir),
            "review_sync_hash_assets": bool(args.review_sync_hash_assets),
            "imported_run_params_snapshot": str(imported_run_params_snapshot) if imported_run_params_snapshot else None,
            "imported_run_summary_snapshot": str(imported_run_summary_snapshot) if imported_run_summary_snapshot else None,
            "mag_bin": args.mag_bin,
            # Tag parameters
            "min_time_span": args.min_time_span,
            "min_points_per_day": args.min_points_per_day,
            "min_cameras": args.min_cameras,
            "mag_lo": args.mag_lo,
            "mag_hi": args.mag_hi,
            "skip_sparse": args.skip_sparse,
            "skip_multi_camera": args.skip_multi_camera,
            "skip_mag_range": args.skip_mag_range,
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
            # Cleaning thresholds (hardcoded, not CLI args)
            "clean_max_error_absolute": 1.0,
            "clean_max_error_sigma": 5.0,
            "bad_camera_scatter_ratio": 2.5,
            # Tag stage (camera median)
            "skip_camera_median": args.skip_camera_median,
            "camera_median_tolerance": args.camera_median_tolerance,
            # Pre-events periodicity gate
            "apply_pre_periodicity_gate": args.apply_pre_periodicity_gate,
            "pre_periodicity_min_period": args.pre_periodicity_min_period,
            "pre_periodicity_max_period": args.pre_periodicity_max_period,
            "pre_periodicity_n_periods": args.pre_periodicity_n_periods,
            "pre_periodicity_router_mode": PREGATE_ROUTER_MODE,
            "pre_periodicity_ce_snr_threshold": args.pre_periodicity_ce_snr_threshold,
            "pre_periodicity_min_points": args.pre_periodicity_min_points,
            "pre_periodicity_scatter_ratio_max": args.pre_periodicity_scatter_ratio_max,
            "pre_periodicity_workers": args.pre_periodicity_workers,
            "pre_periodicity_checkpoint": str(pre_periodicity_checkpoint),
            "periodic_branch_events_output": str(periodic_branch_events_output),
            "stochastic_branch_events_output": str(stochastic_branch_events_output),
            # Step 5: Filter
            "run_filter": args.run_filter,
            "skip_evidence_strength": args.skip_evidence_strength,
            "min_bayes_factor": args.min_bayes_factor,
            "allow_infinite_local_bf": args.allow_infinite_local_bf,
            "skip_significant_detection": args.skip_significant_detection,
            "significant_no_require_flag": args.significant_no_require_flag,
            "significant_min_peak_count": args.significant_min_peak_count,
            "significant_min_run_count": args.significant_min_run_count,
            "skip_run_robustness": args.skip_run_robustness,
            "min_run_count": args.min_run_count,
            "max_run_count": args.max_run_count,
            "filter_min_run_cameras": args.filter_min_run_cameras,
            "filter_min_run_points": args.filter_min_run_points,
            "apply_morphology": args.apply_morphology,
            "dip_morphology": args.dip_morphology,
            "jump_morphology": args.jump_morphology,
            "min_delta_bic": args.min_delta_bic,
            "skip_score_filter": args.skip_score_filter,
            "min_dip_score": args.min_dip_score,
            "min_jump_score": args.min_jump_score,
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
            "skip_gaia_pm_validation": args.skip_gaia_pm_validation,
            "gaia_max_pm": args.gaia_max_pm,
            "gaia_pm_reject": args.gaia_pm_reject,
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
            "gaia_fetch_chunk_size": args.gaia_fetch_chunk_size,
            "characterize_crossmatch": str(args.characterize_crossmatch),
            "characterize_chunk_size": args.characterize_chunk_size,
            "characterize_starhorse": args.characterize_starhorse,
            "characterize_starhorse_cache": str(args.characterize_starhorse_cache) if args.characterize_starhorse_cache else None,
            "characterize_unwise_checkpoint_every": args.characterize_unwise_checkpoint_every,
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
            # Step 12: Vetting
            "run_vetting": args.run_vetting,
            "vetting_min_score": args.vetting_min_score,
            "vetting_simbad_radius": args.vetting_simbad_radius,
            "vetting_asassn_radius": args.vetting_asassn_radius,
            # File paths
            "index_root": str(args.index_root),
            "lc_root": str(args.lc_root),
            "flat_lc_dir": str(args.flat_lc_dir.expanduser()) if args.flat_lc_dir else None,
            "index_file": str(args.index_file.expanduser()) if args.index_file else None,
            "out_dir": str(out_dir),
            "manifest_file": str(manifest_file),
            "filtered_file": str(filtered_file),
            "pre_periodicity_file": str(pre_periodicity_file),
            "periodic_branch_file": str(periodic_branch_file),
            "stochastic_branch_file": str(stochastic_branch_file),
            "periodic_paths_file": str(periodic_paths_file),
            "branch_cache_dir": str(branch_cache_dir),
            "events_output": str(events_output),
            "bundle_lightcurve_count": bundle_lightcurve_count,
            "manifests_file_count": manifests_file_count,
            "tags_file_count": tags_file_count,
            "paths_file_count": paths_file_count,
        }

        for p in (run_params_file, run_params_tagged_file):
            try:
                with open(p, "w") as f:
                    json.dump(run_params, f, indent=2, default=str)
            except Exception as e:
                if args.verbose:
                    print(f"Warning: could not write {p.name}: {e}")

    except Exception as e:
        if args.verbose:
            print(f"Warning: could not write run_params.json: {e}")

    # Write a simple run log with the command and key paths.
    run_log = out_dir / "run.log"
    run_log_tagged = out_dir / f"run_{mag_bin_tag}.log"
    try:
        events_cmd_preview = shlex.join([sys.executable, "-m", "malca.events", *events_args, "--", "<paths_file>"])
        run_log_text = "\n".join([
                f"timestamp: {run_start_time.isoformat()}",
                f"command: {cmd}",
                f"events_cmd: {events_cmd_preview}",
                f"out_dir: {out_dir}",
                f"run_params: {run_params_file}",
                f"manifests_dir: {manifests_dir}",
                f"tags_dir: {tags_dir}",
                f"paths_dir: {paths_dir}",
                f"results_dir: {results_dir}",
                f"results_output: {events_output}",
                f"branch_cache_dir: {branch_cache_dir}",
                f"manifest_file: {manifest_file}",
                f"filtered_file: {filtered_file}",
                f"stats_checkpoint: {stats_checkpoint_file}",
                f"rejected_tag: {tags_dir / f'rejected_tag_{mag_bin_tag}.csv'}",
            ]) + "\n"
        for p in (run_log, run_log_tagged):
            p.write_text(run_log_text)
    except Exception as e:
        if args.verbose:
            print(f"Warning: could not write run log: {e}")

    df_manifest = pd.DataFrame()
    df_filtered = pd.DataFrame()
    df_periodic_candidates = pd.DataFrame()
    pre_periodicity_stats: dict[str, object] | None = None
    branch_detection_stats: dict[str, object] | None = None

    def _normalized_output_path(path: Path, fmt: str) -> Path:
        if fmt == "parquet_chunk":
            return path if not path.suffix else path.with_suffix("")
        expected_suffix = ".csv" if fmt == "csv" else ".parquet"
        return path if path.suffix.lower() == expected_suffix else path.with_suffix(expected_suffix)

    def _output_files_for_path(path: Path, fmt: str) -> list[Path]:
        path = _normalized_output_path(path, fmt)
        if fmt == "parquet_chunk":
            return sorted(path.glob("chunk_*.parquet")) if path.exists() and path.is_dir() else []
        return [path] if path.exists() and path.is_file() else []

    def _load_events_output(path: Path, fmt: str) -> pd.DataFrame:
        path = _normalized_output_path(path, fmt)
        files = _output_files_for_path(path, fmt)
        if not files:
            return pd.DataFrame()
        if fmt == "csv":
            return pd.read_csv(files[0])
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def _write_events_output(df: pd.DataFrame, path: Path, fmt: str) -> list[Path]:
        path = _normalized_output_path(path, fmt)
        if fmt == "parquet_chunk":
            if path.exists():
                clear_existing_output(path, fmt)
            path.mkdir(parents=True, exist_ok=True)
            chunk_path = path / "chunk_000000.parquet"
            df.to_parquet(chunk_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)
            return [chunk_path]
        if fmt == "csv":
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
        else:
            safe_write_parquet(df, path)
        return [path]

    def _run_events_branch(
        file_paths: list[str],
        *,
        branch_name: str,
        branch_output: Path,
        baseline_func_override: str | None = None,
        metadata_df: pd.DataFrame | None = None,
        branch_paths_file: Path | None = None,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        branch_output = Path(branch_output)
        checkpoint_log = branch_output.with_name(f"{branch_output.stem}_PROCESSED.txt")
        error_log = branch_output.with_name(f"{branch_output.stem}_ERRORS.csv")
        metadata_path: Path | None = None

        if args.overwrite:
            clear_existing_output(branch_output, "parquet")
            checkpoint_log.unlink(missing_ok=True)
            error_log.unlink(missing_ok=True)

        if metadata_df is not None and not metadata_df.empty:
            metadata_dir = tags_dir / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = metadata_dir / f"metadata_{branch_name}_{mag_bin_tag}.csv"
            metadata_df.to_csv(metadata_path, index=False)
            log(
                f"Wrote {branch_name} metadata CSV with columns: "
                f"{', '.join(col for col in metadata_df.columns if col != 'path')}"
            )

        if branch_paths_file is not None:
            branch_paths_file.parent.mkdir(parents=True, exist_ok=True)
            with open(branch_paths_file, "w") as f:
                for path_value in file_paths:
                    f.write(f"{path_value}\n")

        processed_paths: set[str] = set()
        if checkpoint_log.exists() and not args.overwrite:
            try:
                with open(checkpoint_log, "r") as f:
                    processed_paths = {line.strip() for line in f if line.strip()}
                log(
                    f"{branch_name.title()} branch checkpoint detected, "
                    f"skipping {len(processed_paths)} already-processed paths"
                )
            except Exception as e:
                log(f"Warning: could not read checkpoint log {checkpoint_log}: {e}")

        remaining = [path_value for path_value in file_paths if str(path_value) not in processed_paths]
        if file_paths and (not remaining):
            log(f"All {branch_name} branch paths already processed according to checkpoint.")

        batch_size = max(1, args.batch_size)
        total_batches = (len(remaining) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(len(remaining), start + batch_size)
            batch_paths = remaining[start:end]
            log(
                f"\nRunning {branch_name} branch batch {batch_idx + 1}/{total_batches} "
                f"({len(batch_paths)} LCs)..."
            )

            batch_paths_file = paths_dir / f"batch_paths_{branch_name}_{mag_bin_tag}_{batch_idx}.txt"
            with open(batch_paths_file, "w") as f:
                for path_value in batch_paths:
                    f.write(f"{path_value}\n")

            branch_events_args = list(events_args)
            if baseline_func_override is not None:
                branch_events_args.extend(["--baseline-func", baseline_func_override])
            branch_events_args.extend([
                "--output-format",
                "parquet",
                "--output",
                str(branch_output),
                "--error-output",
                str(error_log),
            ])
            if metadata_path is not None:
                branch_events_args.extend(["--metadata-csv", str(metadata_path)])

            events_cmd = [
                sys.executable,
                "-m",
                "malca.events",
                *branch_events_args,
                "--input-file",
                str(batch_paths_file),
            ]

            try:
                result = subprocess.run(events_cmd, check=False)
                if result.returncode != 0:
                    print(f"events.py returned non-zero exit ({result.returncode}); stopping.")
                    sys.exit(result.returncode)
            except Exception as e:
                print(f"\nError running events.py for {branch_name} branch: {e}")
                if branch_paths_file is not None:
                    print(f"\nBranch paths saved to: {branch_paths_file}")
                sys.exit(1)

        df_branch = pd.read_parquet(branch_output) if branch_output.exists() else pd.DataFrame()
        stats = {
            "branch": branch_name,
            "baseline_func": baseline_func_override or args.baseline_func,
            "total_input": int(len(file_paths)),
            "total_results": int(len(df_branch)),
            "dip_significant": int(df_branch["dip_significant"].fillna(False).sum()) if "dip_significant" in df_branch.columns else 0,
            "jump_significant": int(df_branch["jump_significant"].fillna(False).sum()) if "jump_significant" in df_branch.columns else 0,
            "output_file": str(branch_output),
            "metadata_file": str(metadata_path) if metadata_path is not None else None,
            "paths_file": str(branch_paths_file) if branch_paths_file is not None else None,
        }
        return df_branch, stats

    # Step 1: Build or load manifest
    if run_upstream:
        if args.force_manifest or not manifest_file.exists():
            if args.flat_lc_dir:
                log(f"Building flat-directory manifest for mag_bin={args.mag_bin} from {Path(args.flat_lc_dir).expanduser()}...")
            else:
                log(f"Building manifest for mag_bin={args.mag_bin}...")
            df_manifest = build_manifest(
                args.index_root,
                args.lc_root,
                mag_bins=args.mag_bin,
                id_column="asas_sn_id",
                file_ext=args.extension,
                show_progress=args.verbose,
                n_workers=args.workers,
                flat_lc_dir=args.flat_lc_dir.expanduser() if args.flat_lc_dir else None,
                index_file=args.index_file.expanduser() if args.index_file else None,
            )

            # Only keep sources where light curve files exist
            df_manifest = df_manifest[df_manifest["dat_exists"]].reset_index(drop=True)

            log(f"Saving manifest to {manifest_file} ({len(df_manifest)} sources)")
            safe_write_parquet(df_manifest, manifest_file)
        else:
            log(f"Loading existing manifest from {manifest_file}")
            df_manifest = pd.read_parquet(manifest_file)
            log(f"Loaded {len(df_manifest)} sources")

        # Step 2: Apply tags
        if args.force_tag or not filtered_file.exists():
            log(f"\nApplying tags with {args.workers} workers...")

            # Use lc_dir as the directory path for tag input (path/<id>.dat2)
            df_to_filter = df_manifest.rename(columns={"lc_dir": "path"}).copy()

            df_filtered = apply_tags(
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
                apply_mag_range=not args.skip_mag_range,
                mag_lo=args.mag_lo,
                mag_hi=args.mag_hi,
                n_workers=args.workers,
                show_tqdm=args.verbose,
                rejected_log_csv=str(tags_dir / f"rejected_tag_{mag_bin_tag}.csv"),
                stats_checkpoint=str(stats_checkpoint_file),
                stats_chunk_size=args.stats_chunk_size,
                file_ext=args.extension,
            )

            # Exclude rows based on tag results
            if not args.pass_all_tags:
                failed_cols = [c for c in df_filtered.columns if c.startswith("failed_") and c != "failed_any"]

                if args.enforce_tags:
                    # Only enforce specified filters
                    enforce_set = {f"failed_{f.strip()}" for f in args.enforce_tags.split(",")}
                    enforce_cols = [c for c in failed_cols if c in enforce_set]
                else:
                    enforce_cols = failed_cols

                if enforce_cols:
                    exclude_mask = df_filtered[enforce_cols].any(axis=1)
                    df_filtered = df_filtered[~exclude_mask].reset_index(drop=True)

            log(f"\nKept {len(df_filtered)}/{len(df_manifest)} sources after tagging")
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
    camera_median_file = tags_dir / f"camera_medians_{mag_bin_tag}.parquet"
    if run_upstream and (not args.skip_camera_median) and ("mag_bin" in df_filtered.columns):
        if args.force_tag or not camera_median_file.exists():
            log(f"\nApplying camera median filter (tolerance={args.camera_median_tolerance} mag)...")
            # Camera median validation needs per-source file paths (.dat2 -> .raw2).
            # Keep the original path column unchanged for downstream code.
            camera_median_df = df_filtered.copy()
            if "dat_path" in camera_median_df.columns:
                camera_median_df["path"] = camera_median_df["dat_path"]
            camera_median_checkpoint = tags_dir / f"camera_medians_{mag_bin_tag}_CHECKPOINT.parquet"
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

    # Step 2.75: Pre-events periodicity gate and branch split
    if run_upstream and args.apply_pre_periodicity_gate and not df_filtered.empty:
        rerun_pre_periodicity = bool(args.force_tag or not pre_periodicity_file.exists())
        if not rerun_pre_periodicity:
            log(f"\nLoading cached pre-periodicity gate results from {pre_periodicity_file}")
            df_gate = pd.read_parquet(pre_periodicity_file)
            cached_router_ok = False
            if "pre_periodicity_router_mode" in df_gate.columns:
                cached_router_modes = (
                    df_gate["pre_periodicity_router_mode"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .unique()
                    .tolist()
                )
                cached_router_ok = cached_router_modes == [PREGATE_ROUTER_MODE.lower()]
            if "pre_periodic_flag" not in df_gate.columns or not cached_router_ok:
                rerun_pre_periodicity = True
                log(
                    "Cached pre-periodicity gate output is incompatible with the requested "
                    f"router mode '{PREGATE_ROUTER_MODE}'; recomputing."
                )

        if rerun_pre_periodicity:
            log(
                "\nRunning pre-events periodicity gate "
                f"({PREGATE_ROUTER_MODE}, workers={args.pre_periodicity_workers})..."
            )
            df_gate = apply_pre_periodicity_gate(
                df_filtered,
                path_col="dat_path" if "dat_path" in df_filtered.columns else "path",
                excluded_cameras_col="excluded_cameras" if "excluded_cameras" in df_filtered.columns else None,
                bad_camera_scatter_ratio=BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
                clean_max_error_absolute=CLEAN_LC_MAX_ERROR_ABSOLUTE,
                clean_max_error_sigma=CLEAN_LC_MAX_ERROR_SIGMA,
                min_period=args.pre_periodicity_min_period,
                max_period=args.pre_periodicity_max_period,
                n_periods=args.pre_periodicity_n_periods,
                ce_snr_threshold=args.pre_periodicity_ce_snr_threshold,
                min_points=args.pre_periodicity_min_points,
                scatter_ratio_max=args.pre_periodicity_scatter_ratio_max,
                workers=args.pre_periodicity_workers,
                checkpoint_path=pre_periodicity_checkpoint,
                show_tqdm=args.verbose,
            )
            safe_write_parquet(df_gate, pre_periodicity_file)

        if "pre_periodic_flag" not in df_gate.columns:
            raise ValueError(
                f"Pre-periodicity gate output at {pre_periodicity_file} is missing 'pre_periodic_flag'"
            )

        df_periodic_candidates = df_gate[df_gate["pre_periodic_flag"].fillna(False)].reset_index(drop=True)
        df_filtered = df_gate[~df_gate["pre_periodic_flag"].fillna(False)].reset_index(drop=True)

        safe_write_parquet(df_periodic_candidates, periodic_branch_file)
        safe_write_parquet(df_filtered, stochastic_branch_file)

        periodic_paths: list[str] = []
        if not df_periodic_candidates.empty:
            periodic_path_col = "dat_path" if "dat_path" in df_periodic_candidates.columns else "path"
            periodic_paths = [str(value) for value in df_periodic_candidates[periodic_path_col].tolist()]
        with open(periodic_paths_file, "w") as handle:
            for item in periodic_paths:
                handle.write(f"{item}\n")

        label_counts = (
            df_gate["pre_periodicity_label"].value_counts(dropna=False).to_dict()
            if "pre_periodicity_label" in df_gate.columns else {}
        )
        pre_periodicity_stats = {
            "total_input": int(len(df_gate)),
            "periodic_branch": int(len(df_periodic_candidates)),
            "stochastic_branch": int(len(df_filtered)),
            "label_counts": {str(key): int(value) for key, value in label_counts.items()},
            "pre_periodicity_file": str(pre_periodicity_file),
            "periodic_branch_file": str(periodic_branch_file),
            "stochastic_branch_file": str(stochastic_branch_file),
            "periodic_paths_file": str(periodic_paths_file),
        }
        log(
            f"Pre-periodicity gate routed {len(df_periodic_candidates)}/{len(df_gate)} "
            "candidates to the periodic branch"
        )

    periodic_audit_cols = [
        "pre_periodicity_label",
        "pre_periodic_flag",
        "pre_periodicity_selected_period",
        "pre_periodicity_method",
    ]

    def _build_branch_metadata_df(df_branch: pd.DataFrame, path_col: str) -> pd.DataFrame | None:
        meta_cols = [path_col]
        if not args.skip_vsx and "vsx_sep_arcsec" in df_branch.columns and "vsx_class" in df_branch.columns:
            meta_cols.extend(["vsx_sep_arcsec", "vsx_class"])
        if "excluded_cameras" in df_branch.columns:
            meta_cols.append("excluded_cameras")
        for col in periodic_audit_cols:
            if col in df_branch.columns and col not in meta_cols:
                meta_cols.append(col)
        if len(meta_cols) == 1:
            return None
        return df_branch[meta_cols].rename(columns={path_col: "path"}).copy()

    paths_file = paths_dir / f"filtered_paths_{mag_bin_tag}.txt"
    if run_upstream:
        if run_log.exists():
            try:
                with run_log.open("a") as f:
                    f.write(f"paths_file: {paths_file}\n")
                    f.write(f"periodic_paths_file: {periodic_paths_file}\n")
            except Exception as e:
                if args.verbose:
                    print(f"Warning: could not update run log with branch paths files: {e}")

        stochastic_file_col = "dat_path" if "dat_path" in df_filtered.columns else "path"
        stochastic_paths = [str(value) for value in df_filtered[stochastic_file_col].tolist()] if not df_filtered.empty else []
        stochastic_metadata_df = _build_branch_metadata_df(df_filtered, stochastic_file_col) if not df_filtered.empty else None

        periodic_file_col = "dat_path" if "dat_path" in df_periodic_candidates.columns else "path"
        periodic_paths = [str(value) for value in df_periodic_candidates[periodic_file_col].tolist()] if not df_periodic_candidates.empty else []
        periodic_metadata_df = _build_branch_metadata_df(df_periodic_candidates, periodic_file_col) if not df_periodic_candidates.empty else None

        if stochastic_paths:
            log(f"\nPreparing to run stochastic branch events on {len(stochastic_paths)} light curves...")
        else:
            log("\nNo stochastic-branch sources to process after filtering.")

        if periodic_paths:
            log(
                "\nPreparing to run periodic-branch residual events on "
                f"{len(periodic_paths)} light curves..."
            )
        elif args.apply_pre_periodicity_gate:
            log("\nNo periodic-branch sources to process after the pre-periodicity gate.")

        df_stochastic_events, stochastic_stats = _run_events_branch(
            stochastic_paths,
            branch_name="stochastic",
            branch_output=stochastic_branch_events_output,
            metadata_df=stochastic_metadata_df,
            branch_paths_file=paths_file,
        )

        df_periodic_events = pd.DataFrame()
        periodic_stats = {
            "branch": "periodic",
            "baseline_func": "phase_template",
            "total_input": 0,
            "total_results": 0,
            "dip_significant": 0,
            "jump_significant": 0,
            "output_file": str(periodic_branch_events_output),
            "metadata_file": None,
            "paths_file": str(periodic_paths_file),
        }
        if args.apply_pre_periodicity_gate:
            df_periodic_events, periodic_stats = _run_events_branch(
                periodic_paths,
                branch_name="periodic",
                branch_output=periodic_branch_events_output,
                baseline_func_override="phase_template",
                metadata_df=periodic_metadata_df,
                branch_paths_file=periodic_paths_file,
            )

        branch_frames = [df for df in (df_stochastic_events, df_periodic_events) if not df.empty]
        if branch_frames:
            df_events_merged = pd.concat(branch_frames, ignore_index=True)
            if "path" in df_events_merged.columns:
                df_events_merged = df_events_merged.drop_duplicates(subset=["path"], keep="last")
            results_files = _write_events_output(df_events_merged, events_output, events_format)
            log(f"\nMerged branch outputs into canonical events product at {events_output}")
        else:
            if args.overwrite:
                clear_existing_output(_normalized_output_path(events_output, events_format), events_format)
            results_files = []
            log("\nNo event-branch results were produced.")
        branch_detection_stats = {
            "stochastic": stochastic_stats,
            "periodic": periodic_stats if args.apply_pre_periodicity_gate else None,
        }
    else:
        results_files = _output_files_for_path(events_output, events_format)

    # Generate run summary with results statistics
    run_end_time = datetime.now()
    run_summary_file = out_dir / "run_summary.json"
    try:
        summary = build_run_summary(
            previous_summary=summary_state if isinstance(summary_state, dict) else {},
            run_start_time=run_start_time,
            run_end_time=run_end_time,
            config_fingerprint=config_fingerprint,
            run_upstream=run_upstream,
            manifest_total_sources=(int(len(df_manifest)) if run_upstream else None),
            manifest_filtered_sources=(int(len(df_filtered) + len(df_periodic_candidates)) if run_upstream else None),
            artifact_context={
                "stage": stage,
                "bundle_lightcurve_count": int(bundle_lightcurve_count),
                "manifests_file_count": int(manifests_file_count),
                "tags_file_count": int(tags_file_count),
                "paths_file_count": int(paths_file_count),
                "imported_run_params_snapshot": str(imported_run_params_snapshot) if imported_run_params_snapshot else None,
                "imported_run_summary_snapshot": str(imported_run_summary_snapshot) if imported_run_summary_snapshot else None,
            },
        )
        if pre_periodicity_stats is not None:
            summary["pre_periodicity_gate"] = pre_periodicity_stats
        if branch_detection_stats is not None:
            summary["events_branches"] = branch_detection_stats

        # Tag rejection breakdown
        rejected_log = tags_dir / f"rejected_tag_{mag_bin_tag}.csv"
        if rejected_log.exists():
            try:
                df_rejected = pd.read_csv(rejected_log)
                if "reason" in df_rejected.columns:
                    rejection_counts = df_rejected["reason"].value_counts().to_dict()
                    summary["tag_rejections"] = {
                        "total_rejected": len(df_rejected),
                        "by_reason": rejection_counts,
                    }
            except Exception as e:
                if args.verbose:
                    print(f"Warning: could not parse rejection log: {e}")

        if results_files:
            try:
                df_results = _load_events_output(events_output, events_format)

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

        # Write summary (will be updated again if filter/postprocess run)
        with open(run_summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        log(f"\nRun summary saved to {run_summary_file}")

    except Exception as e:
        if args.verbose:
            print(f"Warning: could not write run summary: {e}")

    # Step 5: Apply filters (optional)
    if run_upstream and args.run_filter and results_files:
        log("\n=== Step 5: Applying filters ===")
        try:
            # Load events results
            if events_format == "parquet_chunk":
                df_events = pd.concat([pd.read_parquet(f) for f in results_files], ignore_index=True)
            else:
                df_events = load_table(results_files[0])

            # Apply filters
            filter_kwargs = _build_filter_kwargs(args)
            if stage == "cluster":
                # Cluster stage must avoid internet catalog lookups.
                filter_kwargs["apply_gaia_ruwe_validation"] = False
                filter_kwargs["apply_gaia_pm_validation"] = False
                filter_kwargs["apply_periodic_catalog_validation"] = False

            # Add gaia_id from ASASSN index (needed for validate_gaia_ruwe/pm filters)
            if filter_kwargs.get("apply_gaia_ruwe_validation", True) or filter_kwargs.get("apply_gaia_pm_validation", True):
                _gaia_index_path, _ = _resolve_asassn_index_path(out_dir, index_override=getattr(args, "index_file", None))
                if _gaia_index_path:
                    df_events = _add_gaia_ids_from_index(df_events, _gaia_index_path)
                else:
                    _log("Warning: ASASSN index not found; gaia_id will be missing — RUWE/PM filters will have no matches")

            df_post_filtered = apply_filters(df_events, **filter_kwargs)

            # Save filtered results
            post_filter_output = results_dir / f"lc_events_filtered_{mag_bin_tag}.parquet"
            save_table(df_post_filtered, post_filter_output)
            log(f"Filtered results saved to {post_filter_output}")

            # Update summary with filter stats
            n_passed = int((~df_post_filtered["failed_any"]).sum()) if "failed_any" in df_post_filtered.columns else len(df_post_filtered)
            n_failed = int(df_post_filtered["failed_any"].sum()) if "failed_any" in df_post_filtered.columns else 0
            summary["filter_stats"] = {
                "total_input": len(df_events),
                "passed": n_passed,
                "failed": n_failed,
                "pass_rate": n_passed / len(df_events) if len(df_events) > 0 else 0.0,
            }
            summary["post_filter_stats"] = summary["filter_stats"]

            # Overwrite summary with updated stats
            with open(run_summary_file, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            log(f"Filter: {n_passed}/{len(df_events)} passed")

        except Exception as e:
            print(f"Error in filter step: {e}")
            if args.verbose:

                traceback.print_exc()

    # Step 6: Enrich with compute_stats (optional, runs immediately after filter)
    if run_upstream and args.run_enrich:
        if not args.run_filter:
            print("Warning: --run-enrich requires --run-filter. Skipping enrichment.")
        else:
            log("\n=== Step 6: Enriching with light curve stats ===")
            try:
                # Enrichment now runs directly from filter output
                post_filter_output = results_dir / f"lc_events_filtered_{mag_bin_tag}.parquet"

                if post_filter_output.exists():
                    df_to_enrich = load_table(post_filter_output)
                else:
                    print(f"Warning: No filter output found at {post_filter_output}")
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
                        enrich_checkpoint = results_dir / f"lc_events_enriched_{mag_bin_tag}_CHECKPOINT.parquet"
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
                        n_enrich_workers = max(1, args.workers)
                        log(f"Using {n_enrich_workers} workers for compute_stats enrichment")

                        # Build task list, handling already-enriched and missing paths serially
                        pending_tasks: list[tuple] = []
                        for idx, row in df_passed.iterrows():
                            lc_path = Path(row["path"])
                            if str(lc_path) in already_enriched:
                                continue
                            if not lc_path.exists():
                                enriched_rows.append(row.to_dict())
                                continue
                            asassn_id = lc_path.stem.split("-")[0]
                            dir_path = str(lc_path.parent)
                            pending_tasks.append((row.to_dict(), asassn_id, dir_path, args.enrich_compute_ls))

                        new_count = 0
                        with ProcessPoolExecutor(max_workers=n_enrich_workers) as executor:
                            for result in tqdm(
                                executor.map(_enrich_row_worker, pending_tasks),
                                total=len(pending_tasks),
                                desc="compute_stats",
                                disable=not args.verbose,
                            ):
                                enriched_rows.append(result)
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
                        enrich_output = results_dir / f"lc_events_enriched_{mag_bin_tag}.parquet"
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

                    traceback.print_exc()

    # Step 7: Generate review plots (optional)
    if run_upstream and args.run_postprocess:
        if not args.run_filter:
            print("Warning: --run-postprocess requires --run-filter. Skipping postprocess plots.")
        else:
            log("\n=== Step 7: Generating candidate plots ===")
            try:
                post_filter_output = results_dir / f"lc_events_filtered_{mag_bin_tag}.parquet"
                if not post_filter_output.exists():
                    print(f"Warning: No filter output found at {post_filter_output}; skipping postprocess plots.")
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
                        jd_offset=JD_OFFSET,
                        clean_max_error_absolute=CLEAN_LC_MAX_ERROR_ABSOLUTE,
                        clean_max_error_sigma=CLEAN_LC_MAX_ERROR_SIGMA,
                        run_params=run_params if 'run_params' in locals() else None,
                        filter_bad_cameras=True,
                        bad_camera_scatter_ratio=BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
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

                    traceback.print_exc()



    # Merge per-mag-bin outputs into canonical (untagged) files for downstream stages.
    # Only merge when entering downstream/home phase — NOT during concurrent cluster runs.
    if run_downstream:
        log("\n=== Merging per-mag-bin outputs ===")
        merge_started = time.perf_counter()
        for merge_prefix in ("lc_events_results", "lc_events_filtered", "lc_events_enriched"):
            tagged_files = sorted(results_dir.glob(f"{merge_prefix}_*.parquet"))
            # Exclude checkpoint and temp files from merging
            tagged_files = [
                f for f in tagged_files
                if "_CHECKPOINT" not in f.name and "_PROCESSED" not in f.name and not f.name.endswith(".tmp")
            ]
            merged_path = results_dir / f"{merge_prefix}.parquet"
            if tagged_files:
                try:
                    dfs = [pd.read_parquet(f) for f in tagged_files]
                    merged = pd.concat(dfs, ignore_index=True)
                    if "path" in merged.columns:
                        merged = merged.drop_duplicates(subset=["path"], keep="last")
                    save_table(merged, merged_path)
                    log(f"Merged {len(tagged_files)} files into {merged_path} ({len(merged)} rows)")
                except Exception as e:
                    log(f"Warning: could not merge {merge_prefix} files: {e}")
        log(f"Merge step completed in {time.perf_counter() - merge_started:.1f}s")

    post_filter_output = results_dir / "lc_events_filtered.parquet"
    has_post_filter_output = post_filter_output.exists()

    # Home-only external catalog validations (Gaia RUWE + periodic catalog)
    if stage == "home" and args.run_filter and has_post_filter_output:
        home_validation_steps = ["Gaia RUWE", "periodic catalog"]
        if args.apply_periodicity_validation:
            home_validation_steps.append("periodicity")
        log(f"\n=== Home External Validation: {' + '.join(home_validation_steps)} ===")
        validation_started = time.perf_counter()
        try:
            index_file, index_candidates = _resolve_asassn_index_path(out_dir, index_override=args.index_file)
            if index_file is None:
                tried_paths = ", ".join(str(p) for p in index_candidates[:6])
                if len(index_candidates) > 6:
                    tried_paths += ", ..."
                if not tried_paths:
                    tried_paths = "(no candidate paths)"
                raise FileNotFoundError(
                    "Index file not found for home external validation. "
                    f"Tried: {tried_paths}. "
                    "Expected bundle_assets/asassn_index_full.parquet from a --full-bundle export, "
                    "or pass --index-file explicitly."
                )
            log(f"Using index file for home external validation: {index_file}")

            external_validation_cmd = _build_home_external_validation_cmd(
                args,
                post_filter_output=post_filter_output,
                index_file=index_file,
            )

            result = subprocess.run(external_validation_cmd, check=False)
            if result.returncode != 0:
                print(f"Home external validation failed with exit code {result.returncode}")
                sys.exit(result.returncode)

            has_post_filter_output = post_filter_output.exists()
            log(f"Home external validation wrote updated filtered results to {post_filter_output}")
            log(f"Home external validation completed in {time.perf_counter() - validation_started:.1f}s")
        except Exception as e:
            print(f"Error in home external validation step: {e}")
            if args.verbose:

                traceback.print_exc()
            sys.exit(1)



    if run_downstream and (args.run_characterize or args.run_dust) and (not has_post_filter_output):
        print(f"Warning: downstream stage requires filtered results at {post_filter_output}. Skipping characterization.")

    # Step 7b: Auto-fetch Gaia data for characterization (incremental)
    if run_downstream and (args.run_characterize or args.run_dust) and has_post_filter_output:
        log("\n=== Ensuring local Gaia catalog is up to date ===")
        gaia_fetch_started = time.perf_counter()
        try:
            gaia_catalog_path = args.gaia_cache.expanduser() if args.gaia_cache else GAIA_LOCAL_CATALOG
            gaia_ids = _extract_gaia_ids(
                post_filter_output,
                args.characterize_crossmatch.expanduser(),
                only_passers=True,
            )
            if gaia_ids:
                fetch_gaia_catalog(gaia_ids, output_path=gaia_catalog_path, chunk_size=args.gaia_fetch_chunk_size)
            else:
                log("No Gaia IDs found; skipping Gaia fetch.")
            log(f"Gaia catalog check completed in {time.perf_counter() - gaia_fetch_started:.1f}s")
        except Exception as e:
            print(f"Warning: Gaia auto-fetch failed: {e}")
            if args.verbose:

                traceback.print_exc()

    # Step 8: Characterization + dust (optional)
    if run_downstream and (args.run_characterize or args.run_dust) and has_post_filter_output:
        log("\n=== Step 8: Characterizing candidates ===")
        characterize_started = time.perf_counter()
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
                starhorse_cache=args.characterize_starhorse_cache.expanduser() if args.characterize_starhorse_cache else None,
                run_banyan=args.run_characterize and args.characterize_banyan,
                run_iphas=args.run_characterize and args.characterize_iphas,
                run_sfr=args.run_characterize and args.characterize_sfr,
                run_clusters=args.run_characterize and args.characterize_clusters,
                run_unwise=args.run_characterize and args.characterize_unwise,
                unwise_checkpoint_every=args.characterize_unwise_checkpoint_every,
                checkpoint_path=char_checkpoint,
            )

            characterize_output = results_dir / "lc_events_characterized.parquet"
            save_table(df_char, characterize_output)
            log(f"Characterization results saved to {characterize_output}")
            log(f"Step 8 completed in {time.perf_counter() - characterize_started:.1f}s")

        except Exception as e:
            print(f"Error in characterization step: {e}")
            if args.verbose:

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
                classify_started = time.perf_counter()
                try:
                    characterize_output = results_dir / "lc_events_characterized.parquet"
                    post_filter_output = results_dir / "lc_events_filtered.parquet"

                    if characterize_output.exists():
                        df_post_filtered = load_table(characterize_output)
                    elif post_filter_output.exists():
                        df_post_filtered = load_table(post_filter_output)
                    else:
                        df_post_filtered = None
                        print(f"Warning: filter output not found at {post_filter_output}")

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
                            log(f"Step 9 completed in {time.perf_counter() - classify_started:.1f}s")
                        else:
                            log("No passing candidates to classify.")
                            log(f"Step 9 completed in {time.perf_counter() - classify_started:.1f}s")

                except Exception as e:
                    print(f"Error in classification step: {e}")
                    if args.verbose:

                        traceback.print_exc()

    # Step 10: Neighbor enrichment (optional)
    if run_downstream and args.run_neighbor_enrich:
        if not has_post_filter_output:
            print(f"Warning: downstream stage requires filtered results at {post_filter_output}. Skipping neighbor enrichment.")
        else:
            log("\n=== Step 10: Bulk neighbor enrichment ===")
            neighbor_started = time.perf_counter()
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
                        show_progress=args.verbose,
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
                    log(f"Step 10 completed in {time.perf_counter() - neighbor_started:.1f}s")

            except Exception as e:
                print(f"Error in neighbor enrichment step: {e}")
                if args.verbose:

                    traceback.print_exc()

    # Step 11: Spectra availability enrichment (optional)
    if run_downstream and args.run_spectra_enrich:
        if not has_post_filter_output:
            print(f"Warning: downstream stage requires filtered results at {post_filter_output}. Skipping spectra enrichment.")
        else:
            log("\n=== Step 11: Spectra availability enrichment ===")
            spectra_started = time.perf_counter()
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
                        show_progress=args.verbose,
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
                    log(f"Step 11 completed in {time.perf_counter() - spectra_started:.1f}s")

            except Exception as e:
                print(f"Error in spectra enrichment step: {e}")
                if args.verbose:

                    traceback.print_exc()

    # Step 12: Post-review vetting (enabled by default)
    if run_downstream and args.run_vetting:
        log("\n=== Step 12: Post-review vetting ===")
        vetting_started = time.perf_counter()
        try:
            # Find the best input file for vetting
            vetting_input = args.vetting_input
            if vetting_input is None:
                for candidate_file in [
                    results_dir / "lc_events_spectra.parquet",
                    results_dir / "lc_events_neighbors.parquet",
                    results_dir / "lc_events_characterized.parquet",
                    post_filter_output,
                ]:
                    if candidate_file.exists():
                        vetting_input = candidate_file
                        break

            if vetting_input is None or not Path(vetting_input).exists():
                log("Warning: no suitable input found for vetting, skipping")
            else:
                df_vet = load_table(vetting_input)
                df_vet = _select_passing_candidates(df_vet)
                log(f"Vetting input: {vetting_input} ({len(df_vet)} passing candidates)")

                if args.vetting_min_score is not None and "interest_score" in df_vet.columns:
                    before = len(df_vet)
                    df_vet = df_vet[df_vet["interest_score"] >= args.vetting_min_score].copy()
                    log(f"Filtered to {len(df_vet)} candidates with score >= {args.vetting_min_score} (from {before})")

                vetting_checkpoint = results_dir / "lc_events_vetting_CHECKPOINT.parquet"
                df_vet = vet_candidates(
                    df_vet,
                    run_simbad=not args.no_vetting_simbad,
                    run_gaia_var=not args.no_vetting_gaia_var,
                    run_gaia_epoch=not args.no_vetting_gaia_epoch,
                    run_asassn_var=not args.no_vetting_asassn_var,
                    run_alerce=not args.no_vetting_alerce,
                    run_erosita=not args.no_vetting_erosita,
                    run_atlas=args.vetting_atlas,
                    run_pm_check=not args.no_vetting_pm_check,
                    run_neowise_lc=args.vetting_neowise_lc,
                    simbad_radius_arcsec=args.vetting_simbad_radius,
                    asassn_radius_arcsec=args.vetting_asassn_radius,
                    atlas_token=args.vetting_atlas_token,
                    checkpoint_path=vetting_checkpoint,
                )

                vetting_output = results_dir / "lc_events_vetted.parquet"
                save_table(df_vet, vetting_output)
                log(f"Vetting output: {vetting_output}")

                def _count_col(col, empty=""):
                    s = df_vet.get(col, pd.Series(dtype=str))
                    return int((s != empty).sum()) if not s.empty else 0

                summary["vetting_stats"] = {
                    "rows_input": int(len(df_vet)),
                    "simbad_matches": _count_col("simbad_main_id"),
                    "gaia_var_flagged": int(df_vet.get("gaia_var_flag", pd.Series(dtype=bool)).sum()),
                    "gaia_epoch_available": int(df_vet.get("gaia_epoch_available", pd.Series(dtype=bool)).sum()),
                    "asassn_var_matches": _count_col("asassn_var_type"),
                    "alerce_matches": _count_col("alerce_oid"),
                    "erosita_xray_det": int(df_vet.get("xray_det", pd.Series(dtype=bool)).sum()),
                    "likely_known": int(df_vet.get("vetting_likely_known", pd.Series(dtype=bool)).sum()),
                }
                with open(run_summary_file, "w") as f:
                    json.dump(summary, f, indent=2, default=str)
                log(f"Step 12 completed in {time.perf_counter() - vetting_started:.1f}s")

        except Exception as e:
            print(f"Error in vetting step: {e}")
            if args.verbose:

                traceback.print_exc()

    # Step 13: Auto-import into review DB
    if run_downstream and has_post_filter_output:
        log("\n=== Step 13: Importing candidates into review DB ===")
        review_db_path = out_dir / "review" / "review.db"
        review_db_updated = False
        try:


            # Find best available results file
            _import_file = None
            for _candidate_path in [
                results_dir / "lc_events_vetted.parquet",
                results_dir / "lc_events_spectra.parquet",
                results_dir / "lc_events_neighbors.parquet",
                results_dir / "lc_events_classified.parquet",
                results_dir / "lc_events_characterized.parquet",
                results_dir / "lc_events_filtered.parquet",
            ]:
                if _candidate_path.exists():
                    _import_file = _candidate_path
                    break

            if _import_file is not None:
                conn = db_connect(review_db_path)
                df_import = load_table(_import_file)
                df_import = _select_passing_candidates(df_import)
                if df_import.empty:
                    conn.close()
                    log(f"No passing candidates to import into {review_db_path}")
                else:
                    if "candidate_id" not in df_import.columns:
                        if "asas_sn_id" in df_import.columns:
                            df_import = df_import.copy()
                            df_import["candidate_id"] = df_import["asas_sn_id"].astype(str)
                        elif "path" in df_import.columns:
                            df_import = df_import.copy()
                            df_import["candidate_id"] = df_import["path"].apply(
                                lambda p: Path(str(p)).stem.split(".")[0]
                            )
                    n_total, n_new = import_candidates(
                        conn,
                        df_import,
                        source_path=str(out_dir.resolve()),
                        characterize_before_import=False,
                        vet_before_import=False,
                    )
                    conn.close()
                    review_db_updated = True
                    log(f"Imported {n_new} new candidates ({n_total} total) into {review_db_path}")
            else:
                log("No results file found for review DB import, skipping")

        except Exception as e:
            print(f"Warning: review DB import failed: {e}")
            if args.verbose:

                traceback.print_exc()
        if review_db_updated:
            if args.review_sync_enabled:
                auto_export_review_bundle(
                    review_db_path,
                    args.review_sync_dir,
                    hash_assets=bool(args.review_sync_hash_assets),
                    logger=log,
                )
            else:
                log("Review Git bundle auto-sync disabled by --no-review-sync")

    if args.export_bundle_enabled:
        export_bundle_path = args.export_bundle if args.export_bundle is not None else out_dir / f"{out_dir.name}_bundle_{mag_bin_tag}.zip"
        log(f"\n=== Exporting bundle to {export_bundle_path} ===")
        try:
            if args.full_bundle:
                source_index_file, index_candidates = _resolve_asassn_index_path(out_dir, index_override=args.index_file)
                if source_index_file is None:
                    tried_paths = ", ".join(str(p) for p in index_candidates[:6])
                    if len(index_candidates) > 6:
                        tried_paths += ", ..."
                    if not tried_paths:
                        tried_paths = "(no candidate paths)"
                    raise FileNotFoundError(
                        "Required index file not found for bundle export. "
                        f"Tried: {tried_paths}. "
                        "Pass --index-file or place the index at input/asassn_index_*.parquet."
                    )

                if source_index_file.suffix.lower() not in {".parquet", ".pq"}:
                    raise ValueError(
                        f"--full-bundle requires a parquet ASAS-SN index file, got: {source_index_file}"
                    )

                bundle_assets_dir = out_dir / "bundle_assets"
                bundle_assets_dir.mkdir(parents=True, exist_ok=True)
                bundle_index_file = bundle_assets_dir / "asassn_index_full.parquet"
                if (not bundle_index_file.exists()) or (bundle_index_file.stat().st_size != source_index_file.stat().st_size):
                    log(f"Copying full index into bundle assets: {source_index_file} -> {bundle_index_file}")
                    shutil.copy2(source_index_file, bundle_index_file)

            bundled = export_bundle_zip(export_bundle_path, out_dir, include_all=args.full_bundle, mag_bin_tag=mag_bin_tag)
            log(f"Exported bundle to {export_bundle_path.expanduser()} with {len(bundled)} files")
        except Exception as e:
            print(f"Error creating export bundle: {e}")


if __name__ == "__main__":
    main()
