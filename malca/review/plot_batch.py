"""
Plot light curves for candidates that passed all filters.

Reads the filtered events results and plots only sources with failed_any == False.
"""
from __future__ import annotations

from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
import argparse
import hashlib
import json
import os
import shlex
import sys

from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from malca.config import (
    CLEAN_LC_MAX_ERROR_ABSOLUTE,
    CLEAN_LC_MAX_ERROR_SIGMA,
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
)
from malca.config import (
    JD_OFFSET, LOGBF_THRESHOLD_DIP, LOGBF_THRESHOLD_JUMP,
)
from malca.core.phase import resolve_phase_period
from malca.stv.plot import plot_bayes_results, plot_phase_folded_lightcurve, BASELINE_FUNCTIONS
from malca.review.metadata import REVIEW_METADATA_FIELDS, normalize_vsx_df, normalize_vsx_record
from malca.io.table_io import read_feature_table, write_parquet_table
from malca.products.candidates import coerce_strict_bool_series, select_passing_candidates, validate_candidate_ids
from malca.products.feature_layers import expand_feature_layers
from malca.products.stage_state import file_signature






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
    Load candidates that passed all filters.

    Parameters
    ----------
    filtered_path : Path
        Path to filtered events results Parquet
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
        df = read_feature_table(filtered_path)

    df = expand_feature_layers(df)
    df = normalize_vsx_df(df)

    # Filter to passing candidates
    if require_failed_any_false:
        df = select_passing_candidates(df, require_failed_col=True)

    # Include only rows where all required flags are True
    if require_flags:
        for flag_col in require_flags:
            if flag_col not in df.columns:
                raise ValueError(f"Required plot-selection flag is missing: {flag_col}")
            df = df[coerce_strict_bool_series(df[flag_col], field_name=flag_col)].copy()

    # Exclude rows where any exclude flag is True
    if exclude_flags:
        for flag_col in exclude_flags:
            if flag_col not in df.columns:
                raise ValueError(f"Required exclusion flag is missing: {flag_col}")
            df = df[~coerce_strict_bool_series(df[flag_col], field_name=flag_col)].copy()

    # Optional quantitative periodicity filters
    if min_lsp_power is not None:
        if "lsp_power" in df.columns:
            values = pd.to_numeric(df["lsp_power"], errors="coerce")
            df = df[values.fillna(-np.inf) >= float(min_lsp_power)].copy()
        else:
            raise ValueError("Requested --min-lsp-power but lsp_power is missing")

    if max_lsp_bootstrap_sig is not None:
        sig_col = "periodicity_bootstrap_sig" if "periodicity_bootstrap_sig" in df.columns else "lsp_bootstrap_sig"
        if sig_col in df.columns:
            values = pd.to_numeric(df[sig_col], errors="coerce")
            df = df[values.fillna(np.inf) <= float(max_lsp_bootstrap_sig)].copy()
        else:
            raise ValueError("Requested periodicity significance cut but no significance field is present")

    if min_periodicity_score is not None:
        if "periodicity_score" in df.columns:
            values = pd.to_numeric(df["periodicity_score"], errors="coerce")
            df = df[values.fillna(-np.inf) >= float(min_periodicity_score)].copy()
        else:
            raise ValueError("Requested --min-periodicity-score but periodicity_score is missing")

    if "candidate_id" not in df.columns:
        raise ValueError("Plot input is missing canonical candidate_id")
    df["candidate_id"] = validate_candidate_ids(df, key_col="candidate_id", require_unique=True)
    lc_col = "lc_path" if "lc_path" in df.columns else "path" if "path" in df.columns else None
    if lc_col is None:
        raise ValueError("Plot input is missing canonical lc_path")
    if bool(df[lc_col].isna().any()) or bool(df[lc_col].astype("string").str.strip().eq("").any()):
        raise ValueError("Plot input contains blank/null light-curve paths")
    normalized_paths = df[lc_col].astype("string").str.strip()
    if bool(normalized_paths.duplicated(keep=False).any()):
        examples = normalized_paths[normalized_paths.duplicated(keep=False)].drop_duplicates().head(5).tolist()
        raise ValueError(f"Plot input contains duplicate light-curve paths: {examples}")

    if max_plots is not None:
        df = df.head(max_plots)

    return df.reset_index(drop=True)


def _plot_single_candidate(args: tuple) -> tuple[str, str, bool, str, str, str | None, bool, str]:
    """Worker function for parallel plotting."""
    (
        lc_path_str, out_path_str, baseline, baseline_kwargs,
        skip_events, plot_fits, logbf_threshold_dip, logbf_threshold_jump,
        jd_offset, clean_max_error_absolute, clean_max_error_sigma,
        detection_results_csv, annotations, metadata, run_params,
        filter_bad_cameras, bad_camera_scatter_ratio,
        phase_plot_ready, phase_period_days, phase_out_path_str,
    ) = args

    lc_path = Path(lc_path_str)
    out_path = Path(out_path_str)

    if not lc_path.exists():
        return (lc_path_str, out_path_str, False, "file not found", "", phase_out_path_str, False, "missing input file")

    try:
        baseline_func = BASELINE_FUNCTIONS[baseline]
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

        phase_success = False
        phase_error = ""
        if phase_plot_ready and phase_out_path_str:
            try:
                plot_phase_folded_lightcurve(
                    lc_path,
                    period_days=float(phase_period_days),
                    out_path=Path(phase_out_path_str),
                    show=False,
                    clean_max_error_absolute=clean_max_error_absolute,
                    clean_max_error_sigma=clean_max_error_sigma,
                    filter_bad_cameras=filter_bad_cameras,
                    bad_camera_scatter_ratio=bad_camera_scatter_ratio,
                )
                phase_success = True
            except Exception as exc:
                phase_success = False
                phase_error = str(exc)

        return (lc_path_str, out_path_str, True, "", filtered_str, phase_out_path_str, phase_success, phase_error)
    except Exception as e:
        return (lc_path_str, out_path_str, False, str(e), "", phase_out_path_str, False, "")


def _as_bool(v: object) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, (float, np.floating)):
        return bool(v) if np.isfinite(float(v)) else False
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


def _optional_bool(v: object) -> bool | None:
    if v is None or v is pd.NA:
        return None
    if isinstance(v, (float, np.floating)) and not np.isfinite(float(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)) and int(v) in {0, 1}:
        return bool(v)
    if isinstance(v, str):
        normalized = v.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n"}:
            return False
    return None


def _candidate_filename_token(candidate_id: str, *, max_length: int = 120) -> str:
    """Return a deterministic, collision-resistant filesystem token."""
    raw = str(candidate_id).strip()
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    if not safe:
        return f"candidate-{digest}"
    if safe != raw or len(safe) > max_length:
        safe = safe[: max(1, max_length - len(digest) - 1)].rstrip("._-") or "candidate"
        return f"{safe}-{digest}"
    return safe


def _resolve_filtered_result(results_dir: Path) -> Path:
    """Resolve one filtered product without filesystem-order ambiguity."""
    canonical = results_dir / "lc_events_filtered.parquet"
    if canonical.exists():
        return canonical
    candidates = sorted(set(results_dir.glob("*_filtered.parquet")) | set(results_dir.glob("*filtered*.parquet")))
    if not candidates:
        raise FileNotFoundError(f"No filtered results found in {results_dir}")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple filtered products found in {results_dir}; pass --input explicitly: "
            f"{[path.name for path in candidates]}"
        )
    return candidates[0]


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


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
    baseline: str | None = None,
    baseline_kwargs: dict | None = None,
    skip_events: bool = False,
    plot_fits: bool = False,
    format: str = "png",
    show: bool = False,
    verbose: bool = False,
    workers: int = 1,
    logbf_threshold_dip: float = LOGBF_THRESHOLD_DIP,
    logbf_threshold_jump: float = LOGBF_THRESHOLD_JUMP,
    jd_offset: float = JD_OFFSET,
    clean_max_error_absolute: float = CLEAN_LC_MAX_ERROR_ABSOLUTE,
    clean_max_error_sigma: float = CLEAN_LC_MAX_ERROR_SIGMA,
    detection_results_csv: Path | None = None,
    run_params: dict | None = None,
    filter_bad_cameras: bool = True,
    bad_camera_scatter_ratio: float = BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    show_tqdm: bool = True,
    allow_partial: bool = False,
) -> dict[str, object]:
    """
    Plot all candidates that passed filters.

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
        Optional detection results Parquet for metadata lookup

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

    if baseline is None:
        baseline = str((run_params or {}).get("baseline_func") or "").strip() or None
    if baseline is None:
        raise ValueError(
            "Audit plots require the detection baseline from run_params or an explicit --baseline"
        )
    if baseline not in BASELINE_FUNCTIONS:
        raise ValueError(f"Unknown detection baseline for plot replay: {baseline}")

    if df.empty:
        print("No passing candidates found")
        return {
            "total_selected": 0,
            "plotted": 0,
            "failed": 0,
            "phase_plotted": 0,
            "phase_failed": 0,
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
    filename_tokens: set[str] = set()
    for _, row in df.iterrows():
        row_dict = normalize_vsx_record({k: row[k] for k in row.index})
        lc_path = Path(row.get("lc_path") or row.get("path"))
        candidate_id = str(row_dict["candidate_id"])
        safe_candidate_id = _candidate_filename_token(candidate_id)
        if safe_candidate_id in filename_tokens:
            raise ValueError(f"Candidate filename token collision: {safe_candidate_id}")
        filename_tokens.add(safe_candidate_id)
        asas_sn_id = str(row_dict.get("asas_sn_id") or lc_path.stem)
        bucket = _candidate_bucket(row)
        out_path = bucket_dirs[bucket] / f"{safe_candidate_id}_candidate.{format}"

        phase_ready_raw = row.get("phase_plot_ready", False)
        phase_period_raw, _phase_source = resolve_phase_period(row_dict)
        try:
            phase_period = float(phase_period_raw)
            phase_ready = bool(phase_ready_raw) and np.isfinite(phase_period) and phase_period > 0
        except Exception:
            phase_period = np.nan
            phase_ready = False
        phase_out_path = bucket_dirs[bucket] / f"{safe_candidate_id}_candidate_phase.{format}" if phase_ready else None

        # Build annotations from filter results
        annotations = {}
        if "dip_bayes_factor" in row.index:
            annotations["dip_logBF"] = f"{row['dip_bayes_factor']:.1f}" if pd.notna(row["dip_bayes_factor"]) else "N/A"
        if "jump_bayes_factor" in row.index:
            annotations["jump_logBF"] = f"{row['jump_bayes_factor']:.1f}" if pd.notna(row["jump_bayes_factor"]) else "N/A"
        if "ruwe" in row.index and pd.notna(row["ruwe"]):
            annotations["RUWE"] = f"{row['ruwe']:.2f}"
        if "catalog_match" in row.index:
            catalog_match = _optional_bool(row["catalog_match"])
            annotations["periodic"] = "Unknown" if catalog_match is None else ("Yes" if catalog_match else "No")
        if "periodicity_period" in row.index and pd.notna(row["periodicity_period"]):
            annotations["periodicity_period_d"] = f"{row['periodicity_period']:.4f}"
        elif "lsp_period" in row.index and pd.notna(row["lsp_period"]):
            annotations["legacy_lsp_period_d"] = f"{row['lsp_period']:.4f}"
        if "lsp_power" in row.index and pd.notna(row["lsp_power"]):
            annotations["LSP_power"] = f"{row['lsp_power']:.4f}"
        if "periodicity_bootstrap_sig" in row.index and pd.notna(row["periodicity_bootstrap_sig"]):
            annotations["periodicity_boot_sig"] = f"{row['periodicity_bootstrap_sig']:.4g}"
        elif "lsp_bootstrap_sig" in row.index and pd.notna(row["lsp_bootstrap_sig"]):
            annotations["legacy_lsp_boot_sig"] = f"{row['lsp_bootstrap_sig']:.4g}"
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
        if "ra" in row.index and pd.notna(row["ra"]):
            annotations["RA"] = f"{row['ra']:.5f}"
        if "dec" in row.index and pd.notna(row["dec"]):
            annotations["Dec"] = f"{row['dec']:.5f}"

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
            filter_bad_cameras, bad_camera_scatter_ratio,
            phase_ready, phase_period,
            str(phase_out_path) if phase_out_path else None,
        ))
        manifest_rows.append(
            {
                "candidate_id": candidate_id,
                "asas_sn_id": row_dict.get("asas_sn_id", asas_sn_id),
                "path": str(lc_path),
                "plot_bucket": bucket,
                "plot_path": str(out_path),
                "phase_plot_ready": bool(phase_ready),
                "phase_period_days": phase_period if phase_ready else np.nan,
                "phase_plot_path": str(phase_out_path) if phase_out_path else "",
                "input_signature_json": json.dumps(file_signature(lc_path, content_hash=True), sort_keys=True),
                "plot_config_json": json.dumps(
                    {
                        "baseline": baseline,
                        "baseline_kwargs": baseline_kwargs,
                        "logbf_threshold_dip": logbf_threshold_dip,
                        "logbf_threshold_jump": logbf_threshold_jump,
                        "jd_offset": jd_offset,
                        "clean_max_error_absolute": clean_max_error_absolute,
                        "clean_max_error_sigma": clean_max_error_sigma,
                        "skip_events": bool(skip_events),
                        "plot_fits": bool(plot_fits),
                        "format": str(format),
                        "filter_bad_cameras": bool(filter_bad_cameras),
                        "bad_camera_scatter_ratio": float(bad_camera_scatter_ratio),
                        "phase_plot_ready": bool(phase_ready),
                        "phase_period_days": phase_period if phase_ready else None,
                        "detection_results_signature": (
                            file_signature(detection_results_csv, content_hash=True)
                            if detection_results_csv is not None else None
                        ),
                        "run_params": run_params or {},
                    },
                    sort_keys=True,
                    default=str,
                ),
            }
        )

    n_plotted = 0
    n_failed = 0
    n_phase_plotted = 0
    n_phase_failed = 0
    all_filtered_cameras: dict[str, str] = {}  # path -> filtered cameras string
    results: list[tuple[str, str, bool, str, str, str | None, bool, str]] = []

    if workers > 1:

        actual_workers = min(workers, cpu_count(), len(work_items))
        print(f"Plotting with {actual_workers} workers...")

        with Pool(processes=actual_workers, maxtasksperchild=50) as pool:
            results = list(tqdm(
                pool.imap_unordered(_plot_single_candidate, work_items),
                total=len(work_items),
                desc="Plotting candidates",
                disable=not show_tqdm,
            ))

        for lc_path, _, success, error, filtered_str, _, phase_success, phase_error in results:
            if success:
                n_plotted += 1
                if filtered_str:
                    all_filtered_cameras[lc_path] = filtered_str
            else:
                n_failed += 1
                if verbose:
                    print(f"Failed to plot {lc_path}: {error}")
            if phase_success:
                n_phase_plotted += 1
            elif phase_error:
                n_phase_failed += 1
                if verbose:
                    print(f"Failed phase plot {lc_path}: {phase_error}")
    else:
        for item in tqdm(work_items, desc="Plotting candidates", disable=not show_tqdm):
            lc_path, out_path_str, success, error, filtered_str, phase_out_path, phase_success, phase_error = _plot_single_candidate(item)
            results.append((lc_path, out_path_str, success, error, filtered_str, phase_out_path, phase_success, phase_error))
            if success:
                n_plotted += 1
                if filtered_str:
                    all_filtered_cameras[lc_path] = filtered_str
            else:
                n_failed += 1
                if verbose:
                    print(f"Failed to plot {lc_path}: {error}")
            if phase_success:
                n_phase_plotted += 1
            elif phase_error:
                n_phase_failed += 1
                if verbose:
                    print(f"Failed phase plot {lc_path}: {phase_error}")

    # Report filtered cameras
    if all_filtered_cameras:
        print(f"\nFiltered cameras summary ({len(all_filtered_cameras)} light curves had bad cameras removed):")
        for lc_path, cams in sorted(all_filtered_cameras.items())[:20]:
            print(f"  {Path(lc_path).name}: cameras {cams}")
        if len(all_filtered_cameras) > 20:
            print(f"  ... and {len(all_filtered_cameras) - 20} more")

    print(f"\nGenerated {n_plotted} plots, {n_failed} failed")
    if n_phase_plotted or n_phase_failed:
        print(f"Generated {n_phase_plotted} phase plots, {n_phase_failed} phase failures")
    failed_paths = [lc_path for lc_path, _, success, _, _, _, _, _ in results if not success]

    result_by_path = {lc_path: (success, err) for lc_path, _, success, err, _, _, _, _ in results}
    phase_by_path = {lc_path: (phase_success, phase_err) for lc_path, _, _, _, _, _, phase_success, phase_err in results}
    manifest_df = pd.DataFrame(manifest_rows)
    if not manifest_df.empty:
        manifest_df["plot_success"] = manifest_df["path"].map(lambda p: result_by_path.get(str(p), (False, "missing"))[0])
        manifest_df["plot_error"] = manifest_df["path"].map(lambda p: result_by_path.get(str(p), (False, "missing"))[1])
        manifest_df["phase_plot_success"] = manifest_df["path"].map(lambda p: phase_by_path.get(str(p), (False, ""))[0])
        manifest_df["phase_plot_error"] = manifest_df["path"].map(lambda p: phase_by_path.get(str(p), (False, ""))[1])
        for bucket in ("dip", "jump", "both"):
            bucket_manifest = manifest_df[manifest_df["plot_bucket"] == bucket].copy()
            write_parquet_table(bucket_manifest, out_dir / f"manifest_{bucket}.parquet")
        for record in manifest_df.to_dict("records"):
            if not bool(record.get("plot_success")):
                continue
            plot_path = Path(str(record["plot_path"]))
            provenance = {
                "candidate_id": str(record["candidate_id"]),
                "input_signature": json.loads(str(record["input_signature_json"])),
                "plot_signature": file_signature(plot_path, content_hash=True),
                "plot_config": json.loads(str(record["plot_config_json"])),
                "plot_success": True,
                "phase_plot_success": bool(record.get("phase_plot_success")),
            }
            phase_plot_path = Path(str(record.get("phase_plot_path") or ""))
            if bool(record.get("phase_plot_success")) and str(phase_plot_path):
                provenance["phase_plot_signature"] = file_signature(phase_plot_path, content_hash=True)
            _write_json_atomic(
                plot_path.with_suffix(plot_path.suffix + ".provenance.json"),
                provenance,
            )

    plotted_by_bucket: dict[str, int] = {"dip": 0, "jump": 0, "both": 0}
    if not manifest_df.empty:
        bucket_counts = manifest_df[manifest_df["plot_success"]]["plot_bucket"].value_counts().to_dict()
        for bucket, count in bucket_counts.items():
            plotted_by_bucket[str(bucket)] = int(count)

    summary = {
        "total_selected": total_selected,
        "plotted": n_plotted,
        "failed": n_failed,
        "phase_plotted": n_phase_plotted,
        "phase_failed": n_phase_failed,
        "failed_paths": failed_paths,
        "filtered_camera_sources": len(all_filtered_cameras),
        "filtered_cameras_by_path": all_filtered_cameras,
        "plotted_by_bucket": plotted_by_bucket,
        "manifest_files": {
            "dip": str(out_dir / "manifest_dip.parquet"),
            "jump": str(out_dir / "manifest_jump.parquet"),
            "both": str(out_dir / "manifest_both.parquet"),
        },
    }
    if (n_failed or n_phase_failed) and not allow_partial:
        raise RuntimeError(
            f"Plot batch incomplete: {n_failed} candidate plot failure(s), "
            f"{n_phase_failed} phase plot failure(s). Manifests were written for diagnosis."
        )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Plot light curves for candidates passing all filters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  malca stv-plot --detect-run output/runs/20260128_163911
  malca stv-plot --results results_filtered.parquet --output-dir plots/
  malca stv-plot --detect-run output/runs/20260128_163911 --max-plots 10
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
        help="Path to filtered events results Parquet. Overrides --detect-run.",
    )
    parser.add_argument(
        "--output-dir",
        dest="out_dir",
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
        default=None,
        help="Baseline function to use; defaults to the exact baseline in run_params.json",
    )
    parser.add_argument(
        "--skip-events",
        action="store_true",
        help="Skip event detection, just plot baseline/residuals",
    )
    parser.add_argument(
        "--plot-fits",
        action="store_true",
        help="Overlay the winning residual-space morphology fit",
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
        default=None,
        help="Override the detection-run dip log-BF threshold",
    )
    parser.add_argument(
        "--logbf-threshold-jump",
        type=float,
        default=None,
        help="Override the detection-run jump log-BF threshold",
    )
    parser.add_argument(
        "--jd-offset",
        type=float,
        default=JD_OFFSET,
        help="JD offset for plotting (default: 2458000.0)",
    )
    parser.add_argument(
        "--clean-max-error-absolute",
        type=float,
        default=CLEAN_LC_MAX_ERROR_ABSOLUTE,
        help="Absolute error cutoff for clean_lc (default: 1.0)",
    )
    parser.add_argument(
        "--clean-max-error-sigma",
        type=float,
        default=CLEAN_LC_MAX_ERROR_SIGMA,
        help="Sigma cutoff for clean_lc MAD filter (default: 5.0)",
    )
    parser.add_argument(
        "--results",
        dest="results",
        type=Path,
        default=None,
        help="Optional detection results Parquet for metadata lookup",
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
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Return success despite individual plotting failures (diagnostic use only)",
    )
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
        default=BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
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

        input_path = _resolve_filtered_result(results_dir)
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

    # Load the detection configuration before resolving the replay baseline.
    run_params = None
    if args.detect_run:
        run_params_path = args.detect_run.expanduser() / "run_params.json"
        if run_params_path.exists():
            try:
                with open(run_params_path) as f:
                    run_params = json.load(f)
                print(f"Loaded run params from: {run_params_path}")
            except Exception as e:
                raise ValueError(f"Could not load detection run params from {run_params_path}: {e}") from e

    effective_baseline = args.baseline or str((run_params or {}).get("baseline_func") or "").strip() or None
    if effective_baseline is not None and effective_baseline not in BASELINE_FUNCTIONS:
        raise ValueError(f"Unknown detection baseline in run parameters: {effective_baseline}")
    args.baseline = effective_baseline

    effective_run_params = dict(run_params or {})
    if args.logbf_threshold_dip is None:
        stored_dip_threshold = effective_run_params.get("logbf_threshold_dip")
        args.logbf_threshold_dip = float(
            LOGBF_THRESHOLD_DIP if stored_dip_threshold is None else stored_dip_threshold
        )
    else:
        effective_run_params["logbf_threshold_dip"] = float(args.logbf_threshold_dip)
    if args.logbf_threshold_jump is None:
        stored_jump_threshold = effective_run_params.get("logbf_threshold_jump")
        args.logbf_threshold_jump = float(
            LOGBF_THRESHOLD_JUMP if stored_jump_threshold is None else stored_jump_threshold
        )
    else:
        effective_run_params["logbf_threshold_jump"] = float(args.logbf_threshold_jump)
    if args.baseline is not None:
        effective_run_params["baseline_func"] = args.baseline
    run_params = effective_run_params or None

    # Build explicit overrides; all unspecified hyperparameters are replayed
    # from run_params inside plot_bayes_results.
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
    if gp_params:
        if args.baseline not in {"per_camera_gp", "gp", "gp_masked"}:
            raise ValueError("GP parameters require an explicit GP detection baseline")
        baseline_kwargs.update(gp_params)

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
        detection_results_csv=args.results,
        run_params=run_params,
        filter_bad_cameras=args.filter_bad_cameras,
        bad_camera_scatter_ratio=args.bad_camera_scatter_ratio,
        show_tqdm=not args.no_progress,
        allow_partial=bool(args.allow_partial),
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
            "detection_results": str(args.results) if args.results else None,
            "filter_bad_cameras": args.filter_bad_cameras,
            "bad_camera_scatter_ratio": args.bad_camera_scatter_ratio,
            "show_tqdm": not args.no_progress,
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
