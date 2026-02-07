"""
Pre-filters that run BEFORE events.py.
These filters depend only on raw light curve data (dat2 files) and external catalogs.

Filters (ordered by execution speed for efficiency):
1. filter_sparse_lightcurves - remove LCs with insufficient time span or cadence
2. filter_multi_camera - remove single-camera detections
3. filter_vsx_match - optionally remove known variable stars from VSX

Input format:
    DataFrame with columns: asas_sn_id (or id/source_id), path (to directory containing dat2 files)

    Index files provide astrometry: ra_deg, dec_deg, pm_ra, pm_dec

    Filters compute required stats from dat2 files on-the-fly:
    - time_span_days, points_per_day (computed from JD column)
    - vsx_sep_arcsec, vsx_class (attached via VSX crossmatch)
    - n_cameras (counted from camera# column)

Note:
- Bright nearby star (BNS) filtering is handled upstream by ASAS-SN pipeline.
  LC files are only generated for sources without BNS contamination.
- Periodic variable filtering moved to post_filter.py (expensive LSP, run after event detection).
"""

from __future__ import annotations
from pathlib import Path
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from malca.config.config_parquet import PARQUET_CACHE_COMPRESSION, PARQUET_OUTPUT_COMPRESSION
from malca.utils import (
    read_lc_dat2,
    get_id_col,
    compute_time_stats,
    compute_n_cameras,
)


def _compute_stats_for_row(asas_sn_id: str, dir_path: str, compute_time: bool, compute_cameras: bool) -> dict:
    """
    Helper function for parallel processing. Computes requested stats for a single light curve.
    Returns a dict with requested stats.
    """
    result = {"asas_sn_id": asas_sn_id}

    try:
        df_g, df_v = read_lc_dat2(asas_sn_id, dir_path)
        df_lc = pd.concat([df_g, df_v], ignore_index=True) if not df_g.empty or not df_v.empty else pd.DataFrame()

        if compute_time:
            time_stats = compute_time_stats(df_lc)
            result.update(time_stats)

        if compute_cameras:
            result["n_cameras"] = compute_n_cameras(df_lc)

    except Exception as e:
        # If there's an error, return default values
        if compute_time:
            result["time_span_days"] = 0.0
            result["points_per_day"] = 0.0
        if compute_cameras:
            result["n_cameras"] = 0

    return result


def _compute_stats_parallel(
    df: pd.DataFrame,
    id_col: str,
    path_col: str,
    compute_time: bool = False,
    compute_cameras: bool = False,
    n_workers: int = 4,
    show_tqdm: bool = False,
    checkpoint_path: str | Path | None = None,
    chunk_size: int = 5000,
) -> pd.DataFrame:
    """
    Compute stats for all rows in parallel using ProcessPoolExecutor.
    Returns a copy of df with new columns added.

    Parameters
    ----------
    checkpoint_path : str | Path | None
        Path to parquet file for saving/resuming progress. If provided and file exists,
        already-computed stats will be loaded and only missing rows will be processed.
    chunk_size : int
        Number of rows to process before saving a checkpoint (default 5000).
    """
    df_with_stats = df.copy()

    # Initialize columns
    if compute_time:
        df_with_stats["time_span_days"] = np.nan
        df_with_stats["points_per_day"] = np.nan
    if compute_cameras:
        df_with_stats["n_cameras"] = np.nan

    # Load checkpoint if exists
    checkpoint_df = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            try:
                checkpoint_df = pd.read_parquet(checkpoint_path)
                if show_tqdm:
                    tqdm.write(f"[stats] Loaded checkpoint with {len(checkpoint_df)} rows from {checkpoint_path}")
            except Exception as e:
                if show_tqdm:
                    tqdm.write(f"[stats] Warning: Could not load checkpoint: {e}")

    # Merge checkpoint data if available
    already_computed = set()
    if checkpoint_df is not None and id_col in checkpoint_df.columns:
        # Use vectorized merge instead of row-by-row iteration
        checkpoint_df[id_col] = checkpoint_df[id_col].astype(str)
        already_computed = set(checkpoint_df[id_col].unique())

        # Determine which columns to update
        update_cols = []
        if compute_time and "time_span_days" in checkpoint_df.columns:
            update_cols.extend(["time_span_days", "points_per_day"])
        if compute_cameras and "n_cameras" in checkpoint_df.columns:
            update_cols.append("n_cameras")

        if update_cols:
            # Merge checkpoint values into df_with_stats
            df_with_stats[id_col] = df_with_stats[id_col].astype(str)
            checkpoint_subset = checkpoint_df[[id_col] + update_cols].drop_duplicates(subset=[id_col])
            df_with_stats = df_with_stats.drop(columns=update_cols, errors='ignore')
            df_with_stats = df_with_stats.merge(checkpoint_subset, on=id_col, how='left')

    # Prepare tasks for rows not yet computed
    tasks = []
    for idx, row in df.iterrows():
        asas_sn_id = str(row[id_col])
        if asas_sn_id in already_computed:
            continue
        dir_path = str(row[path_col])
        tasks.append((idx, asas_sn_id, dir_path))

    if not tasks:
        if show_tqdm:
            tqdm.write(f"[stats] All {len(df)} rows already computed from checkpoint")
        return df_with_stats

    if show_tqdm:
        tqdm.write(f"[stats] {len(already_computed)} rows from checkpoint, {len(tasks)} remaining to compute")

    # Process in chunks with checkpoint saves
    pbar = tqdm(total=len(tasks), desc="Computing stats (parallel)", leave=False, disable=not show_tqdm)

    results_buffer = []  # Buffer results for checkpoint saves

    def save_checkpoint():
        """Save current progress to checkpoint file."""
        if checkpoint_path is None:
            return

        # Build checkpoint dataframe from df_with_stats (rows with non-NaN stats)
        if compute_time:
            mask = df_with_stats["time_span_days"].notna()
        elif compute_cameras:
            mask = df_with_stats["n_cameras"].notna()
        else:
            return

        checkpoint_cols = [id_col]
        if compute_time:
            checkpoint_cols.extend(["time_span_days", "points_per_day"])
        if compute_cameras:
            checkpoint_cols.append("n_cameras")

        df_checkpoint = df_with_stats.loc[mask, checkpoint_cols].drop_duplicates(subset=[id_col])

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        df_checkpoint.to_parquet(checkpoint_path, index=False, compression=PARQUET_CACHE_COMPRESSION)

    processed_since_save = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit tasks in chunks to avoid memory issues with millions of futures
        for chunk_start in range(0, len(tasks), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(tasks))
            chunk_tasks = tasks[chunk_start:chunk_end]

            futures = {
                executor.submit(_compute_stats_for_row, asas_sn_id, dir_path, compute_time, compute_cameras): idx
                for idx, asas_sn_id, dir_path in chunk_tasks
            }

            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()

                if compute_time:
                    df_with_stats.loc[idx, "time_span_days"] = result["time_span_days"]
                    df_with_stats.loc[idx, "points_per_day"] = result["points_per_day"]
                if compute_cameras:
                    df_with_stats.loc[idx, "n_cameras"] = result["n_cameras"]

                pbar.update(1)
                processed_since_save += 1

            # Save checkpoint after each chunk
            if checkpoint_path is not None:
                save_checkpoint()
                if show_tqdm:
                    tqdm.write(f"[stats] Checkpoint saved: {chunk_end}/{len(tasks)} rows processed")

    pbar.close()

    # Final checkpoint save
    if checkpoint_path is not None:
        save_checkpoint()
        if show_tqdm:
            tqdm.write(f"[stats] Final checkpoint saved to {checkpoint_path}")

    return df_with_stats


def log_rejections(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    filter_name: str,
    log_csv: str | Path | None,
) -> None:
    """
    Log rejected candidates to a CSV file.
    """
    if log_csv is None:
        return

    # Prefer stable IDs over path (path can be a shared directory).
    id_col = None
    for candidate in ["source_id", "asas_sn_id", "id", "path"]:
        if candidate in df_before.columns:
            id_col = candidate
            break

    if id_col is None:
        return

    before_ids = set(df_before[id_col].astype(str))
    after_ids = set(df_after[id_col].astype(str))
    rejected = sorted(before_ids - after_ids)
    if not rejected:
        return

    log_path = Path(log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df_log = pd.DataFrame({id_col: rejected, "filter": filter_name})
    df_log.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


def filter_sparse_lightcurves(
    df: pd.DataFrame,
    *,
    min_time_span: float = 100.0,
    min_points_per_day: float = 0.05,
    show_tqdm: bool = False,
    compute_stats: bool = True,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Remove candidates with:
    - less than min_time_span days of observation (default 100)
    - less than min_points_per_day on average (default 0.05 = 1 point per 20 days)

    Stats are computed from the dat2 files; set compute_stats=False only if the
    columns were already added upstream.
    """
    n0 = len(df)

    if compute_stats:
        id_col = get_id_col(df)
        path_col = "path" if "path" in df.columns else None

        if path_col is None:
            raise ValueError("Need 'path' column to read dat2 files")

        df_with_stats = df.copy()
        df_with_stats["time_span_days"] = 0.0
        df_with_stats["points_per_day"] = 0.0

        pbar = tqdm(total=len(df), desc="filter_sparse_lightcurves (computing stats)", leave=False, disable=not show_tqdm)
        for idx, row in df.iterrows():
            asas_sn_id = str(row[id_col])
            dir_path = str(row[path_col])

            df_g, df_v = read_lc_dat2(asas_sn_id, dir_path)
            df_lc = pd.concat([df_g, df_v], ignore_index=True) if not df_g.empty or not df_v.empty else pd.DataFrame()

            stats = compute_time_stats(df_lc)
            df_with_stats.loc[idx, "time_span_days"] = stats["time_span_days"]
            df_with_stats.loc[idx, "points_per_day"] = stats["points_per_day"]

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        df = df_with_stats
    else:
        missing_cols = [c for c in ("time_span_days", "points_per_day") if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Set compute_stats=True to compute from dat2.")

    # Apply filter
    mask = (df["time_span_days"] >= min_time_span) & \
           (df["points_per_day"] >= min_points_per_day)
    out = df.loc[mask].reset_index(drop=True)

    if show_tqdm:
        tqdm.write(f"[filter_sparse_lightcurves] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_sparse_lightcurves", rejected_log_csv)

    return out


def attach_vsx_info(
    df: pd.DataFrame,
    *,
    vsx_crossmatch_csv: str | Path | None = "input/vsx/asassn_x_vsx_matches_20250919_2252.csv",
) -> pd.DataFrame:
    """
    Attach VSX crossmatch info (vsx_sep_arcsec/vsx_class) to the dataframe.

    Uses the provided crossmatch CSV, or requires vsx_sep_arcsec/vsx_class columns.
    """
    if "vsx_sep_arcsec" in df.columns and "vsx_class" in df.columns:
        return df
    if vsx_crossmatch_csv is None:
        raise ValueError("vsx_crossmatch_csv is required to attach VSX info.")

    xmatch = pd.read_csv(vsx_crossmatch_csv, usecols=["asas_sn_id", "vsx_sep_arcsec", "vsx_class"])
    xmatch["asas_sn_id"] = xmatch["asas_sn_id"].astype(str)
    id_col = get_id_col(df)
    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    df = df.merge(xmatch, left_on=id_col, right_on="asas_sn_id", how="left", suffixes=("", "_vsx"))
    if id_col != "asas_sn_id" and "asas_sn_id_vsx" in df.columns:
        df = df.drop(columns=["asas_sn_id_vsx"], errors="ignore")
    return df


def filter_vsx_match(
    df: pd.DataFrame,
    *,
    max_sep_arcsec: float = 3.0,
    exclude_classes: list[str] | None = None,
    vsx_crossmatch_csv: str | Path | None = "input/vsx/asassn_x_vsx_matches_20250919_2252.csv",
    show_tqdm: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Filter candidates based on pre-crossmatched VSX results.

    Either:
    - Provide vsx_crossmatch_csv: merges on asas_sn_id to get vsx_sep_arcsec/vsx_class
    - Or have vsx_sep_arcsec/vsx_class columns already in df

    Removes candidates within max_sep_arcsec of a VSX source.
    """
    n0 = len(df)
    df = attach_vsx_info(df, vsx_crossmatch_csv=vsx_crossmatch_csv)

    # Check required columns exist
    if "vsx_sep_arcsec" not in df.columns or "vsx_class" not in df.columns:
        raise ValueError(
            f"Missing required columns for VSX filter: need 'vsx_sep_arcsec' and 'vsx_class'. Got: {list(df.columns)}"
        )

    # Apply filter
    has_match = df["vsx_sep_arcsec"].fillna(999) <= max_sep_arcsec

    if exclude_classes is not None:
        is_excluded_type = df["vsx_class"].fillna("").isin(exclude_classes)
        mask = ~(has_match & is_excluded_type)
    else:
        mask = ~has_match

    out = df.loc[mask].reset_index(drop=True)

    if show_tqdm:
        tqdm.write(f"[filter_vsx_match] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_vsx_match", rejected_log_csv)

    return out


def filter_multi_camera(
    df: pd.DataFrame,
    *,
    min_cameras: int = 2,
    show_tqdm: bool = False,
    compute_stats: bool = True,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Remove candidates that were only detected on one camera.

    Stats are computed from the dat2 files; set compute_stats=False only if the
    column was already added upstream.
    """
    n0 = len(df)

    if compute_stats:
        id_col = get_id_col(df)
        path_col = "path" if "path" in df.columns else None

        if path_col is None:
            raise ValueError("Need 'path' column to read dat2 files")

        df_with_cameras = df.copy()
        df_with_cameras["n_cameras"] = 0

        pbar = tqdm(total=len(df), desc="filter_multi_camera (counting cameras)", leave=False, disable=not show_tqdm)
        for idx, row in df.iterrows():
            asas_sn_id = str(row[id_col])
            dir_path = str(row[path_col])

            df_g, df_v = read_lc_dat2(asas_sn_id, dir_path)
            df_lc = pd.concat([df_g, df_v], ignore_index=True) if not df_g.empty or not df_v.empty else pd.DataFrame()

            n_cams = compute_n_cameras(df_lc)
            df_with_cameras.loc[idx, "n_cameras"] = n_cams

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        df = df_with_cameras
    else:
        if "n_cameras" not in df.columns:
            raise ValueError("Missing required column: n_cameras. Set compute_stats=True to compute from dat2.")

    # Apply filter
    out = df.loc[df["n_cameras"] >= min_cameras].reset_index(drop=True)

    if show_tqdm:
        tqdm.write(f"[filter_multi_camera] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_multi_camera", rejected_log_csv)

    return out


# =============================================================================
# Filter: Camera Median Validation
# =============================================================================

RAW2_COLUMNS = ["camera", "median", "sig1_low", "sig1_high", "pct90_low", "pct90_high"]


def _parse_mag_bin_range(mag_bin: str | None) -> tuple[float, float] | None:
    """Parse magnitude bin string (e.g., '13_13.5') into (min, max) tuple."""
    if not mag_bin:
        return None
    token = mag_bin.strip().replace("-", "_")
    parts = token.split("_")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _read_raw2_camera_stats(raw2_path: Path) -> pd.DataFrame:
    """
    Read .raw2 file with per-camera statistics.

    Format (space-separated):
        camera_id median 1sig_low 1sig_high 90pct_low 90pct_high
    """
    if not raw2_path.exists():
        return pd.DataFrame(columns=RAW2_COLUMNS)

    try:
        df = pd.read_csv(
            raw2_path,
            sep=r"\s+",
            header=None,
            names=RAW2_COLUMNS,
            comment="#",
        )
        for col in ("median", "sig1_low", "sig1_high", "pct90_low", "pct90_high"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df["median"].notna()].reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=RAW2_COLUMNS)


def filter_camera_medians(
    df: pd.DataFrame,
    *,
    mag_tolerance: float = 0.2,
    show_tqdm: bool = False,
    n_workers: int = 1,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Identify cameras with median magnitudes outside the expected mag bin range.

    Adds 'excluded_cameras' column (comma-separated camera IDs to exclude).
    Does NOT reject sources - just marks which cameras to skip during LC loading.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'path' and 'mag_bin' columns
    mag_tolerance : float
        Tolerance beyond mag bin edges (default 0.2). Camera is excluded if
        its median falls outside [mag_min - tolerance, mag_max + tolerance].
    show_tqdm : bool
        Show progress bar
    n_workers : int
        Number of parallel workers (not yet implemented, placeholder)

    Returns
    -------
    pd.DataFrame
        Input dataframe with added 'excluded_cameras' column containing
        comma-separated camera IDs outside the expected magnitude range.

    Notes
    -----
    The .raw2 file format:
        camera_id median 1sig_low 1sig_high 90pct_low 90pct_high

    Example (from guidance):
        1   14.0666       14.0395       14.0981       14.0233 14.1128
        6   12.9477       12.9177       12.9683       12.8972 12.9810

    If mag_bin is 14_14.5, camera 6 (median 12.9) would be excluded.
    """
    if "path" not in df.columns:
        raise ValueError("Need 'path' column to find .raw2 files")
    if "mag_bin" not in df.columns:
        raise ValueError("Need 'mag_bin' column to determine expected magnitude range")

    excluded_cameras_list = []
    n_excluded_total = 0

    paths = df["path"].astype(str).tolist()
    mag_bins = df["mag_bin"].astype(str).tolist()

    iterator = zip(paths, mag_bins)
    if show_tqdm:
        iterator = tqdm(list(iterator), desc="filter_camera_medians", leave=False)

    for path_str, mag_bin in iterator:
        path = Path(path_str)

        # Determine .raw2 path (same as .dat2 but different extension)
        if path.is_dir():
            # path is directory, need source_id to construct filename
            # This case requires the dat2 filename - skip for now
            excluded_cameras_list.append("")
            continue

        raw2_path = path.with_suffix(".raw2")
        stats = _read_raw2_camera_stats(raw2_path)

        if stats.empty:
            excluded_cameras_list.append("")
            continue

        mag_range = _parse_mag_bin_range(mag_bin)
        if mag_range is None:
            excluded_cameras_list.append("")
            continue

        mag_min = mag_range[0] - mag_tolerance
        mag_max = mag_range[1] + mag_tolerance

        # Find cameras with medians outside the expected range
        bad_cameras = stats[
            (stats["median"] < mag_min) | (stats["median"] > mag_max)
        ]["camera"].astype(int).tolist()

        if bad_cameras:
            excluded_cameras_list.append(",".join(map(str, bad_cameras)))
            n_excluded_total += len(bad_cameras)
        else:
            excluded_cameras_list.append("")

    df_out = df.copy()
    df_out["excluded_cameras"] = excluded_cameras_list

    if show_tqdm:
        n_sources_with_exclusions = sum(1 for x in excluded_cameras_list if x)
        tqdm.write(
            f"[filter_camera_medians] {n_sources_with_exclusions}/{len(df)} sources have excluded cameras "
            f"({n_excluded_total} total cameras excluded)"
        )

    return df_out


def apply_pre_filters(

    df: pd.DataFrame,
    *,
    # Filter 1: VSX crossmatch
    apply_vsx: bool = False,
    vsx_max_sep_arcsec: float = 3.0,
    vsx_exclude_classes: list[str] | None = None,
    vsx_crossmatch_csv: str | Path = "input/vsx/asassn_x_vsx_matches_20250919_2252.csv",
    vsx_mode: str = "filter",
    # Filter 2: sparse lightcurves
    apply_sparse: bool = True,
    min_time_span: float = 100.0,
    min_points_per_day: float = 0.05,
    # Filter 3: multi camera
    apply_multi_camera: bool = True,
    min_cameras: int = 2,
    # General
    n_workers: int = 1,
    show_tqdm: bool = True,
    rejected_log_csv: str | Path | None = "rejected_pre_filter.csv",
    # Checkpoint for stats computation
    stats_checkpoint: str | Path | None = None,
    stats_chunk_size: int = 5000,
) -> pd.DataFrame:
    """
    Apply pre-filters before running events.py.

    Filters are applied in order of execution speed (fast to slow) for efficiency.
    Note: Periodic variable filtering moved to post_filter.py (expensive, run after event detection).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with ID, path, and astrometry columns (ra_deg, dec_deg, pm_ra, pm_dec)
    apply_* : bool
        Whether to apply each filter
    vsx_mode : str
        "tag" to attach VSX metadata only, "filter" to remove VSX matches (default)
    n_workers : int
        Number of parallel workers for computing stats (default 1 = sequential).
        Filters 2 and 3 (sparse, multi_camera) can benefit from parallelization.
    show_tqdm : bool
        Show progress bars
    rejected_log_csv : str | Path | None
        Path to log rejected candidates
    stats_checkpoint : str | Path | None
        Path to parquet file for checkpointing stats computation. If provided,
        progress can be resumed if interrupted.
    stats_chunk_size : int
        Number of rows to process before saving checkpoint (default 5000).

    Returns
    -------
    pd.DataFrame
        Full dataframe with added columns:
        - failed_<filter_name>: bool, True if row failed that filter
        - failed_any: bool, True if row failed any filter
    """
    df_filtered = df.copy()
    n_start = len(df_filtered)

    precomputed_time = False
    precomputed_cameras = False

    # Pre-compute stats in parallel if requested and needed
    if n_workers > 1 and "path" in df_filtered.columns:
        id_col = get_id_col(df_filtered)

        compute_time = apply_sparse
        compute_cameras = apply_multi_camera

        if compute_time or compute_cameras:
            if show_tqdm:
                tqdm.write(f"[apply_pre_filters] Pre-computing stats with {n_workers} workers")
            df_filtered = _compute_stats_parallel(
                df_filtered, id_col, "path",
                compute_time=compute_time,
                compute_cameras=compute_cameras,
                n_workers=n_workers,
                show_tqdm=show_tqdm,
                checkpoint_path=stats_checkpoint,
                chunk_size=stats_chunk_size,
            )
            precomputed_time = compute_time
            precomputed_cameras = compute_cameras

    if vsx_mode not in {"tag", "filter"}:
        raise ValueError(f"vsx_mode must be 'tag' or 'filter', got {vsx_mode!r}")

    if apply_vsx:
        df_filtered = attach_vsx_info(
            df_filtered,
            vsx_crossmatch_csv=vsx_crossmatch_csv,
        )

    filters = []

    # Filter 1: Sparse lightcurves - cheap, reads dat2 files
    if apply_sparse:
        filters.append(("sparse", filter_sparse_lightcurves, {
            "min_time_span": min_time_span,
            "min_points_per_day": min_points_per_day,
            "show_tqdm": show_tqdm,
            "compute_stats": not precomputed_time,
            "rejected_log_csv": rejected_log_csv,
        }))

    # Filter 2: Multi camera - cheap, reads dat2 files
    if apply_multi_camera:
        filters.append(("multi_camera", filter_multi_camera, {
            "min_cameras": min_cameras,
            "show_tqdm": show_tqdm,
            "compute_stats": not precomputed_cameras,
            "rejected_log_csv": rejected_log_csv,
        }))

    # Filter 3: VSX crossmatch
    if apply_vsx and vsx_mode == "filter":
        filters.append(("vsx_match", filter_vsx_match, {
            "max_sep_arcsec": vsx_max_sep_arcsec,
            "exclude_classes": vsx_exclude_classes,
            "vsx_crossmatch_csv": None,
            "show_tqdm": show_tqdm,
            "rejected_log_csv": rejected_log_csv,
        }))

    # Apply filters and tag failures (all rows kept)
    id_col = get_id_col(df_filtered)
    total_steps = len(filters)
    if total_steps > 0:
        with tqdm(total=total_steps, desc="apply_pre_filters", leave=True, disable=not show_tqdm) as pbar:
            for label, func, kwargs in filters:
                start = perf_counter()
                # Run filter to identify which rows pass
                kwargs_clean = {k: v for k, v in kwargs.items() if k != "rejected_log_csv"}
                df_passed = func(df_filtered, **kwargs_clean)
                elapsed = perf_counter() - start

                # Determine which rows failed using stable source IDs.
                # Path values can be shared directory paths across many sources.
                passed_ids = set(df_passed[id_col].astype(str))
                failed_mask = ~df_filtered[id_col].astype(str).isin(passed_ids)
                df_filtered[f"failed_{label}"] = failed_mask

                n_failed = int(failed_mask.sum())
                pbar.set_postfix_str(f"{label}: {n_failed}/{n_start} failed ({elapsed:.2f}s)")
                pbar.update(1)

    # Add summary column
    failed_cols = [c for c in df_filtered.columns if c.startswith("failed_")]
    if failed_cols:
        df_filtered["failed_any"] = df_filtered[failed_cols].any(axis=1)

    if show_tqdm:
        n_failed_any = int(df_filtered["failed_any"].sum()) if "failed_any" in df_filtered.columns else 0
        tqdm.write(f"\n[apply_pre_filters] {n_failed_any}/{n_start} failed at least one filter")

    return df_filtered.reset_index(drop=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Apply pre-filters to candidate/source table")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV/Parquet")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV/Parquet")

    parser.add_argument("--apply-vsx", action="store_true", help="Enable VSX-based filtering/tagging")
    parser.add_argument("--vsx-mode", choices=["filter", "tag"], default="filter")
    parser.add_argument("--vsx-max-sep-arcsec", type=float, default=3.0)
    parser.add_argument("--vsx-crossmatch-csv", type=Path, default=Path("input/vsx/asassn_x_vsx_matches_20250919_2252.csv"))

    parser.add_argument("--no-sparse", dest="apply_sparse", action="store_false", help="Disable sparse-LC filter")
    parser.add_argument("--min-time-span", type=float, default=100.0)
    parser.add_argument("--min-points-per-day", type=float, default=0.05)

    parser.add_argument("--no-multi-camera", dest="apply_multi_camera", action="store_false", help="Disable multi-camera filter")
    parser.add_argument("--min-cameras", type=int, default=2)

    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stats-checkpoint", type=Path, default=None)
    parser.add_argument("--stats-chunk-size", type=int, default=5000)
    parser.add_argument("--rejected-log-csv", type=Path, default=None)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.set_defaults(apply_sparse=True, apply_multi_camera=True)

    args = parser.parse_args()

    input_path = args.input.expanduser()
    output_path = args.output.expanduser()
    if input_path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    out = apply_pre_filters(
        df,
        apply_vsx=args.apply_vsx,
        vsx_mode=args.vsx_mode,
        vsx_max_sep_arcsec=args.vsx_max_sep_arcsec,
        vsx_crossmatch_csv=args.vsx_crossmatch_csv,
        apply_sparse=args.apply_sparse,
        min_time_span=args.min_time_span,
        min_points_per_day=args.min_points_per_day,
        apply_multi_camera=args.apply_multi_camera,
        min_cameras=args.min_cameras,
        n_workers=args.workers,
        show_tqdm=not args.no_tqdm,
        rejected_log_csv=args.rejected_log_csv,
        stats_checkpoint=args.stats_checkpoint,
        stats_chunk_size=args.stats_chunk_size,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in (".parquet", ".pq"):
        out.to_parquet(output_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)
    else:
        out.to_csv(output_path, index=False)

    print(f"Saved pre-filter output: {output_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
