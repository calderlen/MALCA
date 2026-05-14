"""
Tagging filters that run BEFORE events.py.
These checks primarily annotate candidates with `failed_*` flags and keep rows,
so downstream steps can decide how strictly to enforce exclusion.

Filters (ordered by execution speed for efficiency):
1. filter_sparse_lightcurves - remove LCs with insufficient time span or cadence
2. filter_multi_camera - remove single-camera detections
3. attach_vsx_info - annotate known variable stars from VSX

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
- Periodic variable filtering moved to filter.py (expensive LSP, run after event detection).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter, time_ns
import argparse
import uuid

from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from malca.config import (
    MIN_TIME_SPAN, MIN_POINTS_PER_DAY, MIN_CAMERAS,
    VSX_MAX_SEP_ARCSEC, STATS_CHUNK_SIZE,
)
from malca.config import PARQUET_CACHE_COMPRESSION, PARQUET_OUTPUT_COMPRESSION
from malca.config import VSX_CROSSMATCH_PATH
from malca.config import WORKERS
from malca.table_io import read_parquet_table, write_parquet_table
from malca.utils import (
    read_lc_dat2,
    get_id_col,
    compute_time_stats,
    compute_n_cameras,
    compute_field_summary,
    FIELD_SUMMARY_COLUMNS,
    log_rejections,
)






def _compute_stats_for_row(
    asas_sn_id: str,
    dir_path: str,
    compute_time: bool,
    compute_cameras: bool,
    compute_fields: bool = False,
    file_ext: str | None = None,
) -> dict:
    """
    Helper function for parallel processing. Computes requested stats for a single light curve.
    Returns a dict with requested stats.
    """
    result = {"asas_sn_id": asas_sn_id}

    try:
        df_g, df_v = read_lc_dat2(asas_sn_id, dir_path, file_ext=file_ext)
        df_lc = pd.concat([df_g, df_v], ignore_index=True) if not df_g.empty or not df_v.empty else pd.DataFrame()

        if compute_time:
            time_stats = compute_time_stats(df_lc)
            result.update(time_stats)

        if compute_cameras:
            result["n_cameras"] = compute_n_cameras(df_lc)

        if compute_fields:
            result.update(compute_field_summary(df_lc))

    except Exception as e:
        # If there's an error, return default values
        if compute_time:
            result["time_span_days"] = 0.0
            result["points_per_day"] = 0.0
        if compute_cameras:
            result["n_cameras"] = 0
        if compute_fields:
            result.update(compute_field_summary(pd.DataFrame()))

    return result


def _compute_stats_for_batch(
    batch: list[tuple[int, str, str]],
    compute_time: bool,
    compute_cameras: bool,
    compute_fields: bool = False,
    file_ext: str | None = None,
) -> list[tuple[int, dict]]:
    """Compute stats for a small batch of rows in one worker call."""
    return [
        (
            idx,
            _compute_stats_for_row(
                asas_sn_id,
                dir_path,
                compute_time,
                compute_cameras,
                compute_fields,
                file_ext,
            ),
        )
        for idx, asas_sn_id, dir_path in batch
    ]


def _stats_checkpoint_parts_dir(checkpoint_path: Path) -> Path:
    """Directory for incremental stats checkpoint shards."""
    return checkpoint_path.with_name(f"{checkpoint_path.name}.parts")


def _read_stats_checkpoint_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return pd.read_parquet(path)


def _load_stats_checkpoint(
    checkpoint_path: Path,
    columns: list[str],
    *,
    show_tqdm: bool = False,
) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    source_messages: list[str] = []

    if checkpoint_path.exists():
        try:
            frame = _read_stats_checkpoint_frame(checkpoint_path, columns)
            frames.append(frame)
            source_messages.append(str(checkpoint_path))
        except Exception as e:
            if show_tqdm:
                tqdm.write(f"[stats] Warning: Could not load checkpoint {checkpoint_path}: {e}")

    parts_dir = _stats_checkpoint_parts_dir(checkpoint_path)
    if parts_dir.exists():
        part_paths = sorted(parts_dir.glob("part-*.parquet"))
        part_frames = []
        for part_path in part_paths:
            try:
                part_frames.append(_read_stats_checkpoint_frame(part_path, columns))
            except Exception as e:
                if show_tqdm:
                    tqdm.write(f"[stats] Warning: Could not load checkpoint part {part_path}: {e}")
        if part_frames:
            frames.extend(part_frames)
            source_messages.append(f"{len(part_frames)} part file(s) in {parts_dir}")

    if not frames:
        return None

    checkpoint_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if show_tqdm:
        tqdm.write(
            f"[stats] Loaded checkpoint with {len(checkpoint_df)} rows from "
            f"{', '.join(source_messages)}"
        )
    return checkpoint_df


def _iter_batches(
    tasks: list[tuple[int, str, str]],
    batch_size: int,
) -> list[list[tuple[int, str, str]]]:
    return [
        tasks[start : start + batch_size]
        for start in range(0, len(tasks), batch_size)
    ]


def _compute_stats_parallel(
    df: pd.DataFrame,
    id_col: str,
    path_col: str,
    compute_time: bool = False,
    compute_cameras: bool = False,
    compute_fields: bool = False,
    file_ext: str | None = None,
    n_workers: int = 4,
    show_tqdm: bool = False,
    checkpoint_path: str | Path | None = None,
    chunk_size: int = 10000,
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
        Number of rows to process before saving a checkpoint (default 10000).
    """
    df_with_stats = df.copy()

    # Initialize columns
    if compute_time:
        df_with_stats["time_span_days"] = np.nan
        df_with_stats["points_per_day"] = np.nan
    if compute_cameras:
        df_with_stats["n_cameras"] = np.nan
    if compute_fields:
        for col in FIELD_SUMMARY_COLUMNS:
            df_with_stats[col] = "" if col.endswith("_key") or col.endswith("_fields") else np.nan

    stats_cols: list[str] = []
    if compute_time:
        stats_cols.extend(["time_span_days", "points_per_day"])
    if compute_cameras:
        stats_cols.append("n_cameras")
    if compute_fields:
        stats_cols.extend([col for col in FIELD_SUMMARY_COLUMNS if col not in stats_cols])

    checkpoint_cols = [id_col] + stats_cols

    # Load checkpoint if exists. New progress is stored in sidecar part files so
    # large legacy checkpoints do not get rewritten on every save.
    checkpoint_df = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_df = _load_stats_checkpoint(
            checkpoint_path,
            checkpoint_cols,
            show_tqdm=show_tqdm,
        )

    # Merge checkpoint data if available
    already_computed = set()
    if checkpoint_df is not None and id_col in checkpoint_df.columns:
        available_cols = [col for col in checkpoint_cols if col in checkpoint_df.columns]
        checkpoint_df = checkpoint_df.loc[:, available_cols]
        checkpoint_df[id_col] = checkpoint_df[id_col].astype(str)

        required_complete_cols: list[str] = []
        if compute_time:
            required_complete_cols.extend(["time_span_days", "points_per_day"])
        if compute_cameras:
            required_complete_cols.append("n_cameras")
        if not required_complete_cols and compute_fields:
            required_complete_cols.extend([col for col in FIELD_SUMMARY_COLUMNS if col in checkpoint_df.columns])

        if required_complete_cols and all(col in checkpoint_df.columns for col in required_complete_cols):
            complete_mask = checkpoint_df[required_complete_cols].notna().all(axis=1)
            checkpoint_complete = checkpoint_df.loc[complete_mask].drop_duplicates(subset=[id_col], keep="last")
        else:
            checkpoint_complete = checkpoint_df.drop_duplicates(subset=[id_col], keep="last")

        already_computed = set(checkpoint_complete[id_col].unique())

        update_cols = [col for col in stats_cols if col in checkpoint_complete.columns]
        if update_cols:
            # Merge checkpoint values into df_with_stats
            df_with_stats[id_col] = df_with_stats[id_col].astype(str)
            checkpoint_subset = checkpoint_complete[[id_col] + update_cols]
            df_with_stats = df_with_stats.drop(columns=update_cols, errors='ignore')
            df_with_stats = df_with_stats.merge(checkpoint_subset, on=id_col, how='left')

    # Prepare tasks for rows not yet computed
    df_with_stats[id_col] = df_with_stats[id_col].astype(str)
    pending_mask = ~df_with_stats[id_col].isin(already_computed) if already_computed else pd.Series(True, index=df_with_stats.index)
    pending = df_with_stats.loc[pending_mask, [id_col, path_col]]
    tasks = list(
        zip(
            pending.index.tolist(),
            pending[id_col].astype(str).tolist(),
            pending[path_col].astype(str).tolist(),
        )
    )

    if not tasks:
        if show_tqdm:
            tqdm.write(f"[stats] All {len(df)} rows already computed from checkpoint")
        return df_with_stats

    if show_tqdm:
        tqdm.write(f"[stats] {len(already_computed)} rows from checkpoint, {len(tasks)} remaining to compute")

    # Process in chunks with checkpoint saves
    pbar = tqdm(total=len(tasks), desc="Computing stats (parallel)", leave=False, disable=not show_tqdm)

    def apply_results(result_rows: list[tuple[int, dict]]) -> None:
        if not result_rows:
            return
        records = []
        indices = []
        for idx, result in result_rows:
            record = {col: result.get(col) for col in stats_cols}
            records.append(record)
            indices.append(idx)

        result_df = pd.DataFrame.from_records(records, index=indices)
        for col in result_df.columns:
            df_with_stats.loc[result_df.index, col] = result_df[col].values

    def save_checkpoint(result_rows: list[tuple[int, dict]]) -> None:
        """Save newly computed progress to an incremental checkpoint part."""
        if checkpoint_path is None or not result_rows:
            return

        records = []
        for _, result in result_rows:
            record = {id_col: str(result["asas_sn_id"])}
            for col in stats_cols:
                record[col] = result.get(col)
            records.append(record)

        df_checkpoint = pd.DataFrame.from_records(records, columns=checkpoint_cols)
        df_checkpoint = df_checkpoint.drop_duplicates(subset=[id_col], keep="last")

        parts_dir = _stats_checkpoint_parts_dir(checkpoint_path)
        parts_dir.mkdir(parents=True, exist_ok=True)
        part_id = f"{time_ns()}-{uuid.uuid4().hex}"
        tmp_path = parts_dir / f".part-{part_id}.tmp"
        final_path = parts_dir / f"part-{part_id}.parquet"
        try:
            df_checkpoint.to_parquet(tmp_path, index=False, compression=PARQUET_CACHE_COMPRESSION)
            tmp_path.replace(final_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    task_batch_size = max(1, min(chunk_size, max(32, min(1000, chunk_size // max(1, n_workers)))))

    def process_chunk(
        chunk_tasks: list[tuple[int, str, str]],
        executor: ProcessPoolExecutor | None = None,
    ) -> list[tuple[int, dict]]:
        chunk_results: list[tuple[int, dict]] = []
        task_batches = _iter_batches(chunk_tasks, task_batch_size)

        if executor is None:
            for batch in task_batches:
                batch_results = _compute_stats_for_batch(batch, compute_time, compute_cameras, compute_fields, file_ext)
                apply_results(batch_results)
                chunk_results.extend(batch_results)
                pbar.update(len(batch_results))
            return chunk_results

        futures = [
            executor.submit(
                _compute_stats_for_batch,
                batch,
                compute_time,
                compute_cameras,
                compute_fields,
                file_ext,
            )
            for batch in task_batches
        ]
        for future in as_completed(futures):
            batch_results = future.result()
            apply_results(batch_results)
            chunk_results.extend(batch_results)
            pbar.update(len(batch_results))
        return chunk_results

    def finish_chunk(chunk_end: int, chunk_results: list[tuple[int, dict]]) -> None:
        if checkpoint_path is None:
            return

        # Save only new rows after each chunk. Existing checkpoint rows are read
        # on resume and are not rewritten.
        save_checkpoint(chunk_results)
        parts_dir = _stats_checkpoint_parts_dir(checkpoint_path)
        if show_tqdm:
            tqdm.write(f"[stats] Checkpoint part saved: {chunk_end}/{len(tasks)} rows processed in {parts_dir}")

    try:
        # Submit work in chunks to bound memory use and checkpoint periodically.
        if n_workers <= 1:
            for chunk_start in range(0, len(tasks), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(tasks))
                chunk_results = process_chunk(tasks[chunk_start:chunk_end])
                finish_chunk(chunk_end, chunk_results)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for chunk_start in range(0, len(tasks), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(tasks))
                    chunk_results = process_chunk(tasks[chunk_start:chunk_end], executor)
                    finish_chunk(chunk_end, chunk_results)
    finally:
        pbar.close()


    return df_with_stats


def filter_sparse_lightcurves(
    df: pd.DataFrame,
    *,
    min_time_span: float = 100.0,
    min_points_per_day: float = 0.05,
    file_ext: str | None = None,
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

            df_g, df_v = read_lc_dat2(asas_sn_id, dir_path, file_ext=file_ext)
            df_lc = pd.concat([df_g, df_v], ignore_index=True) if not df_g.empty or not df_v.empty else pd.DataFrame()

            stats = compute_time_stats(df_lc)
            df_with_stats.loc[idx, "time_span_days"] = stats["time_span_days"]
            df_with_stats.loc[idx, "points_per_day"] = stats["points_per_day"]
            for col, value in compute_field_summary(df_lc).items():
                df_with_stats.loc[idx, col] = value

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
    vsx_crossmatch_csv: str | Path | None = VSX_CROSSMATCH_PATH,
) -> pd.DataFrame:
    """
    Attach VSX crossmatch info (vsx_sep_arcsec/vsx_class) to the dataframe.

    Uses the provided crossmatch Parquet, or requires vsx_sep_arcsec/vsx_class columns.
    """
    if "vsx_sep_arcsec" in df.columns and "vsx_class" in df.columns:
        return df
    if vsx_crossmatch_csv is None:
        raise ValueError("vsx_crossmatch_csv is required to attach VSX info.")

    vsx_crossmatch_csv = Path(vsx_crossmatch_csv)
    xmatch = read_parquet_table(vsx_crossmatch_csv)
    rename_map = {}
    if "vsx_sep_arcsec" not in xmatch.columns and "sep_arcsec" in xmatch.columns:
        rename_map["sep_arcsec"] = "vsx_sep_arcsec"
    if "vsx_class" not in xmatch.columns and "class" in xmatch.columns:
        rename_map["class"] = "vsx_class"
    if rename_map:
        xmatch = xmatch.rename(columns=rename_map)

    missing_cols = [c for c in ("asas_sn_id", "vsx_sep_arcsec", "vsx_class") if c not in xmatch.columns]
    if missing_cols:
        raise ValueError(
            f"VSX crossmatch file {vsx_crossmatch_csv} is missing required columns {missing_cols}. "
            f"Found columns: {list(xmatch.columns)}"
        )

    xmatch = xmatch[["asas_sn_id", "vsx_sep_arcsec", "vsx_class"]]
    xmatch["asas_sn_id"] = xmatch["asas_sn_id"].astype(str)
    id_col = get_id_col(df)
    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    df = df.merge(xmatch, left_on=id_col, right_on="asas_sn_id", how="left", suffixes=("", "_vsx"))
    if id_col != "asas_sn_id" and "asas_sn_id_vsx" in df.columns:
        df = df.drop(columns=["asas_sn_id_vsx"], errors="ignore")
    return df


def filter_multi_camera(
    df: pd.DataFrame,
    *,
    min_cameras: int = 2,
    file_ext: str | None = None,
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

            df_g, df_v = read_lc_dat2(asas_sn_id, dir_path, file_ext=file_ext)
            df_lc = pd.concat([df_g, df_v], ignore_index=True) if not df_g.empty or not df_v.empty else pd.DataFrame()

            n_cams = compute_n_cameras(df_lc)
            df_with_cameras.loc[idx, "n_cameras"] = n_cams
            for col, value in compute_field_summary(df_lc).items():
                df_with_cameras.loc[idx, col] = value

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


def filter_mag_range(
    df: pd.DataFrame,
    *,
    mag_lo: float = 10.0,
    mag_hi: float = 18.0,
    show_tqdm: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Remove sources whose mag_bin falls entirely outside the [mag_lo, mag_hi] range.

    Uses the mag_bin column (e.g. '13_13.5') to determine source magnitude.
    A source is kept if its mag_bin overlaps with [mag_lo, mag_hi].
    """
    n0 = len(df)
    if "mag_bin" not in df.columns:
        if show_tqdm:
            tqdm.write("[filter_mag_range] no mag_bin column; skipping")
        return df

    def _in_range(mb: str) -> bool:
        parsed = _parse_mag_bin_range(mb)
        if parsed is None:
            return True  # keep rows with unparseable mag_bin
        lo, hi = parsed
        return hi >= mag_lo and lo <= mag_hi

    mask = df["mag_bin"].astype(str).map(_in_range)
    out = df.loc[mask].reset_index(drop=True)

    if show_tqdm:
        tqdm.write(f"[filter_mag_range] kept {len(out)}/{n0} (range {mag_lo}-{mag_hi})")
    log_rejections(df, out, "filter_mag_range", rejected_log_csv)

    return out


# =============================================================================
# Filter: Camera Median Validation
# =============================================================================

RAW2_COLUMNS = ["camera", "median", "sig1_low", "sig1_high", "pct90_low", "pct90_high"]
RAW_MEDIAN_SUSPECT_COL = "raw_median_suspect_cameras"


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


def _process_camera_median_row(
    asas_sn_id: str,
    path_str: str,
    mag_bin: str,
    mag_tolerance: float
) -> tuple[str, str]:
    """
    Helper for parallel camera median filtering.
    """
    try:
        path = Path(path_str)
        if path.is_dir():
            return asas_sn_id, ""

        raw2_path = path.with_suffix(".raw2")
        stats = _read_raw2_camera_stats(raw2_path)

        if stats.empty:
            return asas_sn_id, ""

        mag_range = _parse_mag_bin_range(mag_bin)
        if mag_range is None:
            return asas_sn_id, ""

        mag_min = mag_range[0] - mag_tolerance
        mag_max = mag_range[1] + mag_tolerance

        suspect_cameras = stats[
            (stats["median"] < mag_min) | (stats["median"] > mag_max)
        ]["camera"].astype(int).tolist()

        if suspect_cameras:
            return asas_sn_id, ",".join(map(str, suspect_cameras))
        
        return asas_sn_id, ""
    except Exception:
        return asas_sn_id, ""


def filter_camera_medians(
    df: pd.DataFrame,
    *,
    mag_tolerance: float = 0.2,
    show_tqdm: bool = False,
    n_workers: int = 1,
    rejected_log_csv: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    chunk_size: int = 10000,
) -> pd.DataFrame:
    """
    Identify cameras with raw median magnitudes outside the expected mag bin range.
    Supports parallel execution and checkpointing.

    Adds a raw-space suspect-camera column. These cameras are not excluded here;
    event detection makes final camera removal decisions in residual space.
    """
    if "path" not in df.columns:
        raise ValueError("Need 'path' column to find .raw2 files")
    if "mag_bin" not in df.columns:
        raise ValueError("Need 'mag_bin' column to determine expected magnitude range")

    id_col = get_id_col(df)
    
    # Initialize results container
    df_out = df.copy()
    
    # Load checkpoint if exists
    checkpoint_df = None
    already_processed = set()
    
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            try:
                checkpoint_df = pd.read_parquet(checkpoint_path)
                if id_col in checkpoint_df.columns and RAW_MEDIAN_SUSPECT_COL in checkpoint_df.columns:
                    checkpoint_df[id_col] = checkpoint_df[id_col].astype(str)
                    already_processed = set(checkpoint_df[id_col])
                    if show_tqdm:
                        tqdm.write(f"[filter_camera_medians] Loaded checkpoint with {len(checkpoint_df)} rows")
                else:
                    checkpoint_df = None
                    if show_tqdm:
                        tqdm.write("[filter_camera_medians] Ignoring incompatible checkpoint without raw suspect cameras")
            except Exception as e:
                if show_tqdm:
                    tqdm.write(f"[filter_camera_medians] Warning: Could not load checkpoint: {e}")

    # Prepare tasks
    tasks = []
    df_out[id_col] = df_out[id_col].astype(str)
    
    for idx, row in df_out.iterrows():
        asas_sn_id = str(row[id_col])
        if asas_sn_id in already_processed:
            continue
            
        path_str = str(row["path"])
        mag_bin = str(row["mag_bin"])
        tasks.append((asas_sn_id, path_str, mag_bin))

    # If everything is checkpointed, just merge and return
    if not tasks and checkpoint_df is not None:
        if show_tqdm:
            tqdm.write("[filter_camera_medians] All rows found in checkpoint.")
        
        # Merge checkpoint results
        checkpoint_subset = checkpoint_df[[id_col, RAW_MEDIAN_SUSPECT_COL]].drop_duplicates(subset=[id_col])
        df_out = df_out.merge(checkpoint_subset, on=id_col, how="left")
        
        # Fill NaN with empty string
        if RAW_MEDIAN_SUSPECT_COL in df_out.columns:
            df_out[RAW_MEDIAN_SUSPECT_COL] = df_out[RAW_MEDIAN_SUSPECT_COL].fillna("")
        else:
            df_out[RAW_MEDIAN_SUSPECT_COL] = ""
             
        return df_out

    # Container for new results
    new_results = []
    
    if show_tqdm:
        tqdm.write(f"[filter_camera_medians] Processing {len(tasks)} rows with {n_workers} workers")

    # Function to save checkpoint
    def save_checkpoint(current_results):
        if checkpoint_path is None:
            return
            
        new_df = pd.DataFrame(current_results, columns=[id_col, RAW_MEDIAN_SUSPECT_COL])
        
        if checkpoint_df is not None:
            combined = pd.concat([checkpoint_df, new_df], ignore_index=True)
        else:
            combined = new_df
            
        combined = combined.drop_duplicates(subset=[id_col], keep="last")
        
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(checkpoint_path, index=False, compression=PARQUET_CACHE_COMPRESSION)

    # Run in parallel
    pbar = tqdm(total=len(tasks), desc="filter_camera_medians", leave=False, disable=not show_tqdm)
    
    # If n_workers is 1, run sequentially to avoid overhead
    if n_workers <= 1:
        for item in tasks:
            tid, tpath, tbin = item
            res_id, res_str = _process_camera_median_row(tid, tpath, tbin, mag_tolerance)
            new_results.append((res_id, res_str))
            pbar.update(1)
            
            if len(new_results) % chunk_size == 0:
                save_checkpoint(new_results)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit in chunks
            for i in range(0, len(tasks), chunk_size):
                chunk = tasks[i : i + chunk_size]
                futures = [
                    executor.submit(_process_camera_median_row, tid, tpath, tbin, mag_tolerance)
                    for tid, tpath, tbin in chunk
                ]
                
                chunk_results = []
                for future in as_completed(futures):
                    res_id, res_str = future.result()
                    chunk_results.append((res_id, res_str))
                    pbar.update(1)
                
                new_results.extend(chunk_results)
                save_checkpoint(new_results)

    pbar.close()
    
    # Final save
    save_checkpoint(new_results)
    
    # Final merge
    new_results_df = pd.DataFrame(new_results, columns=[id_col, RAW_MEDIAN_SUSPECT_COL])
    
    if checkpoint_df is not None:
        final_results_df = pd.concat([checkpoint_df, new_results_df], ignore_index=True)
    else:
        final_results_df = new_results_df
        
    final_results_df = final_results_df.drop_duplicates(subset=[id_col], keep="last")
    
    df_out = df_out.drop(columns=[RAW_MEDIAN_SUSPECT_COL], errors="ignore")
    df_out = df_out.merge(final_results_df, on=id_col, how="left")
    df_out[RAW_MEDIAN_SUSPECT_COL] = df_out[RAW_MEDIAN_SUSPECT_COL].fillna("")
    
    if show_tqdm:
        n_suspect = sum(1 for x in df_out[RAW_MEDIAN_SUSPECT_COL] if x)
        tqdm.write(f"[filter_camera_medians] {n_suspect}/{len(df_out)} sources have raw median suspect cameras")
        
    return df_out


def apply_tags(

    df: pd.DataFrame,
    *,
    # Filter 1: VSX crossmatch
    apply_vsx: bool = False,
    vsx_max_sep_arcsec: float = 3.0,
    vsx_exclude_classes: list[str] | None = None,
    vsx_crossmatch_csv: str | Path = VSX_CROSSMATCH_PATH,
    # Filter 2: sparse lightcurves
    apply_sparse: bool = True,
    min_time_span: float = 100.0,
    min_points_per_day: float = 0.05,
    # Filter 3: multi camera
    apply_multi_camera: bool = True,
    min_cameras: int = 2,
    # Filter 4: magnitude range
    apply_mag_range: bool = True,
    mag_lo: float = 10.0,
    mag_hi: float = 18.0,
    file_ext: str | None = None,
    # General
    n_workers: int = 1,
    show_tqdm: bool = True,
    rejected_log_csv: str | Path | None = "rejected_tag.parquet",
    # Checkpoint for stats computation
    stats_checkpoint: str | Path | None = None,
    stats_chunk_size: int = STATS_CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Apply tagging filters before running events.py.

    Filters are applied in order of execution speed (fast to slow) for efficiency.
    Note: Periodic variable filtering moved to filter.py (expensive, run after event detection).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with ID, path, and astrometry columns (ra_deg, dec_deg, pm_ra, pm_dec)
    apply_* : bool
        Whether to apply each filter
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
        Number of rows to process before saving checkpoint (default from STATS_CHUNK_SIZE).

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
    if "path" in df_filtered.columns:
        id_col = get_id_col(df_filtered)

        compute_time = apply_sparse
        compute_cameras = apply_multi_camera
        compute_fields = compute_time or compute_cameras

        if compute_time or compute_cameras or compute_fields:
            if show_tqdm:
                tqdm.write(f"[apply_tags] Pre-computing stats with {n_workers} workers")
            df_filtered = _compute_stats_parallel(
                df_filtered, id_col, "path",
                compute_time=compute_time,
                compute_cameras=compute_cameras,
                compute_fields=compute_fields,
                file_ext=file_ext,
                n_workers=n_workers,
                show_tqdm=show_tqdm,
                checkpoint_path=stats_checkpoint,
                chunk_size=stats_chunk_size,
            )
            precomputed_time = compute_time
            precomputed_cameras = compute_cameras

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
            "file_ext": file_ext,
            "show_tqdm": show_tqdm,
            "compute_stats": not precomputed_time,
            "rejected_log_csv": rejected_log_csv,
        }))

    # Filter 2: Multi camera - cheap, reads dat2 files
    if apply_multi_camera:
        filters.append(("multi_camera", filter_multi_camera, {
            "min_cameras": min_cameras,
            "file_ext": file_ext,
            "show_tqdm": show_tqdm,
            "compute_stats": not precomputed_cameras,
            "rejected_log_csv": rejected_log_csv,
        }))

    # Filter 3: Magnitude range - instant, uses mag_bin column
    if apply_mag_range:
        filters.append(("mag_range", filter_mag_range, {
            "mag_lo": mag_lo,
            "mag_hi": mag_hi,
            "show_tqdm": show_tqdm,
            "rejected_log_csv": rejected_log_csv,
        }))

    # Apply filters and tag failures (all rows kept)
    id_col = get_id_col(df_filtered)
    total_steps = len(filters)
    if total_steps > 0:
        with tqdm(total=total_steps, desc="apply_tags", leave=True, disable=not show_tqdm) as pbar:
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
        tqdm.write(f"\n[apply_tags] {n_failed_any}/{n_start} failed at least one filter")

    return df_filtered.reset_index(drop=True)



def main() -> None:


    parser = argparse.ArgumentParser(description="Apply tagging filters to candidate/source table")
    parser.add_argument("--input", type=Path, required=True, help="Input Parquet")
    parser.add_argument("--output", type=Path, required=True, help="Output Parquet")

    parser.add_argument("--apply-vsx", action="store_true", help="Enable VSX-based filtering/tagging")
    parser.add_argument("--vsx-max-sep-arcsec", type=float, default=VSX_MAX_SEP_ARCSEC)
    parser.add_argument("--vsx-crossmatch", type=Path, default=VSX_CROSSMATCH_PATH)

    parser.add_argument("--skip-sparse", dest="apply_sparse", action="store_false", help="Disable sparse-LC filter")
    parser.add_argument("--min-time-span", type=float, default=MIN_TIME_SPAN)
    parser.add_argument("--min-points-per-day", type=float, default=MIN_POINTS_PER_DAY)

    parser.add_argument("--skip-multi-camera", dest="apply_multi_camera", action="store_false", help="Disable multi-camera filter")
    parser.add_argument("--min-cameras", type=int, default=MIN_CAMERAS)

    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--extension", "-e", type=str, default=None,
                        help="Light curve file extension (e.g., dat, dat2, dat3). Default comes from config.")
    parser.add_argument("--stats-checkpoint", type=Path, default=None)
    parser.add_argument("--stats-chunk-size", type=int, default=STATS_CHUNK_SIZE)
    parser.add_argument("--rejected-log", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.set_defaults(apply_sparse=True, apply_multi_camera=True)

    args = parser.parse_args()

    input_path = args.input.expanduser()
    output_path = args.output.expanduser()
    df = read_parquet_table(input_path)

    out = apply_tags(
        df,
        apply_vsx=args.apply_vsx,
        vsx_max_sep_arcsec=args.vsx_max_sep_arcsec,
        vsx_crossmatch_csv=args.vsx_crossmatch,
        apply_sparse=args.apply_sparse,
        min_time_span=args.min_time_span,
        min_points_per_day=args.min_points_per_day,
        apply_multi_camera=args.apply_multi_camera,
        min_cameras=args.min_cameras,
        file_ext=args.extension,
        n_workers=args.workers,
        show_tqdm=not args.no_progress,
        rejected_log_csv=args.rejected_log,
        stats_checkpoint=args.stats_checkpoint,
        stats_chunk_size=args.stats_chunk_size,
    )

    write_parquet_table(out, output_path)

    print(f"Saved tag output: {output_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
