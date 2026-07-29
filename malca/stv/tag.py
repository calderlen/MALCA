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
from malca.io.table_io import read_feature_table, read_parquet_table, write_feature_table
from malca.products.stage_state import (
    StageResult,
    assert_reusable_stage_state,
    build_stage_fingerprint,
    read_stage_state,
    write_stage_state,
)
from malca.vsx.metadata import normalize_vsx_match_columns, select_best_vsx_matches
from malca.core.utils import (
    read_lc_dat2,
    get_id_col,
    validate_candidate_ids,
    clean_lc,
    compute_time_stats,
    compute_n_cameras,
    compute_field_summary,
    FIELD_SUMMARY_COLUMNS,
    log_rejections,
)

TAG_STATS_VERSION = 2
TAG_MIN_GOOD_POINTS_PER_CAMERA = 2
TAG_STATS_COLUMNS: tuple[str, ...] = (
    "tag_stats_status",
    "tag_stats_error",
    "tag_stats_version",
    "raw_n_points",
    "clean_n_points",
    "raw_n_cameras",
)

def _compute_stats_for_row(
    asas_sn_id: str,
    dir_path: str,
    compute_time: bool,
    compute_cameras: bool,
    compute_fields: bool = False,
    file_ext: str | None = None,
    lc_source_id: str | None = None,
) -> dict:
    """
    Helper function for parallel processing. Computes requested stats for a single light curve.
    Returns a dict with requested stats.
    """
    result = {
        "asas_sn_id": asas_sn_id,
        "tag_stats_status": "error",
        "tag_stats_error": "",
        "tag_stats_version": TAG_STATS_VERSION,
        "raw_n_points": np.nan,
        "clean_n_points": np.nan,
        "raw_n_cameras": np.nan,
    }

    try:
        source_id = str(lc_source_id if lc_source_id is not None else asas_sn_id)
        df_g, df_v = read_lc_dat2(source_id, dir_path, file_ext=file_ext)
        df_raw = pd.concat([df_g, df_v], ignore_index=True) if not df_g.empty or not df_v.empty else pd.DataFrame()
        df_lc = clean_lc(df_raw) if not df_raw.empty else df_raw.copy()

        result["raw_n_points"] = int(len(df_raw))
        result["clean_n_points"] = int(len(df_lc))
        result["raw_n_cameras"] = compute_n_cameras(df_raw)

        if compute_time:
            time_stats = compute_time_stats(df_lc)
            result.update(time_stats)

        if compute_cameras:
            result["n_cameras"] = compute_n_cameras(
                df_lc,
                min_points=TAG_MIN_GOOD_POINTS_PER_CAMERA,
            )

        if compute_fields:
            result.update(compute_field_summary(df_lc))

        result["tag_stats_status"] = "ok"

    except Exception as e:
        # Missing/error values must not be confused with genuine zero coverage.
        result["tag_stats_error"] = f"{type(e).__name__}: {e}"
        if compute_time:
            result["time_span_days"] = np.nan
            result["points_per_day"] = np.nan
        if compute_cameras:
            result["n_cameras"] = np.nan
        if compute_fields:
            result.update(compute_field_summary(pd.DataFrame()))

    return result


def _compute_stats_for_batch(
    batch: list[tuple[int, str, str, str]],
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
                lc_source_id=lc_source_id,
            ),
        )
        for idx, asas_sn_id, lc_source_id, dir_path in batch
    ]


def _stats_checkpoint_parts_dir(checkpoint_path: Path) -> Path:
    """Directory for incremental stats checkpoint shards."""
    return checkpoint_path.with_name(f"{checkpoint_path.name}.parts")


def _stats_checkpoint_state_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.name}_STAGE.json")


def _stats_checkpoint_outputs(checkpoint_path: Path) -> tuple[Path, Path]:
    """Artifacts whose signatures jointly define resumable tag-stat progress."""
    return checkpoint_path, _stats_checkpoint_parts_dir(checkpoint_path)


def _assert_reusable_stats_checkpoint_state(
    state: dict | None,
    *,
    fingerprint: dict,
    checkpoint_path: Path,
) -> None:
    """Validate provenance and require signatures for both checkpoint layers."""
    assert_reusable_stage_state(
        state,
        fingerprint=fingerprint,
        require_complete=False,
    )
    outputs = state.get("outputs", []) if isinstance(state, dict) else []
    recorded_paths = {
        str(Path(str(signature.get("path"))).expanduser())
        for signature in outputs
        if isinstance(signature, dict) and signature.get("path")
    }
    expected_paths = {
        str(path.expanduser())
        for path in _stats_checkpoint_outputs(checkpoint_path)
    }
    if recorded_paths != expected_paths:
        raise ValueError(
            "Tag-stat stage state must sign both the base checkpoint and its .parts directory"
        )


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
    tasks: list[tuple[int, str, str, str]],
    batch_size: int,
) -> list[list[tuple[int, str, str, str]]]:
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
    source_id_col: str | None = None,
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
    normalized_ids = validate_candidate_ids(df_with_stats, id_col)
    df_with_stats[id_col] = normalized_ids
    if source_id_col is None:
        source_id_col = "asas_sn_id" if "asas_sn_id" in df_with_stats.columns else id_col
    if source_id_col not in df_with_stats.columns:
        raise KeyError(f"Missing light-curve source ID column: {source_id_col}")
    source_ids = df_with_stats[source_id_col].astype("string").str.strip()
    invalid_source = source_ids.isna() | source_ids.str.lower().isin({"", "nan", "none", "null", "<na>"})
    if bool(invalid_source.any()):
        raise ValueError(
            f"Light-curve source ID column '{source_id_col}' contains "
            f"{int(invalid_source.sum())} invalid value(s)"
        )
    df_with_stats[source_id_col] = source_ids

    # Initialize columns
    if compute_time:
        df_with_stats["time_span_days"] = np.nan
        df_with_stats["points_per_day"] = np.nan
    if compute_cameras:
        df_with_stats["n_cameras"] = np.nan
    if compute_fields:
        for col in FIELD_SUMMARY_COLUMNS:
            df_with_stats[col] = "" if col.endswith("_key") or col in {"asassn_fields", "camera_names"} else np.nan
    for col in TAG_STATS_COLUMNS:
        if col == "tag_stats_status" or col == "tag_stats_error":
            df_with_stats[col] = ""
        else:
            df_with_stats[col] = np.nan

    stats_cols: list[str] = []
    if compute_time:
        stats_cols.extend(["time_span_days", "points_per_day"])
    if compute_cameras:
        stats_cols.append("n_cameras")
    if compute_fields:
        stats_cols.extend([col for col in FIELD_SUMMARY_COLUMNS if col not in stats_cols])
    stats_cols.extend([col for col in TAG_STATS_COLUMNS if col not in stats_cols])

    checkpoint_cols = [id_col] + stats_cols

    # Load checkpoint if exists. New progress is stored in sidecar part files so
    # large legacy checkpoints do not get rewritten on every save.
    checkpoint_df = None
    checkpoint_fingerprint = None
    checkpoint_state_path = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_state_path = _stats_checkpoint_state_path(checkpoint_path)
        from malca.config import LIGHT_CURVE_FILE_EXTENSION

        effective_extension = str(file_ext or LIGHT_CURVE_FILE_EXTENSION).lstrip(".")
        input_paths: list[Path] = []
        for source_id, raw_path in zip(
            df_with_stats[source_id_col].astype(str),
            df_with_stats[path_col].astype(str),
        ):
            path = Path(raw_path).expanduser()
            input_paths.append(path if path.is_file() else path / f"{source_id}.{effective_extension}")
        checkpoint_fingerprint = build_stage_fingerprint(
            stage="tag_stats",
            stage_version=str(TAG_STATS_VERSION),
            candidate_ids=df_with_stats[id_col].astype(str).tolist(),
            input_paths=input_paths,
            settings={
                "compute_time": bool(compute_time),
                "compute_cameras": bool(compute_cameras),
                "compute_fields": bool(compute_fields),
                "file_extension": effective_extension,
                "min_good_points_per_camera": TAG_MIN_GOOD_POINTS_PER_CAMERA,
            },
            code_base=Path(__file__).resolve().parent.parent,
            code_paths=("stv/tag.py", "core/utils.py"),
            hash_input_contents=False,
        )
        checkpoint_artifacts_exist = (
            checkpoint_path.exists()
            or _stats_checkpoint_parts_dir(checkpoint_path).exists()
        )
        if checkpoint_artifacts_exist:
            try:
                _assert_reusable_stats_checkpoint_state(
                    read_stage_state(checkpoint_state_path),
                    fingerprint=checkpoint_fingerprint,
                    checkpoint_path=checkpoint_path,
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"Unsafe tag-stat checkpoint reuse: {exc}. "
                    "Remove the checkpoint and its .parts directory or start a new run."
                ) from exc
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

        checkpoint_is_current = (
            "tag_stats_status" in checkpoint_df.columns
            and "tag_stats_version" in checkpoint_df.columns
        )
        if checkpoint_is_current:
            status_ok = checkpoint_df["tag_stats_status"].astype("string").str.lower().eq("ok")
            version_ok = pd.to_numeric(checkpoint_df["tag_stats_version"], errors="coerce").eq(TAG_STATS_VERSION)
        else:
            status_ok = pd.Series(False, index=checkpoint_df.index)
            version_ok = pd.Series(False, index=checkpoint_df.index)

        if required_complete_cols and all(col in checkpoint_df.columns for col in required_complete_cols):
            complete_mask = checkpoint_df[required_complete_cols].notna().all(axis=1) & status_ok & version_ok
            checkpoint_complete = checkpoint_df.loc[complete_mask].drop_duplicates(subset=[id_col], keep="last")
        else:
            checkpoint_complete = checkpoint_df.loc[status_ok & version_ok].drop_duplicates(subset=[id_col], keep="last")

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
    pending = df_with_stats.loc[pending_mask]
    tasks = list(
        zip(
            pending.index.tolist(),
            pending[id_col].astype(str).tolist(),
            pending[source_id_col].astype(str).tolist(),
            pending[path_col].astype(str).tolist(),
        )
    )

    if not tasks:
        if show_tqdm:
            tqdm.write(f"[stats] All {len(df)} rows already computed from checkpoint")
        if checkpoint_state_path is not None and checkpoint_fingerprint is not None:
            write_stage_state(
                checkpoint_state_path,
                fingerprint=checkpoint_fingerprint,
                result=StageResult(
                    stage="tag_stats",
                    status="success",
                    expected=len(df_with_stats),
                    succeeded=len(df_with_stats),
                ),
                outputs=_stats_checkpoint_outputs(checkpoint_path),
            )
        return df_with_stats

    if checkpoint_state_path is not None and checkpoint_fingerprint is not None:
        write_stage_state(
            checkpoint_state_path,
            fingerprint=checkpoint_fingerprint,
            result=StageResult(
                stage="tag_stats",
                status="running",
                expected=len(df_with_stats),
                succeeded=len(already_computed),
            ),
            outputs=_stats_checkpoint_outputs(checkpoint_path),
        )

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
        chunk_tasks: list[tuple[int, str, str, str]],
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
        if checkpoint_state_path is not None and checkpoint_fingerprint is not None:
            processed = len(already_computed) + int(chunk_end)
            status_ok = (
                df_with_stats["tag_stats_status"]
                .astype("string")
                .str.lower()
                .eq("ok")
                .fillna(False)
            )
            n_succeeded = int(status_ok.sum())
            write_stage_state(
                checkpoint_state_path,
                fingerprint=checkpoint_fingerprint,
                result=StageResult(
                    stage="tag_stats",
                    status="running",
                    expected=len(df_with_stats),
                    succeeded=n_succeeded,
                    failed=max(0, processed - n_succeeded),
                ),
                outputs=_stats_checkpoint_outputs(checkpoint_path),
            )
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


    if checkpoint_state_path is not None and checkpoint_fingerprint is not None:
        status_ok = (
            df_with_stats["tag_stats_status"]
            .astype("string")
            .str.lower()
            .eq("ok")
            .fillna(False)
            .astype(bool)
        )
        n_succeeded = int(status_ok.sum())
        n_failed = int(len(df_with_stats) - n_succeeded)
        failed_ids = df_with_stats.loc[~status_ok, id_col].astype(str).head(100)
        failed_errors = df_with_stats.loc[~status_ok, "tag_stats_error"].astype(str).head(100)
        errors = tuple(
            f"{row_id}: {error}"
            for row_id, error in zip(failed_ids, failed_errors)
        )
        write_stage_state(
            checkpoint_state_path,
            fingerprint=checkpoint_fingerprint,
            result=StageResult(
                stage="tag_stats",
                status="success" if n_failed == 0 else "partial",
                expected=len(df_with_stats),
                succeeded=n_succeeded,
                failed=n_failed,
                errors=errors,
            ),
            outputs=_stats_checkpoint_outputs(checkpoint_path),
        )

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
        validate_candidate_ids(df, id_col)
        source_id_col = "asas_sn_id" if "asas_sn_id" in df.columns else id_col
        path_col = "path" if "path" in df.columns else None

        if path_col is None:
            raise ValueError("Need 'path' column to read dat2 files")

        df_with_stats = df.copy()
        df_with_stats["time_span_days"] = np.nan
        df_with_stats["points_per_day"] = np.nan

        pbar = tqdm(total=len(df), desc="filter_sparse_lightcurves (computing stats)", leave=False, disable=not show_tqdm)
        for idx, row in df.iterrows():
            asas_sn_id = str(row[id_col])
            lc_source_id = str(row[source_id_col])
            dir_path = str(row[path_col])

            stats = _compute_stats_for_row(
                asas_sn_id,
                dir_path,
                compute_time=True,
                compute_cameras=False,
                compute_fields=True,
                file_ext=file_ext,
                lc_source_id=lc_source_id,
            )
            for col, value in stats.items():
                if col == "asas_sn_id":
                    continue
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
    id_col: str | None = None,
    join_id_col: str | None = None,
) -> pd.DataFrame:
    """
    Attach VSX crossmatch info (vsx_sep_arcsec/vsx_class) to the dataframe.

    Uses the provided full crossmatch Parquet, or keeps existing VSX columns.
    """
    if "vsx_sep_arcsec" in df.columns and "vsx_class" in df.columns:
        return df
    if vsx_crossmatch_csv is None:
        raise ValueError("vsx_crossmatch_csv is required to attach VSX info.")

    vsx_crossmatch_csv = Path(vsx_crossmatch_csv)
    xmatch = normalize_vsx_match_columns(read_parquet_table(vsx_crossmatch_csv))

    missing_cols = [c for c in ("asas_sn_id", "vsx_sep_arcsec", "vsx_class") if c not in xmatch.columns]
    if missing_cols:
        raise ValueError(
            f"VSX crossmatch file {vsx_crossmatch_csv} is missing required columns {missing_cols}. "
            f"Found columns: {list(xmatch.columns)}"
        )

    vsx_cols = [c for c in ("vsx_sep_arcsec", "vsx_class", "vsx_period") if c in xmatch.columns]
    xmatch = select_best_vsx_matches(xmatch[["asas_sn_id", *vsx_cols]], id_column="asas_sn_id")
    id_col = get_id_col(df) if id_col is None else str(id_col)
    if join_id_col is None:
        join_id_col = "asas_sn_id" if "asas_sn_id" in df.columns else id_col
    if join_id_col not in df.columns:
        raise KeyError(f"Missing VSX join ID column: {join_id_col}")
    df = df.copy()
    original_ids = validate_candidate_ids(df, id_col)
    df[id_col] = original_ids
    original_len = len(df)
    df["__vsx_row_order"] = np.arange(original_len, dtype=np.int64)
    xmatch = xmatch.rename(columns={"asas_sn_id": "_vsx_join_id"})
    xmatch["_vsx_join_id"] = xmatch["_vsx_join_id"].astype("string").str.strip()
    df[join_id_col] = df[join_id_col].astype("string").str.strip()
    df = df.merge(
        xmatch,
        left_on=join_id_col,
        right_on="_vsx_join_id",
        how="left",
        suffixes=("", "_vsx"),
        validate="many_to_one",
        sort=False,
    )
    if len(df) != original_len:
        raise RuntimeError("VSX attachment changed the candidate row count")
    df = df.sort_values("__vsx_row_order", kind="stable").drop(
        columns=["__vsx_row_order", "_vsx_join_id"], errors="ignore"
    ).reset_index(drop=True)
    if df[id_col].astype("string").tolist() != original_ids.tolist():
        raise RuntimeError("VSX attachment changed canonical candidate identity")
    for col in vsx_cols:
        xcol = f"{col}_vsx"
        if xcol not in df.columns:
            continue
        if col == "vsx_class":
            base = df[col] if col in df.columns else pd.Series(pd.NA, index=df.index)
            missing = base.isna() | base.astype(str).str.strip().str.lower().isin({"", "nan", "none", "<na>"})
            if col not in df.columns:
                df[col] = pd.NA
            df.loc[missing, col] = df.loc[missing, xcol]
        else:
            base = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
            fill = pd.to_numeric(df[xcol], errors="coerce")
            df[col] = base.combine_first(fill)
        df = df.drop(columns=[xcol])
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
        validate_candidate_ids(df, id_col)
        source_id_col = "asas_sn_id" if "asas_sn_id" in df.columns else id_col
        path_col = "path" if "path" in df.columns else None

        if path_col is None:
            raise ValueError("Need 'path' column to read dat2 files")

        df_with_cameras = df.copy()
        df_with_cameras["n_cameras"] = 0

        pbar = tqdm(total=len(df), desc="filter_multi_camera (counting cameras)", leave=False, disable=not show_tqdm)
        for idx, row in df.iterrows():
            asas_sn_id = str(row[id_col])
            lc_source_id = str(row[source_id_col])
            dir_path = str(row[path_col])

            stats = _compute_stats_for_row(
                asas_sn_id,
                dir_path,
                compute_time=False,
                compute_cameras=True,
                compute_fields=True,
                file_ext=file_ext,
                lc_source_id=lc_source_id,
            )
            for col, value in stats.items():
                if col == "asas_sn_id":
                    continue
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
    medians = _read_raw2_camera_medians(raw2_path)
    if not medians:
        return pd.DataFrame(columns=RAW2_COLUMNS)

    records = [
        {
            "camera": camera,
            "median": median,
            "sig1_low": np.nan,
            "sig1_high": np.nan,
            "pct90_low": np.nan,
            "pct90_high": np.nan,
        }
        for camera, median in medians
    ]
    return pd.DataFrame.from_records(records, columns=RAW2_COLUMNS)


def _read_raw2_camera_medians(raw2_path: Path) -> list[tuple[int, float]]:
    """Read only camera ids and medians from a small whitespace-delimited .raw2 file."""
    if not raw2_path.exists():
        return []

    medians: list[tuple[int, float]] = []
    try:
        with raw2_path.open("r", encoding="ascii", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 2:
                    continue
                try:
                    camera = int(float(parts[0]))
                    median = float(parts[1])
                except ValueError:
                    continue
                if np.isfinite(median):
                    medians.append((camera, median))
    except Exception:
        return []

    return medians


def _process_camera_median_row(
    asas_sn_id: str,
    path_str: str,
    mag_bin: str,
    mag_tolerance: float,
    mag_range_cache: dict[str, tuple[float, float] | None] | None = None,
) -> tuple[str, str]:
    """
    Helper for parallel camera median filtering.
    """
    try:
        path = Path(path_str)
        if path.is_dir():
            return asas_sn_id, ""

        if mag_range_cache is not None and mag_bin in mag_range_cache:
            mag_range = mag_range_cache[mag_bin]
        else:
            mag_range = _parse_mag_bin_range(mag_bin)
            if mag_range_cache is not None:
                mag_range_cache[mag_bin] = mag_range
        if mag_range is None:
            return asas_sn_id, ""

        raw2_path = path.with_suffix(".raw2")
        medians = _read_raw2_camera_medians(raw2_path)
        if not medians:
            return asas_sn_id, ""

        mag_min = mag_range[0] - mag_tolerance
        mag_max = mag_range[1] + mag_tolerance

        suspect_cameras = [
            camera
            for camera, median in medians
            if median < mag_min or median > mag_max
        ]

        if suspect_cameras:
            return asas_sn_id, ",".join(map(str, suspect_cameras))
        
        return asas_sn_id, ""
    except Exception:
        return asas_sn_id, ""


def _process_camera_median_batch(
    batch: list[tuple[str, str, str]],
    mag_tolerance: float,
) -> list[tuple[str, str]]:
    """Process a batch of camera median tasks in one worker call."""
    mag_range_cache: dict[str, tuple[float, float] | None] = {}
    return [
        _process_camera_median_row(asas_sn_id, path_str, mag_bin, mag_tolerance, mag_range_cache)
        for asas_sn_id, path_str, mag_bin in batch
    ]


def _camera_median_checkpoint_parts_dir(checkpoint_path: Path) -> Path:
    """Directory for incremental camera median checkpoint shards."""
    return checkpoint_path.with_name(f"{checkpoint_path.name}.parts")


def _read_camera_median_checkpoint_frame(path: Path, id_col: str) -> pd.DataFrame | None:
    columns = [id_col, RAW_MEDIAN_SUSPECT_COL]
    try:
        frame = pd.read_parquet(path, columns=columns)
    except Exception:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            raise

    if id_col not in frame.columns or RAW_MEDIAN_SUSPECT_COL not in frame.columns:
        return None

    frame = frame.loc[:, columns].copy()
    frame[id_col] = frame[id_col].astype(str)
    frame[RAW_MEDIAN_SUSPECT_COL] = frame[RAW_MEDIAN_SUSPECT_COL].fillna("").astype(str)
    return frame


def _load_camera_median_checkpoint(
    checkpoint_path: Path,
    id_col: str,
    *,
    show_tqdm: bool = False,
) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    source_messages: list[str] = []

    if checkpoint_path.exists():
        try:
            frame = _read_camera_median_checkpoint_frame(checkpoint_path, id_col)
            if frame is not None:
                frames.append(frame)
                source_messages.append(str(checkpoint_path))
            elif show_tqdm:
                tqdm.write("[filter_camera_medians] Ignoring incompatible checkpoint without raw suspect cameras")
        except Exception as e:
            if show_tqdm:
                tqdm.write(f"[filter_camera_medians] Warning: Could not load checkpoint {checkpoint_path}: {e}")

    parts_dir = _camera_median_checkpoint_parts_dir(checkpoint_path)
    if parts_dir.exists():
        part_paths = sorted(parts_dir.glob("part-*.parquet"))
        part_frames: list[pd.DataFrame] = []
        for part_path in part_paths:
            try:
                frame = _read_camera_median_checkpoint_frame(part_path, id_col)
                if frame is not None:
                    part_frames.append(frame)
            except Exception as e:
                if show_tqdm:
                    tqdm.write(f"[filter_camera_medians] Warning: Could not load checkpoint part {part_path}: {e}")
        if part_frames:
            frames.extend(part_frames)
            source_messages.append(f"{len(part_frames)} part file(s) in {parts_dir}")

    if not frames:
        return None

    checkpoint_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    checkpoint_df = checkpoint_df.drop_duplicates(subset=[id_col], keep="last")
    if show_tqdm:
        tqdm.write(
            f"[filter_camera_medians] Loaded checkpoint with {len(checkpoint_df)} rows from "
            f"{', '.join(source_messages)}"
        )
    return checkpoint_df


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
        checkpoint_df = _load_camera_median_checkpoint(
            checkpoint_path,
            id_col,
            show_tqdm=show_tqdm,
        )
        if checkpoint_df is not None:
            already_processed = set(checkpoint_df[id_col])

    # Prepare pending rows without materializing one Python task per full input row.
    df_out[id_col] = df_out[id_col].astype(str)
    pending_mask = (
        ~df_out[id_col].isin(already_processed)
        if already_processed
        else pd.Series(True, index=df_out.index)
    )
    pending = df_out.loc[pending_mask, [id_col, "path", "mag_bin"]]
    n_tasks = len(pending)

    # If everything is checkpointed, just merge and return
    if n_tasks == 0 and checkpoint_df is not None:
        if show_tqdm:
            tqdm.write("[filter_camera_medians] All rows found in checkpoint.")
        
        # Merge checkpoint results
        checkpoint_subset = checkpoint_df[[id_col, RAW_MEDIAN_SUSPECT_COL]]
        df_out = df_out.drop(columns=[RAW_MEDIAN_SUSPECT_COL], errors="ignore")
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
        tqdm.write(f"[filter_camera_medians] Processing {n_tasks} rows with {n_workers} workers")

    # Function to save checkpoint
    def save_checkpoint_part(result_rows: list[tuple[str, str]]) -> None:
        if checkpoint_path is None or not result_rows:
            return

        df_checkpoint = pd.DataFrame.from_records(
            result_rows,
            columns=[id_col, RAW_MEDIAN_SUSPECT_COL],
        )
        df_checkpoint = df_checkpoint.drop_duplicates(subset=[id_col], keep="last")

        parts_dir = _camera_median_checkpoint_parts_dir(checkpoint_path)
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

    chunk_size = max(1, int(chunk_size))
    task_batch_size = max(1, min(chunk_size, max(32, min(1000, chunk_size // max(1, n_workers)))))

    def make_tasks(chunk: pd.DataFrame) -> list[tuple[str, str, str]]:
        return list(
            zip(
                chunk[id_col].astype(str).tolist(),
                chunk["path"].astype(str).tolist(),
                chunk["mag_bin"].astype(str).tolist(),
            )
        )

    def process_chunk(
        chunk_tasks: list[tuple[str, str, str]],
        executor: ProcessPoolExecutor | None = None,
    ) -> list[tuple[str, str]]:
        chunk_results: list[tuple[str, str]] = []
        task_batches = _iter_batches(chunk_tasks, task_batch_size)

        if executor is None:
            for batch in task_batches:
                batch_results = _process_camera_median_batch(batch, mag_tolerance)
                chunk_results.extend(batch_results)
                pbar.update(len(batch_results))
            return chunk_results

        futures = [
            executor.submit(_process_camera_median_batch, batch, mag_tolerance)
            for batch in task_batches
        ]
        for future in as_completed(futures):
            batch_results = future.result()
            chunk_results.extend(batch_results)
            pbar.update(len(batch_results))
        return chunk_results

    # Run in parallel
    pbar = tqdm(total=n_tasks, desc="filter_camera_medians", leave=False, disable=not show_tqdm)
    
    try:
        if n_workers <= 1:
            for chunk_start in range(0, n_tasks, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_tasks)
                chunk_results = process_chunk(make_tasks(pending.iloc[chunk_start:chunk_end]))
                if checkpoint_path is None:
                    new_results.extend(chunk_results)
                else:
                    save_checkpoint_part(chunk_results)
        else:
            try:
                executor_cm = ProcessPoolExecutor(max_workers=n_workers)
            except (OSError, PermissionError) as exc:
                if show_tqdm:
                    tqdm.write(f"[filter_camera_medians] Falling back to sequential execution: {exc}")
                executor_cm = None

            if executor_cm is None:
                for chunk_start in range(0, n_tasks, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n_tasks)
                    chunk_results = process_chunk(make_tasks(pending.iloc[chunk_start:chunk_end]))
                    if checkpoint_path is None:
                        new_results.extend(chunk_results)
                    else:
                        save_checkpoint_part(chunk_results)
            else:
                with executor_cm as executor:
                    for chunk_start in range(0, n_tasks, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, n_tasks)
                        chunk_results = process_chunk(make_tasks(pending.iloc[chunk_start:chunk_end]), executor)
                        if checkpoint_path is None:
                            new_results.extend(chunk_results)
                        else:
                            save_checkpoint_part(chunk_results)
    finally:
        pbar.close()

    # Final merge
    if checkpoint_path is not None:
        final_results_df = _load_camera_median_checkpoint(
            checkpoint_path,
            id_col,
            show_tqdm=False,
        )
        if final_results_df is None:
            final_results_df = pd.DataFrame(columns=[id_col, RAW_MEDIAN_SUSPECT_COL])
    else:
        final_results_df = pd.DataFrame(new_results, columns=[id_col, RAW_MEDIAN_SUSPECT_COL])
        
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

    id_col = get_id_col(df_filtered)
    df_filtered[id_col] = validate_candidate_ids(df_filtered, id_col)
    source_id_col = "asas_sn_id" if "asas_sn_id" in df_filtered.columns else id_col
    if source_id_col != id_col:
        source_ids = df_filtered[source_id_col].astype("string").str.strip()
        invalid_source = source_ids.isna() | source_ids.str.lower().isin(
            {"", "nan", "none", "null", "<na>"}
        )
        if bool(invalid_source.any()):
            raise ValueError(
                f"Light-curve source ID column '{source_id_col}' contains "
                f"{int(invalid_source.sum())} invalid value(s)"
            )
        df_filtered[source_id_col] = source_ids

    row_key = "__tag_row_id"
    if row_key in df_filtered.columns:
        raise ValueError(f"Reserved internal tag column already exists: {row_key}")
    df_filtered[row_key] = np.arange(n_start, dtype=np.int64)

    # These columns are wholly owned by this invocation.  Rerunning tags with a
    # filter disabled must not preserve a stale failure from an earlier run.
    owned_failure_columns = {
        "failed_sparse",
        "failed_multi_camera",
        "failed_mag_range",
        "failed_tag_stats",
        "failed_any",
    }
    df_filtered = df_filtered.drop(
        columns=[column for column in owned_failure_columns if column in df_filtered.columns],
        errors="ignore",
    )

    precomputed_time = False
    precomputed_cameras = False

    # Always compute the light-curve accounting ledger when paths are present.
    # Even when individual rejection rules are disabled, downstream event
    # products must say how many raw/clean rows and cameras were actually seen;
    # a disabled cut is not permission to lose that provenance.
    if "path" in df_filtered.columns:
        compute_time = apply_sparse
        compute_cameras = apply_multi_camera
        compute_fields = compute_time or compute_cameras

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
            source_id_col=source_id_col,
        )
        precomputed_time = compute_time
        precomputed_cameras = compute_cameras

    if apply_vsx:
        df_filtered = attach_vsx_info(
            df_filtered,
            vsx_crossmatch_csv=vsx_crossmatch_csv,
            id_col=id_col,
            join_id_col=source_id_col,
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
    total_steps = len(filters)
    active_failed_cols: list[str] = []
    if total_steps > 0:
        with tqdm(total=total_steps, desc="apply_tags", leave=True, disable=not show_tqdm) as pbar:
            for label, func, kwargs in filters:
                start = perf_counter()
                # Run filter to identify which rows pass
                kwargs_clean = {k: v for k, v in kwargs.items() if k != "rejected_log_csv"}
                df_passed = func(df_filtered, **kwargs_clean)
                elapsed = perf_counter() - start

                # Compare immutable row keys, not a catalogue ID inferred after
                # joins.  This also remains correct when paths are shared.
                passed_rows = set(pd.to_numeric(df_passed[row_key], errors="raise").astype(int))
                failed_mask = ~df_filtered[row_key].isin(passed_rows)
                failed_col = f"failed_{label}"
                df_filtered[failed_col] = failed_mask
                active_failed_cols.append(failed_col)

                n_failed = int(failed_mask.sum())
                pbar.set_postfix_str(f"{label}: {n_failed}/{n_start} failed ({elapsed:.2f}s)")
                pbar.update(1)

    # Add summary column
    if "tag_stats_status" in df_filtered.columns:
        df_filtered["failed_tag_stats"] = ~df_filtered["tag_stats_status"].astype("string").str.lower().eq("ok")
        active_failed_cols.append("failed_tag_stats")

    if active_failed_cols:
        df_filtered["failed_any"] = df_filtered[active_failed_cols].fillna(True).any(axis=1)
    else:
        df_filtered["failed_any"] = False

    if show_tqdm:
        n_failed_any = int(df_filtered["failed_any"].sum()) if "failed_any" in df_filtered.columns else 0
        tqdm.write(f"\n[apply_tags] {n_failed_any}/{n_start} failed at least one filter")

    if df_filtered[row_key].nunique(dropna=False) != n_start:
        raise RuntimeError("Tagging changed or duplicated internal candidate row identity")
    df_filtered = df_filtered.sort_values(row_key, kind="stable").drop(columns=[row_key])
    if df_filtered[id_col].astype("string").tolist() != df[id_col].astype("string").str.strip().tolist():
        raise RuntimeError("Tagging changed canonical candidate identity")
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
    df = read_feature_table(input_path)

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

    write_feature_table(out, output_path)

    print(f"Saved tag output: {output_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
