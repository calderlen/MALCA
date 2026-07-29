from __future__ import annotations

import json
import hashlib
import logging
import multiprocessing as mp
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import argparse
import astropy.units as u
from astropy.table import Table
from astroquery.xmatch import XMatch
from tqdm.auto import tqdm

from malca.products.candidates import select_passing_candidates_if_present
from malca.config import NEIGHBOR_RADIUS_ARCSEC, NEIGHBOR_CHUNK_SIZE
from malca.io.table_io import read_feature_table


DEFAULT_NEIGHBOR_CATALOGS: dict[str, str] = {
    "gaia_dr3": "I/355/gaiadr3",
    "2mass": "II/246/out",
    "allwise": "II/328/allwise",
    "vsx": "B/vsx/vsx",
}

_CATALOG_OBJECT_ID_COLUMNS = (
    "Source", "source_id", "SOURCE_ID", "AllWISE", "_2MASS", "2MASS",
    "Name", "OID", "VarID", "objID", "ObjectId",
)


def _first_nonempty_text(row: pd.Series, columns: tuple[str, ...]) -> str:
    for column in columns:
        if column not in row.index:
            continue
        value = row.get(column)
        try:
            if value is None or pd.isna(value):
                continue
        except Exception:
            if value is None:
                continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "<na>"}:
            return text
    return ""


def _normalize_neighbor_records(
    neighbors: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Add stable row identity, remove repeated cache/query rows, and mark self matches."""
    if neighbors.empty:
        return neighbors
    out = neighbors.copy()
    out["candidate_id"] = out["candidate_id"].astype(str)
    if "catalog" not in out.columns:
        out["catalog"] = ""
    else:
        out["catalog"] = out["catalog"].fillna("").astype(str)
    if "sep_arcsec" not in out.columns:
        out["sep_arcsec"] = np.nan
    else:
        out["sep_arcsec"] = pd.to_numeric(out["sep_arcsec"], errors="coerce")
    out["neighbor_catalog_object_id"] = out.apply(
        lambda row: _first_nonempty_text(row, _CATALOG_OBJECT_ID_COLUMNS), axis=1
    )

    def stable_key(row: pd.Series) -> str:
        object_id = str(row.get("neighbor_catalog_object_id") or "")
        if object_id:
            token = f"{row['catalog']}|id:{object_id}"
        else:
            # The fallback is intentionally based only on stable astronomical
            # values, never on dataframe row position or query ordering.
            coordinate_tokens = []
            for axis, names in (
                ("ra", ("RA_ICRS", "RAJ2000", "RAdeg", "ra")),
                ("dec", ("DE_ICRS", "DEJ2000", "DEdeg", "dec")),
            ):
                for name in names:
                    value = pd.to_numeric(row.get(name), errors="coerce")
                    if np.isfinite(value):
                        coordinate_tokens.append(f"{axis}:{float(value):.8f}")
                        break
            sep = pd.to_numeric(row.get("sep_arcsec"), errors="coerce")
            token = "|".join(
                [str(row.get("catalog") or ""), *coordinate_tokens, f"sep:{float(sep):.5f}" if np.isfinite(sep) else "sep:nan"]
            )
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:20]
        return f"{row['catalog']}:{digest}"

    out["neighbor_record_key"] = out.apply(stable_key, axis=1)
    out = out.sort_values(
        ["candidate_id", "catalog", "neighbor_record_key", "sep_arcsec"],
        na_position="last",
        kind="mergesort",
    ).drop_duplicates(
        subset=["candidate_id", "catalog", "neighbor_record_key"], keep="first"
    )

    target_ids: dict[str, str] = {}
    for source_column in ("source_id", "gaia_id", "gaia_source_id"):
        if source_column not in candidates.columns:
            continue
        for _, row in candidates[["candidate_id", source_column]].dropna().iterrows():
            value = str(row[source_column]).strip()
            if value and value.lower() not in {"nan", "none", "<na>"}:
                target_ids.setdefault(str(row["candidate_id"]), value)
    out["is_target_match"] = False
    gaia = out["catalog"].str.contains("I/355|gaia", case=False, na=False, regex=True)
    if target_ids:
        uploaded_source = out["candidate_id"].map(target_ids).fillna("")
        catalog_source = out["neighbor_catalog_object_id"].fillna("").astype(str)
        out.loc[gaia & uploaded_source.ne("") & catalog_source.eq(uploaded_source), "is_target_match"] = True

    # For catalogs without a shared identifier, only a sub-arcsecond nearest
    # match is treated as the target.  Wider matches remain real neighbors.
    nearest_index = (
        out.loc[~out["is_target_match"] & out["sep_arcsec"].notna()]
        .sort_values("sep_arcsec", kind="mergesort")
        .groupby(["candidate_id", "catalog"], sort=False)
        .head(1)
        .index
    )
    close_nearest = nearest_index[out.loc[nearest_index, "sep_arcsec"].to_numpy(dtype=float) <= 1.0]
    out.loc[close_nearest, "is_target_match"] = True
    return out.reset_index(drop=True)


class XMatchChunkTimeoutError(TimeoutError):
    """Raised when a CDS XMatch chunk exceeds the parent-side timeout."""


def _xmatch_process_context():
    try:
        return mp.get_context("fork")
    except ValueError:  # pragma: no cover - Windows/non-POSIX fallback
        return mp.get_context("spawn")


def _xmatch_upload_table(source_frame: pd.DataFrame) -> Table:
    upload = source_frame[["candidate_id", "ra_deg", "dec_deg"]].copy()
    upload["candidate_id"] = upload["candidate_id"].astype(str)
    upload = upload.rename(columns={"ra_deg": "ra", "dec_deg": "dec"})
    return Table.from_pandas(upload)


def _query_xmatch_frame_direct(
    source_frame: pd.DataFrame,
    *,
    cat2: str,
    radius_arcsec: float,
    timeout_sec: float | None,
) -> pd.DataFrame:
    table = _xmatch_upload_table(source_frame)
    previous_timeout = getattr(XMatch, "TIMEOUT", None)
    try:
        if timeout_sec is not None:
            XMatch.TIMEOUT = float(timeout_sec)
        result = XMatch.query(
            cat1=table,
            cat2=cat2,
            max_distance=float(radius_arcsec) * u.arcsec,
            colRA1="ra",
            colDec1="dec",
        )
    finally:
        if timeout_sec is not None and previous_timeout is not None:
            XMatch.TIMEOUT = previous_timeout
    if result is None or len(result) == 0:
        return pd.DataFrame()
    return result.to_pandas()


def _xmatch_subprocess_worker(
    source_frame: pd.DataFrame,
    cat2: str,
    radius_arcsec: float,
    timeout_sec: float | None,
    result_path: str,
    status_path: str,
) -> None:
    try:
        frame = _query_xmatch_frame_direct(
            source_frame,
            cat2=cat2,
            radius_arcsec=radius_arcsec,
            timeout_sec=timeout_sec,
        )
        frame.to_pickle(result_path)
        status = {"status": "ok", "rows": int(len(frame))}
    except Exception as exc:  # pragma: no cover - exercised through parent path
        status = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    Path(status_path).write_text(json.dumps(status), encoding="utf-8")


def query_xmatch_chunk(
    source_frame: pd.DataFrame,
    *,
    cat2: str,
    radius_arcsec: float,
    timeout_sec: float | None,
) -> pd.DataFrame:
    """
    Run one CDS XMatch upload, optionally in a subprocess with a hard timeout.

    Astroquery's ``XMatch.TIMEOUT`` is still passed through to the child, but the
    parent process enforces the real wall-clock boundary and kills a stuck chunk.
    """
    if timeout_sec is None:
        return _query_xmatch_frame_direct(
            source_frame,
            cat2=cat2,
            radius_arcsec=radius_arcsec,
            timeout_sec=None,
        )

    timeout = max(0.001, float(timeout_sec))
    context = _xmatch_process_context()
    with tempfile.TemporaryDirectory(prefix="malca_xmatch_") as tmpdir:
        result_path = str(Path(tmpdir) / "result.pkl")
        status_path = str(Path(tmpdir) / "status.json")
        process = context.Process(
            target=_xmatch_subprocess_worker,
            args=(source_frame, cat2, float(radius_arcsec), timeout, result_path, status_path),
        )
        process.start()
        try:
            process.join(timeout)
            if process.is_alive():
                process.terminate()
                process.join(5)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(5)
                raise XMatchChunkTimeoutError(
                    f"XMatch chunk timed out after {timeout:g}s for {cat2}"
                )

            status_file = Path(status_path)
            result_file = Path(result_path)
            if status_file.exists():
                status = json.loads(status_file.read_text(encoding="utf-8"))
                if status.get("status") == "ok":
                    if result_file.exists():
                        return pd.read_pickle(result_file)
                    return pd.DataFrame()
                error_type = status.get("error_type") or "RuntimeError"
                error_message = status.get("error_message") or "unknown error"
                raise RuntimeError(f"{error_type}: {error_message}")

            if process.exitcode not in (0, None):
                raise RuntimeError(
                    f"XMatch subprocess exited with code {process.exitcode} for {cat2}"
                )
            raise RuntimeError(f"XMatch subprocess produced no status for {cat2}")
        finally:
            if not process.is_alive():
                try:
                    process.close()
                except ValueError:
                    pass


def _coord_from_layers(df: pd.DataFrame, axis: str) -> pd.Series:
    """Pull RA or Dec from layer-first columns when not present at top level."""
    from malca.products.feature_layers import feature_value_series

    paths = (f"external_stats.{axis}", f"derived_stats.{axis}", f"lc_stats.{axis}")
    for path in paths:
        layer = path.split(".", 1)[0]
        if layer not in df.columns:
            continue
        values = pd.to_numeric(feature_value_series(df, path), errors="coerce")
        if values.notna().any():
            return values
    return pd.Series(pd.NA, index=df.index, dtype="Float64")


def _ensure_candidate_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "candidate_id" not in out.columns:
        if "asas_sn_id" in out.columns:
            out["candidate_id"] = out["asas_sn_id"].astype(str)
        elif "path" in out.columns:
            out["candidate_id"] = out["path"].astype(str).map(lambda p: Path(p).stem.split("-")[0])
        else:
            out["candidate_id"] = np.arange(len(out)).astype(str)
    if "ra_deg" not in out.columns and "ra" in out.columns:
        out["ra_deg"] = pd.to_numeric(out["ra"], errors="coerce")
    if "dec_deg" not in out.columns and "dec" in out.columns:
        out["dec_deg"] = pd.to_numeric(out["dec"], errors="coerce")
    if "ra_deg" not in out.columns or out["ra_deg"].isna().all():
        out["ra_deg"] = _coord_from_layers(out, "ra")
    if "dec_deg" not in out.columns or out["dec_deg"].isna().all():
        out["dec_deg"] = _coord_from_layers(out, "dec")
    return out


def _query_catalog_bulk(
    coords_df: pd.DataFrame,
    *,
    catalog: str,
    radius_arcsec: float,
    chunk_size: int,
    xmatch_timeout_sec: float | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
    status_rows: list[dict] | None = None,
    status_context: dict[str, object] | None = None,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    n = len(coords_df)
    step = max(1, int(chunk_size))
    starts = range(0, n, step)
    total_chunks = (n + step - 1) // step
    iterator = tqdm(
        starts,
        total=total_chunks,
        desc=progress_desc or f"xmatch:{catalog}",
        disable=not show_progress,
    )
    for start in iterator:
        chunk = coords_df.iloc[start : start + int(chunk_size)].copy()
        if chunk.empty:
            continue
        chunk_index = int(start // step) + 1
        if show_progress:
            print(
                f"  {progress_desc or f'xmatch:{catalog}'}: "
                f"chunk {chunk_index}/{total_chunks} ({len(chunk)} rows)",
                flush=True,
            )
        attempted = int(len(chunk))
        status_base = {
            "catalog": catalog,
            "mode": "xmatch",
            "chunk_start": int(start),
            "chunk_stop": int(start + attempted),
            "attempted": attempted,
            "matched": 0,
            "error_message": "",
        }
        if status_context:
            status_base.update(status_context)
        chunk_upload = chunk[["candidate_id", "ra_deg", "dec_deg"]]
        try:
            out = query_xmatch_chunk(
                chunk_upload,
                cat2=f"vizier:{catalog}",
                radius_arcsec=radius_arcsec,
                timeout_sec=xmatch_timeout_sec,
            )
        except XMatchChunkTimeoutError as e:
            logging.warning(f"XMatch query timed out for {catalog}: {e}")
            if show_progress:
                print(
                    f"Warning: {progress_desc or f'xmatch:{catalog}'} "
                    f"chunk {chunk_index}/{total_chunks} timed out: {e}",
                    flush=True,
                )
            if status_rows is not None:
                status_rows.append({
                    **status_base,
                    "status": "timeout",
                    "error_message": str(e),
                })
            continue
        except Exception as e:
            logging.warning(f"XMatch query failed for {catalog}: {e}")
            if show_progress:
                print(
                    f"Warning: {progress_desc or f'xmatch:{catalog}'} "
                    f"chunk {chunk_index}/{total_chunks} failed: {e}",
                    flush=True,
                )
            if status_rows is not None:
                status_rows.append({
                    **status_base,
                    "status": "error",
                    "error_message": str(e),
                })
            continue
        if out.empty:
            if status_rows is not None:
                status_rows.append({**status_base, "status": "no_data"})
            if show_progress:
                print(
                    f"  {progress_desc or f'xmatch:{catalog}'}: "
                    f"chunk {chunk_index}/{total_chunks} matched 0 row(s)",
                    flush=True,
                )
            continue
        sep_col = None
        for candidate in ["angDist", "_r", "separation", "Sep"]:
            if candidate in out.columns:
                sep_col = candidate
                break
        if sep_col is None:
            if status_rows is not None:
                status_rows.append({
                    **status_base,
                    "status": "error",
                    "matched": int(len(out)),
                    "error_message": "missing separation column in XMatch result",
                })
            continue
        out = out.rename(columns={sep_col: "sep_arcsec"})
        out["catalog"] = catalog
        if status_rows is not None:
            status_rows.append({
                **status_base,
                "status": "ok",
                "matched": int(len(out)),
            })
        if show_progress:
            print(
                f"  {progress_desc or f'xmatch:{catalog}'}: "
                f"chunk {chunk_index}/{total_chunks} matched {len(out)} row(s)",
                flush=True,
            )
        chunks.append(out)

    if not chunks:
        return pd.DataFrame()
    merged = pd.concat(chunks, ignore_index=True)
    if "candidate_id" in merged.columns:
        merged["candidate_id"] = merged["candidate_id"].astype(str)
    return merged


def run_neighbor_enrichment(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    radius_arcsec: float = NEIGHBOR_RADIUS_ARCSEC,
    chunk_size: int = NEIGHBOR_CHUNK_SIZE,
    cache_file: Path | None = None,
    catalogs: dict[str, str] | None = None,
    checkpoint_path: Path | None = None,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bulk nearest-neighbor enrichment with optional cache."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalogs = catalogs or DEFAULT_NEIGHBOR_CATALOGS

    df_use = _ensure_candidate_id(df)
    coords_cols = ["candidate_id", "ra_deg", "dec_deg"]
    if not all(c in df_use.columns for c in coords_cols):
        empty = pd.DataFrame()
        empty.to_parquet(out_dir / "neighbors_long.parquet", index=False, compression="zstd")
        empty.to_parquet(out_dir / "neighbors_summary.parquet", index=False, compression="zstd")
        return empty, empty

    coords = df_use[coords_cols].dropna(subset=["ra_deg", "dec_deg"]).copy()
    coords["candidate_id"] = coords["candidate_id"].astype(str)
    coords = coords.drop_duplicates(subset=["candidate_id"])

    # Load checkpoint to skip already-processed candidates
    ckpt_df = pd.DataFrame()
    cached_ids: set[str] = set()
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            ckpt_df = pd.read_parquet(checkpoint_path)
            if "candidate_id" in ckpt_df.columns:
                cached_ids = set(ckpt_df["candidate_id"].astype(str))
                print(f"[neighbor] Loaded checkpoint: {len(cached_ids)} candidates already processed")
        except Exception:
            ckpt_df = pd.DataFrame()

    coords_todo = coords[~coords["candidate_id"].isin(cached_ids)] if cached_ids else coords

    # Load cache only if no checkpoint (checkpoint is a superset of cache)
    cache_df = pd.DataFrame()
    if not cached_ids and cache_file and Path(cache_file).exists():
        try:
            cache_df = pd.read_parquet(cache_file)
        except Exception:
            cache_df = pd.DataFrame()

    fresh_frames: list[pd.DataFrame] = []
    if not coords_todo.empty:
        catalog_items = list(catalogs.items())
        catalog_iter = tqdm(catalog_items, desc="Neighbor catalogs", disable=not show_progress)
        for catalog_name, catalog_id in catalog_iter:
            fresh = _query_catalog_bulk(
                coords_todo,
                catalog=catalog_id,
                radius_arcsec=radius_arcsec,
                chunk_size=chunk_size,
                show_progress=show_progress,
                progress_desc=f"neighbor:{catalog_name}",
            )
            if not fresh.empty:
                fresh_frames.append(fresh)
    elif cached_ids:
        print(f"[neighbor] All {len(coords)} candidates already in checkpoint, skipping queries")

    if fresh_frames:
        fresh_df = pd.concat(fresh_frames, ignore_index=True)
    else:
        fresh_df = pd.DataFrame()

    # Combine: checkpoint OR cache, plus fresh results
    parts = [p for p in [ckpt_df, cache_df, fresh_df] if not p.empty]
    neighbors_long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not neighbors_long.empty:
        neighbors_long = _normalize_neighbor_records(neighbors_long, df_use)
        keep_cols = [c for c in ["candidate_id", "catalog", "sep_arcsec", "phot_g_mean_mag", "VarType", "Type"] if c in neighbors_long.columns]
        keep_cols += [c for c in neighbors_long.columns if c not in keep_cols]
        neighbors_long = neighbors_long[keep_cols]
        if "candidate_id" in neighbors_long.columns:
            neighbors_long["candidate_id"] = neighbors_long["candidate_id"].astype(str)

    if cache_file and not neighbors_long.empty:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        neighbors_long.to_parquet(cache_file, index=False, compression="snappy")

    summary = coords[["candidate_id"]].copy()
    summary["candidate_id"] = summary["candidate_id"].astype(str)
    if neighbors_long.empty:
        summary["neighbor_count"] = 0
        summary["neighbor_unique_count"] = 0
        summary["neighbor_catalog_match_count"] = 0
        summary["neighbor_target_match_count"] = 0
        summary["neighbor_catalog_count"] = 0
        summary["neighbor_count_by_catalog_json"] = "{}"
        summary["nearest_sep_arcsec"] = np.nan
        summary["nearby_known_variable"] = False
        summary["bright_close_neighbor"] = False
        summary["local_density_n_15as"] = 0
    else:
        target_match_count = (
            neighbors_long.groupby("candidate_id")["is_target_match"].sum().astype(int)
            .rename("neighbor_target_match_count")
        )
        actual_neighbors = neighbors_long.loc[~neighbors_long["is_target_match"].fillna(False)].copy()
        grp = actual_neighbors.groupby("candidate_id")
        raw_grp = neighbors_long.groupby("candidate_id")
        summary = summary.merge(grp.size().rename("neighbor_unique_count"), on="candidate_id", how="left")
        summary["neighbor_count"] = summary["neighbor_unique_count"]
        summary = summary.merge(raw_grp.size().rename("neighbor_catalog_match_count"), on="candidate_id", how="left")
        summary = summary.merge(target_match_count, on="candidate_id", how="left")
        summary = summary.merge(grp["catalog"].nunique().rename("neighbor_catalog_count"), on="candidate_id", how="left")
        summary = summary.merge(grp["sep_arcsec"].min().rename("nearest_sep_arcsec"), on="candidate_id", how="left")
        by_catalog = (
            actual_neighbors.groupby(["candidate_id", "catalog"]).size().unstack(fill_value=0)
            if not actual_neighbors.empty else pd.DataFrame()
        )
        count_json = {
            str(candidate_id): json.dumps(
                {str(catalog): int(count) for catalog, count in row.items() if int(count) > 0},
                sort_keys=True,
                separators=(",", ":"),
            )
            for candidate_id, row in by_catalog.iterrows()
        }
        summary["neighbor_count_by_catalog_json"] = summary["candidate_id"].map(count_json).fillna("{}")
        summary["neighbor_count"] = summary["neighbor_count"].fillna(0).astype(int)
        for count_col in (
            "neighbor_unique_count", "neighbor_catalog_match_count",
            "neighbor_target_match_count", "neighbor_catalog_count",
        ):
            summary[count_col] = summary[count_col].fillna(0).astype(int)
        density = (
            actual_neighbors.loc[pd.to_numeric(actual_neighbors["sep_arcsec"], errors="coerce") <= 15.0]
            .groupby("candidate_id").size().rename("local_density_n_15as")
        )
        summary = summary.merge(density, on="candidate_id", how="left")
        summary["local_density_n_15as"] = summary["local_density_n_15as"].fillna(0).astype(int)

        known_var_mask = actual_neighbors["catalog"].astype(str).str.contains("vsx", case=False, na=False)
        known_var = actual_neighbors.loc[known_var_mask, ["candidate_id"]].drop_duplicates()
        known_var["nearby_known_variable"] = True
        summary = summary.merge(known_var, on="candidate_id", how="left")
        summary["nearby_known_variable"] = (
            summary["nearby_known_variable"].astype("boolean").fillna(False).astype(bool)
        )

        if "phot_g_mean_mag" in actual_neighbors.columns:
            bright_mask = (pd.to_numeric(actual_neighbors["phot_g_mean_mag"], errors="coerce") <= 13.0) & (
                pd.to_numeric(actual_neighbors["sep_arcsec"], errors="coerce") <= 5.0
            )
            bright = actual_neighbors.loc[bright_mask, ["candidate_id"]].drop_duplicates()
            bright["bright_close_neighbor"] = True
            summary = summary.merge(bright, on="candidate_id", how="left")
            summary["bright_close_neighbor"] = (
                summary["bright_close_neighbor"].astype("boolean").fillna(False).astype(bool)
            )
        else:
            summary["bright_close_neighbor"] = False

    # Save checkpoint before final output (protects against interruption during summary build)
    if checkpoint_path and not neighbors_long.empty:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        neighbors_long.to_parquet(checkpoint_path, index=False, compression="snappy")

    neighbors_long.to_parquet(out_dir / "neighbors_long.parquet", index=False, compression="zstd")
    summary.to_parquet(out_dir / "neighbors_summary.parquet", index=False, compression="zstd")

    # Clean up checkpoint on success
    if checkpoint_path and Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()

    return neighbors_long, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk neighbor enrichment for candidate tables")
    parser.add_argument("--input", type=Path, required=True, help="Input Parquet with candidate coordinates")
    parser.add_argument("--output-dir", dest="out_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--radius-arcsec", type=float, default=NEIGHBOR_RADIUS_ARCSEC)
    parser.add_argument("--chunk-size", type=int, default=NEIGHBOR_CHUNK_SIZE)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--all-candidates", action="store_true", help="Query all input rows instead of only failed_any=False passers")
    args = parser.parse_args()

    df = read_feature_table(args.input)
    if not getattr(args, "all_candidates", False):
        df = select_passing_candidates_if_present(df, printer=print)
    run_neighbor_enrichment(
        df,
        out_dir=args.out_dir,
        radius_arcsec=args.radius_arcsec,
        chunk_size=args.chunk_size,
        cache_file=args.cache,
    )
    print(f"Neighbor enrichment written to {args.out_dir}")


if __name__ == "__main__":
    main()
