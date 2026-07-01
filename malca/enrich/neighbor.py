from __future__ import annotations

from pathlib import Path

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
        table = Table.from_pandas(chunk[["candidate_id", "ra_deg", "dec_deg"]].rename(columns={"ra_deg": "ra", "dec_deg": "dec"}))
        try:
            res = XMatch.query(
                cat1=table,
                cat2=f"vizier:{catalog}",
                max_distance=float(radius_arcsec) * u.arcsec,
                colRA1="ra",
                colDec1="dec",
            )
        except Exception as e:
            import logging
            logging.warning(f"XMatch query failed for {catalog}: {e}")
            if status_rows is not None:
                status_rows.append({
                    **status_base,
                    "status": "error",
                    "error_message": str(e),
                })
            continue
        if len(res) == 0:
            if status_rows is not None:
                status_rows.append({**status_base, "status": "no_data"})
            continue
        out = res.to_pandas()
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
        keep_cols = [c for c in ["candidate_id", "catalog", "sep_arcsec", "phot_g_mean_mag", "VarType", "Type"] if c in neighbors_long.columns]
        keep_cols += [c for c in neighbors_long.columns if c not in keep_cols]
        neighbors_long = neighbors_long[keep_cols]
        if "candidate_id" in neighbors_long.columns:
            neighbors_long["candidate_id"] = neighbors_long["candidate_id"].astype(str)

    if cache_file and not neighbors_long.empty:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        neighbors_long.to_parquet(cache_file, index=False, compression="snappy")

    summary = coords[["candidate_id"]].copy()
    if neighbors_long.empty:
        summary["neighbor_count"] = 0
        summary["nearest_sep_arcsec"] = np.nan
        summary["nearby_known_variable"] = False
        summary["bright_close_neighbor"] = False
        summary["local_density_n_15as"] = 0
    else:
        grp = neighbors_long.groupby("candidate_id")
        summary = summary.merge(grp.size().rename("neighbor_count"), on="candidate_id", how="left")
        summary = summary.merge(grp["sep_arcsec"].min().rename("nearest_sep_arcsec"), on="candidate_id", how="left")
        summary["neighbor_count"] = summary["neighbor_count"].fillna(0).astype(int)
        summary["local_density_n_15as"] = summary["neighbor_count"]

        known_var_mask = neighbors_long["catalog"].astype(str).str.contains("vsx", case=False, na=False)
        known_var = neighbors_long.loc[known_var_mask, ["candidate_id"]].drop_duplicates()
        known_var["nearby_known_variable"] = True
        summary = summary.merge(known_var, on="candidate_id", how="left")
        summary["nearby_known_variable"] = (
            summary["nearby_known_variable"].astype("boolean").fillna(False).astype(bool)
        )

        if "phot_g_mean_mag" in neighbors_long.columns:
            bright_mask = (pd.to_numeric(neighbors_long["phot_g_mean_mag"], errors="coerce") <= 13.0) & (
                pd.to_numeric(neighbors_long["sep_arcsec"], errors="coerce") <= 5.0
            )
            bright = neighbors_long.loc[bright_mask, ["candidate_id"]].drop_duplicates()
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
