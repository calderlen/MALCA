from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from malca.enrich.neighbor import _ensure_candidate_id, _query_catalog_bulk
from malca.table_io import read_parquet_table, write_parquet_table


DEFAULT_HOST_CATALOGS: dict[str, str] = {
    "ps1": "II/349/ps1",
    "sdss_dr16": "V/154/sdss16",
    "legacy_dr10": "VII/292/ls-dr10",
}

HOST_LONG_COLUMNS = ["candidate_id", "host_source", "catalog", "sep_arcsec"]


def host_nuclear_score(offset_arcsec: pd.Series, *, good_arcsec: float = 0.5, max_arcsec: float = 3.0) -> pd.Series:
    """Score whether a transient/source position is consistent with a catalog host centroid."""
    values = pd.to_numeric(offset_arcsec, errors="coerce")
    score = 1.0 - ((values - float(good_arcsec)) / max(float(max_arcsec) - float(good_arcsec), 1e-6))
    return score.clip(0.0, 1.0).fillna(0.0)


def _nearest_by_candidate(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty or "candidate_id" not in matches.columns or "sep_arcsec" not in matches.columns:
        return pd.DataFrame()
    work = matches.copy()
    work["sep_arcsec"] = pd.to_numeric(work["sep_arcsec"], errors="coerce")
    work = work.sort_values(["candidate_id", "sep_arcsec"], na_position="last")
    return work.drop_duplicates(subset=["candidate_id"], keep="first")


def run_host_association(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    radius_arcsec: float = 5.0,
    nuclear_good_arcsec: float = 0.5,
    nuclear_max_arcsec: float = 3.0,
    chunk_size: int = 250,
    cache_file: Path | None = None,
    catalogs: dict[str, str] | None = None,
    checkpoint_path: Path | None = None,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Associate nuclear candidates with catalog host centroids.

    This v1 host layer intentionally uses catalog centroids rather than image
    fitting.  The output offset should be treated as a review/ranking prior.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalogs = catalogs or DEFAULT_HOST_CATALOGS

    df_use = _ensure_candidate_id(df)
    if not {"candidate_id", "ra_deg", "dec_deg"}.issubset(df_use.columns):
        empty = pd.DataFrame(columns=HOST_LONG_COLUMNS)
        write_parquet_table(empty, out_dir / "host_long.parquet")
        write_parquet_table(empty, out_dir / "host_summary.parquet")
        return empty, empty

    coords = df_use[["candidate_id", "ra_deg", "dec_deg"]].dropna(subset=["ra_deg", "dec_deg"]).copy()
    coords = coords.drop_duplicates(subset=["candidate_id"])
    coords["candidate_id"] = coords["candidate_id"].astype(str)

    ckpt_df = pd.DataFrame()
    cached_ids: set[str] = set()
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            ckpt_df = pd.read_parquet(checkpoint_path)
            if "candidate_id" in ckpt_df.columns:
                cached_ids = set(ckpt_df["candidate_id"].astype(str))
        except Exception:
            ckpt_df = pd.DataFrame()

    cache_df = pd.DataFrame()
    if not cached_ids and cache_file and Path(cache_file).exists():
        try:
            cache_df = pd.read_parquet(cache_file)
        except Exception:
            cache_df = pd.DataFrame()

    coords_todo = coords[~coords["candidate_id"].isin(cached_ids)] if cached_ids else coords
    frames: list[pd.DataFrame] = []
    if not coords_todo.empty:
        for survey, catalog_id in catalogs.items():
            fresh = _query_catalog_bulk(
                coords_todo,
                catalog=catalog_id,
                radius_arcsec=radius_arcsec,
                chunk_size=chunk_size,
                show_progress=show_progress,
                progress_desc=f"host:{survey}",
            )
            if fresh.empty:
                continue
            fresh["host_source"] = survey
            frames.append(fresh)

    fresh_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    parts = [p for p in (ckpt_df, cache_df, fresh_df) if not p.empty]
    host_long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=HOST_LONG_COLUMNS)
    if not host_long.empty:
        host_long["candidate_id"] = host_long["candidate_id"].astype(str)
        if "host_source" not in host_long.columns:
            host_long["host_source"] = host_long.get("catalog", "")

    nearest = _nearest_by_candidate(host_long)
    summary = coords[["candidate_id"]].copy()
    if nearest.empty:
        summary["host_match"] = False
        summary["host_source"] = ""
        summary["host_sep_arcsec"] = np.nan
        summary["nuclear_offset_arcsec"] = np.nan
        summary["host_nuclear_score"] = 0.0
        summary["host_assoc_status"] = "no_match"
    else:
        keep = nearest[[c for c in ("candidate_id", "host_source", "sep_arcsec") if c in nearest.columns]].copy()
        keep = keep.rename(columns={"sep_arcsec": "host_sep_arcsec"})
        summary = summary.merge(keep, on="candidate_id", how="left")
        summary["host_match"] = summary["host_sep_arcsec"].notna()
        summary["host_source"] = summary["host_source"].fillna("")
        summary["nuclear_offset_arcsec"] = summary["host_sep_arcsec"]
        summary["host_nuclear_score"] = host_nuclear_score(
            summary["nuclear_offset_arcsec"],
            good_arcsec=nuclear_good_arcsec,
            max_arcsec=nuclear_max_arcsec,
        )
        summary["host_assoc_status"] = np.where(summary["host_match"], "matched", "no_match")

    if cache_file and not host_long.empty:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        host_long.to_parquet(cache_file, index=False, compression="snappy")
    if checkpoint_path and not host_long.empty:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        host_long.to_parquet(checkpoint_path, index=False, compression="snappy")

    write_parquet_table(host_long, out_dir / "host_long.parquet")
    write_parquet_table(summary, out_dir / "host_summary.parquet")
    if checkpoint_path and Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
    return host_long, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk nuclear host-association enrichment")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", dest="out_dir", type=Path, required=True)
    parser.add_argument("--radius-arcsec", type=float, default=5.0)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--cache", type=Path, default=None)
    args = parser.parse_args()

    df = read_parquet_table(args.input)
    run_host_association(df, out_dir=args.out_dir, radius_arcsec=args.radius_arcsec, chunk_size=args.chunk_size, cache_file=args.cache)
    print(f"Host association written to {args.out_dir}")


if __name__ == "__main__":
    main()
