from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from malca.enrich.neighbor import _ensure_candidate_id, _query_catalog_bulk
from malca.table_io import read_parquet_table, write_parquet_table


DEFAULT_SWIFT_CATALOGS: dict[str, str] = {
    "swift_2sxps": "IX/58/2sxps",
    "swift_1sxps": "J/ApJS/210/8/1sxps",
    "swift_uvotssc2": "II/363/uvotssc2",
}

SWIFT_LONG_COLUMNS = ["candidate_id", "swift_catalog", "catalog", "sep_arcsec", "swift_is_uvot", "swift_is_xrt"]


def _is_uv_catalog(name: object) -> bool:
    text = str(name or "").lower()
    return "uv" in text or "uvot" in text


def _is_xray_catalog(name: object) -> bool:
    text = str(name or "").lower()
    return "xps" in text or "xrt" in text or "xray" in text or "sxps" in text


def run_swift_enrichment(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    radius_arcsec: float = 10.0,
    chunk_size: int = 250,
    cache_file: Path | None = None,
    catalogs: dict[str, str] | None = None,
    checkpoint_path: Path | None = None,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crossmatch candidates against Swift UVOT/XRT-style catalogs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalogs = catalogs or DEFAULT_SWIFT_CATALOGS

    df_use = _ensure_candidate_id(df)
    if not {"candidate_id", "ra_deg", "dec_deg"}.issubset(df_use.columns):
        empty = pd.DataFrame(columns=SWIFT_LONG_COLUMNS)
        write_parquet_table(empty, out_dir / "swift_long.parquet")
        write_parquet_table(empty, out_dir / "swift_summary.parquet")
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
                progress_desc=f"swift:{survey}",
            )
            if fresh.empty:
                continue
            fresh["swift_catalog"] = survey
            fresh["swift_is_uvot"] = _is_uv_catalog(survey)
            fresh["swift_is_xrt"] = _is_xray_catalog(survey)
            frames.append(fresh)

    fresh_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    parts = [p for p in (ckpt_df, cache_df, fresh_df) if not p.empty]
    swift_long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SWIFT_LONG_COLUMNS)
    if not swift_long.empty:
        swift_long["candidate_id"] = swift_long["candidate_id"].astype(str)
        if "swift_catalog" not in swift_long.columns:
            swift_long["swift_catalog"] = swift_long.get("catalog", "")
        if "swift_is_uvot" not in swift_long.columns:
            swift_long["swift_is_uvot"] = swift_long["swift_catalog"].map(_is_uv_catalog)
        if "swift_is_xrt" not in swift_long.columns:
            swift_long["swift_is_xrt"] = swift_long["swift_catalog"].map(_is_xray_catalog)

    summary = coords[["candidate_id"]].copy()
    if swift_long.empty:
        summary["swift_uvot_obs"] = False
        summary["swift_uvot_det"] = False
        summary["swift_xrt_det"] = False
        summary["swift_uvot_sep_arcsec"] = np.nan
        summary["swift_xrt_sep_arcsec"] = np.nan
        summary["swift_source_catalogs"] = ""
        summary["swift_status"] = "no_match"
    else:
        by_id = swift_long.groupby("candidate_id")
        catalogs_agg = by_id["swift_catalog"].apply(lambda s: ",".join(sorted({str(x) for x in s.dropna() if str(x)}))).rename("swift_source_catalogs")
        summary = summary.merge(catalogs_agg, on="candidate_id", how="left")

        uv = swift_long.loc[swift_long["swift_is_uvot"].astype(bool)]
        xrt = swift_long.loc[swift_long["swift_is_xrt"].astype(bool)]
        if uv.empty:
            summary["swift_uvot_sep_arcsec"] = np.nan
        else:
            summary = summary.merge(uv.groupby("candidate_id")["sep_arcsec"].min().rename("swift_uvot_sep_arcsec"), on="candidate_id", how="left")
        if xrt.empty:
            summary["swift_xrt_sep_arcsec"] = np.nan
        else:
            summary = summary.merge(xrt.groupby("candidate_id")["sep_arcsec"].min().rename("swift_xrt_sep_arcsec"), on="candidate_id", how="left")
        summary["swift_source_catalogs"] = summary["swift_source_catalogs"].fillna("")
        summary["swift_uvot_obs"] = summary["swift_uvot_sep_arcsec"].notna()
        summary["swift_uvot_det"] = summary["swift_uvot_obs"]
        summary["swift_xrt_det"] = summary["swift_xrt_sep_arcsec"].notna()
        summary["swift_status"] = np.where(summary["swift_source_catalogs"].str.len() > 0, "matched", "no_match")

    if cache_file and not swift_long.empty:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        swift_long.to_parquet(cache_file, index=False, compression="snappy")
    if checkpoint_path and not swift_long.empty:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        swift_long.to_parquet(checkpoint_path, index=False, compression="snappy")

    write_parquet_table(swift_long, out_dir / "swift_long.parquet")
    write_parquet_table(summary, out_dir / "swift_summary.parquet")
    if checkpoint_path and Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
    return swift_long, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk Swift UV/X-ray enrichment")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", dest="out_dir", type=Path, required=True)
    parser.add_argument("--radius-arcsec", type=float, default=10.0)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--cache", type=Path, default=None)
    args = parser.parse_args()

    df = read_parquet_table(args.input)
    run_swift_enrichment(df, out_dir=args.out_dir, radius_arcsec=args.radius_arcsec, chunk_size=args.chunk_size, cache_file=args.cache)
    print(f"Swift enrichment written to {args.out_dir}")


if __name__ == "__main__":
    main()
