from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from malca.enrich.neighbor import _ensure_candidate_id, _query_catalog_bulk
from malca.io.table_io import read_feature_table, write_parquet_table


DEFAULT_RADIO_CATALOGS: dict[str, str] = {
    "first": "VIII/92/first14",
    "nvss": "VIII/65/nvss",
    "vlass": "VIII/106/vlass1ql",
}

RADIO_LONG_COLUMNS = ["candidate_id", "radio_catalog", "catalog", "sep_arcsec", "radio_flux_mjy"]

RADIO_FLUX_COLUMNS: tuple[str, ...] = (
    "Fpeak",
    "Fint",
    "Speak",
    "Sint",
    "S1.4",
    "Flux",
    "flux",
    "Total_flux",
    "Peak_flux",
)


def _first_numeric_column(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    for col in columns:
        if col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce")
            if values.notna().any():
                return values
    return pd.Series(np.nan, index=frame.index, dtype=float)


def run_radio_enrichment(
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
    """Crossmatch candidates against radio catalogs and summarize AGN-prior evidence."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalogs = catalogs or DEFAULT_RADIO_CATALOGS

    df_use = _ensure_candidate_id(df)
    if not {"candidate_id", "ra_deg", "dec_deg"}.issubset(df_use.columns):
        empty = pd.DataFrame(columns=RADIO_LONG_COLUMNS)
        write_parquet_table(empty, out_dir / "radio_long.parquet")
        write_parquet_table(empty, out_dir / "radio_summary.parquet")
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
                progress_desc=f"radio:{survey}",
            )
            if fresh.empty:
                continue
            fresh["radio_catalog"] = survey
            fresh["radio_flux_mjy"] = _first_numeric_column(fresh, RADIO_FLUX_COLUMNS)
            frames.append(fresh)

    fresh_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    parts = [p for p in (ckpt_df, cache_df, fresh_df) if not p.empty]
    radio_long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=RADIO_LONG_COLUMNS)
    if not radio_long.empty:
        radio_long["candidate_id"] = radio_long["candidate_id"].astype(str)
        if "radio_flux_mjy" not in radio_long.columns:
            radio_long["radio_flux_mjy"] = _first_numeric_column(radio_long, RADIO_FLUX_COLUMNS)

    summary = coords[["candidate_id"]].copy()
    if radio_long.empty:
        summary["radio_det"] = False
        summary["radio_source_catalogs"] = ""
        summary["radio_sep_arcsec"] = np.nan
        summary["radio_flux_mjy"] = np.nan
    else:
        by_id = radio_long.groupby("candidate_id")
        summary = summary.merge(by_id["sep_arcsec"].min().rename("radio_sep_arcsec"), on="candidate_id", how="left")
        summary = summary.merge(by_id["radio_flux_mjy"].max().rename("radio_flux_mjy"), on="candidate_id", how="left")
        catalogs_agg = by_id["radio_catalog"].apply(lambda s: ",".join(sorted({str(x) for x in s.dropna() if str(x)}))).rename("radio_source_catalogs")
        summary = summary.merge(catalogs_agg, on="candidate_id", how="left")
        summary["radio_det"] = summary["radio_source_catalogs"].fillna("").astype(str).str.len() > 0
        summary["radio_source_catalogs"] = summary["radio_source_catalogs"].fillna("")

    if cache_file and not radio_long.empty:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        radio_long.to_parquet(cache_file, index=False, compression="snappy")
    if checkpoint_path and not radio_long.empty:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        radio_long.to_parquet(checkpoint_path, index=False, compression="snappy")

    write_parquet_table(radio_long, out_dir / "radio_long.parquet")
    write_parquet_table(summary, out_dir / "radio_summary.parquet")
    if checkpoint_path and Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
    return radio_long, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk radio counterpart enrichment")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", dest="out_dir", type=Path, required=True)
    parser.add_argument("--radius-arcsec", type=float, default=10.0)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--cache", type=Path, default=None)
    args = parser.parse_args()

    df = read_feature_table(args.input)
    run_radio_enrichment(df, out_dir=args.out_dir, radius_arcsec=args.radius_arcsec, chunk_size=args.chunk_size, cache_file=args.cache)
    print(f"Radio enrichment written to {args.out_dir}")


if __name__ == "__main__":
    main()
