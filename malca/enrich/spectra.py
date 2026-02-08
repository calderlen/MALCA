from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd

from malca.enrich.neighbor import _ensure_candidate_id, _query_catalog_bulk
from malca.config.config_characterize import SPECTRA_RADIUS_ARCSEC, SPECTRA_CHUNK_SIZE


DEFAULT_SPECTRA_CATALOGS: dict[str, str] = {
    "sdss_dr16_spec": "V/154/sdss16",
    "lamost": "V/164/dr8",
    "rave_dr5": "III/279/rave_dr5",
}


def run_spectra_availability(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    radius_arcsec: float = SPECTRA_RADIUS_ARCSEC,
    chunk_size: int = SPECTRA_CHUNK_SIZE,
    cache_file: Path | None = None,
    catalogs: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalogs = catalogs or DEFAULT_SPECTRA_CATALOGS

    df_use = _ensure_candidate_id(df)
    if not {"candidate_id", "ra_deg", "dec_deg"}.issubset(df_use.columns):
        empty = pd.DataFrame()
        empty.to_parquet(out_dir / "spectra_long.parquet", index=False, compression="zstd")
        empty.to_parquet(out_dir / "spectra_summary.parquet", index=False, compression="zstd")
        return empty, empty

    coords = df_use[["candidate_id", "ra_deg", "dec_deg"]].dropna(subset=["ra_deg", "dec_deg"]).copy()
    coords = coords.drop_duplicates(subset=["candidate_id"])
    coords["candidate_id"] = coords["candidate_id"].astype(str)

    cache_df = pd.DataFrame()
    if cache_file and Path(cache_file).exists():
        try:
            cache_df = pd.read_parquet(cache_file)
        except Exception:
            cache_df = pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for survey, catalog_id in catalogs.items():
        res = _query_catalog_bulk(
            coords,
            catalog=catalog_id,
            radius_arcsec=radius_arcsec,
            chunk_size=chunk_size,
        )
        if res.empty:
            continue
        res["survey"] = survey
        frames.append(res)

    fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    spectra_long = pd.concat([cache_df, fresh], ignore_index=True) if not cache_df.empty else fresh
    if not spectra_long.empty:
        spectra_long["candidate_id"] = spectra_long["candidate_id"].astype(str)
        keep_cols = [c for c in ["candidate_id", "survey", "catalog", "sep_arcsec"] if c in spectra_long.columns]
        keep_cols += [c for c in spectra_long.columns if c not in keep_cols]
        spectra_long = spectra_long[keep_cols]

    if cache_file and not spectra_long.empty:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        spectra_long.to_parquet(cache_file, index=False, compression="snappy")

    summary = coords[["candidate_id"]].copy()
    if spectra_long.empty:
        summary["has_spectrum"] = False
        summary["spectrum_sources"] = ""
        summary["spectrum_links"] = ""
    else:
        by_id = spectra_long.groupby("candidate_id")
        sources = by_id["survey"].apply(lambda s: ",".join(sorted({str(x) for x in s.dropna()}))).rename("spectrum_sources")
        summary = summary.merge(sources, on="candidate_id", how="left")
        summary["spectrum_sources"] = summary["spectrum_sources"].fillna("")
        summary["has_spectrum"] = summary["spectrum_sources"].str.len() > 0
        summary["spectrum_links"] = ""

    spectra_long.to_parquet(out_dir / "spectra_long.parquet", index=False, compression="zstd")
    summary.to_parquet(out_dir / "spectra_summary.parquet", index=False, compression="zstd")
    return spectra_long, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk spectra-availability enrichment")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV/Parquet with candidate coordinates")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--radius-arcsec", type=float, default=SPECTRA_RADIUS_ARCSEC)
    parser.add_argument("--chunk-size", type=int, default=SPECTRA_CHUNK_SIZE)
    parser.add_argument("--cache", type=Path, default=None)
    args = parser.parse_args()

    if args.input.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    run_spectra_availability(
        df,
        out_dir=args.out_dir,
        radius_arcsec=args.radius_arcsec,
        chunk_size=args.chunk_size,
        cache_file=args.cache,
    )
    print(f"Spectra enrichment written to {args.out_dir}")


if __name__ == "__main__":
    main()
