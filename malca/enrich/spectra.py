from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd
from tqdm.auto import tqdm

from malca.enrich.neighbor import _ensure_candidate_id, _query_catalog_bulk
from malca.candidates import select_passing_candidates_if_present
from malca.config import SPECTRA_RADIUS_ARCSEC, SPECTRA_CHUNK_SIZE
from malca.table_io import read_feature_table


DEFAULT_SPECTRA_CATALOGS: dict[str, str] = {
    "sdss_dr17_spec": "V/156/dr17",
    "sixdf_gs": "VII/259/6dfgs",
    "lamost_dr8": "V/164/dr8",
    "galah_dr3": "III/283/galah_dr3",
    "rave_dr5": "III/279/rave_dr5",
}

SPECTRA_REDSHIFT_COLUMNS: tuple[str, ...] = ("z", "Z", "zsp", "zspec", "Redshift", "cz")
SPECTRA_TYPE_COLUMNS: tuple[str, ...] = ("Class", "class", "SpType", "Type", "objtype", "SubClass", "subClass")


def _first_numeric_column(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    for col in columns:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.notna().any():
            return values
    return pd.Series(pd.NA, index=frame.index, dtype="Float64")


def _first_text_column(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    for col in columns:
        if col not in frame.columns:
            continue
        values = frame[col].fillna("").astype(str).str.strip()
        if values.ne("").any():
            return values
    return pd.Series("", index=frame.index, dtype=object)


def _generate_link(row: pd.Series) -> str | None:
    """Generate a direct URL to the spectrum based on catalog metadata."""
    survey = str(row.get("survey", "")).lower()

    # SDSS DR17 (VizieR V/156/dr17 usually has 'SpObjID')
    if "sdss" in survey:
        sid = row.get("SpObjID") or row.get("SpecObjID")
        if sid:
            # Link to the SDSS spectral summary using the spectral ID (sid)
            return f"http://skyserver.sdss.org/dr17/en/tools/explore/Summary.aspx?sid={sid}"

    # LAMOST DR8 (VizieR V/164/dr8 usually has 'ObsID')
    if "lamost" in survey:
        obsid = row.get("ObsID")
        if obsid:
            return f"http://dr8.lamost.org/v2/spectrum/view?obsid={obsid}"

    # GALAH DR3 (VizieR III/283/galah_dr3 has 'sobject_id')
    if "galah" in survey:
        # GALAH Data Central doesn't have a simple public GET link for a spectrum view,
        # but we can link to the Data Central search.
        sobj = row.get("sobject_id")
        if sobj:
            return f"https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR3/spectra/{sobj}.fits"

    if "sixdf" in survey or "6df" in survey:
        sid = row.get("SeqNum") or row.get("Name") or row.get("6dFGS")
        if sid:
            return f"https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=VII/259/6dfgs&Name={sid}"

    return None


def run_spectra_availability(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    radius_arcsec: float = SPECTRA_RADIUS_ARCSEC,
    chunk_size: int = SPECTRA_CHUNK_SIZE,
    cache_file: Path | None = None,
    catalogs: dict[str, str] | None = None,
    checkpoint_path: Path | None = None,
    show_progress: bool = False,
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

    # Load checkpoint to skip already-processed candidates
    ckpt_df = pd.DataFrame()
    cached_ids: set[str] = set()
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            ckpt_df = pd.read_parquet(checkpoint_path)
            if "candidate_id" in ckpt_df.columns:
                cached_ids = set(ckpt_df["candidate_id"].astype(str))
                print(f"[spectra] Loaded checkpoint: {len(cached_ids)} candidates already processed")
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

    frames: list[pd.DataFrame] = []
    if not coords_todo.empty:
        catalog_items = list(catalogs.items())
        catalog_iter = tqdm(catalog_items, desc="Spectra catalogs", disable=not show_progress)
        for survey, catalog_id in catalog_iter:
            res = _query_catalog_bulk(
                coords_todo,
                catalog=catalog_id,
                radius_arcsec=radius_arcsec,
                chunk_size=chunk_size,
                show_progress=show_progress,
                progress_desc=f"spectra:{survey}",
            )
            if res.empty:
                continue
            res["survey"] = survey
            frames.append(res)
    elif cached_ids:
        print(f"[spectra] All {len(coords)} candidates already in checkpoint, skipping queries")

    fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Combine checkpoint + cache + fresh results
    parts = [p for p in [ckpt_df, cache_df, fresh] if not p.empty]
    spectra_long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not spectra_long.empty:
        spectra_long["candidate_id"] = spectra_long["candidate_id"].astype(str)
        # Generate links before filtering columns
        spectra_long["link"] = spectra_long.apply(_generate_link, axis=1)
        spectra_long["spectrum_redshift"] = _first_numeric_column(spectra_long, SPECTRA_REDSHIFT_COLUMNS)
        spectra_long["spectrum_spectral_type"] = _first_text_column(spectra_long, SPECTRA_TYPE_COLUMNS)

        keep_cols = [c for c in ["candidate_id", "survey", "catalog", "sep_arcsec", "link", "spectrum_redshift", "spectrum_spectral_type"] if c in spectra_long.columns]
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
        
        links_agg = by_id["link"].apply(lambda s: ",".join(sorted({str(x) for x in s.dropna() if x}))).rename("spectrum_links")

        summary = summary.merge(sources, on="candidate_id", how="left")
        summary = summary.merge(links_agg, on="candidate_id", how="left")
        summary = summary.merge(by_id["spectrum_redshift"].first().rename("spectrum_redshift"), on="candidate_id", how="left")
        summary = summary.merge(by_id["spectrum_spectral_type"].first().rename("spectrum_spectral_type"), on="candidate_id", how="left")
        
        summary["spectrum_sources"] = summary["spectrum_sources"].fillna("")
        summary["spectrum_links"] = summary["spectrum_links"].fillna("")
        summary["spectrum_spectral_type"] = summary["spectrum_spectral_type"].fillna("")
        summary["has_spectrum"] = summary["spectrum_sources"].str.len() > 0

    # Save checkpoint before final output
    if checkpoint_path and not spectra_long.empty:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        spectra_long.to_parquet(checkpoint_path, index=False, compression="snappy")

    spectra_long.to_parquet(out_dir / "spectra_long.parquet", index=False, compression="zstd")
    summary.to_parquet(out_dir / "spectra_summary.parquet", index=False, compression="zstd")

    # Clean up checkpoint on success
    if checkpoint_path and Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()

    return spectra_long, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk spectra-availability enrichment")
    parser.add_argument("--input", type=Path, required=True, help="Input Parquet with candidate coordinates")
    parser.add_argument("--output-dir", dest="out_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--radius-arcsec", type=float, default=SPECTRA_RADIUS_ARCSEC)
    parser.add_argument("--chunk-size", type=int, default=SPECTRA_CHUNK_SIZE)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--all-candidates", action="store_true", help="Query all input rows instead of only failed_any=False passers")
    args = parser.parse_args()

    df = read_feature_table(args.input)
    if not getattr(args, "all_candidates", False):
        df = select_passing_candidates_if_present(df, printer=print)
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
