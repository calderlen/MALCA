from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from malca.enrich.neighbor import _ensure_candidate_id
from malca.enrich.spectra_catalogs import (
    DEFAULT_SPECTRA_CATALOGS,
    grouped_catalog_queries,
    resolve_spectra_catalogs,
)
from malca.enrich.spectra_provenance import merge_external_spectral_provenance
from malca.enrich.spectra_queries import query_spectra_catalog_group
from malca.enrich.transient_spectra import run_transient_spectra_enrichment
from malca.products.candidates import select_passing_candidates_if_present
from malca.config import SPECTRA_RADIUS_ARCSEC, SPECTRA_CHUNK_SIZE, SPECTRA_TAP_CHUNK_SIZE, SPECTRA_TAP_TIMEOUT
from malca.io.table_io import read_feature_table


SPECTRA_REDSHIFT_COLUMNS: tuple[str, ...] = (
    "z",
    "Z",
    "zsp",
    "zspec",
    "z_best",
    "Redshift",
    "cz",
    "Z_COSMO",
    "ZHEL",
)
SPECTRA_TYPE_COLUMNS: tuple[str, ...] = (
    "Class",
    "class",
    "SpType",
    "Type",
    "objtype",
    "SubClass",
    "subClass",
    "SPECTYPE",
    "spectype",
    "MORPHTYPE",
)


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
    catalog = str(row.get("catalog", "")).lower()

    if survey == "tns" or survey == "tns_spectra":
        name = row.get("provenance_name") or row.get("tns_name")
        if name:
            from urllib.parse import quote

            return f"https://www.wis-tns.org/object/{quote(str(name).strip())}"

    if "sdss" in survey or "sdss" in catalog:
        sid = row.get("SpecObjID") or row.get("SpObjID") or row.get("specobjid")
        if sid:
            return f"http://skyserver.sdss.org/dr17/en/tools/explore/Summary.aspx?sid={sid}"

    if "desi" in survey:
        targetid = row.get("TARGETID") or row.get("targetid") or row.get("TARGET_ID")
        if targetid:
            return f"https://www.desi.lbl.gov/documents/etc/DESI-DR1-target-{targetid}/"
        return "https://data.desi.lbl.gov/documents/"

    if "lamost" in survey:
        obsid = row.get("ObsID") or row.get("obsid") or row.get("obsID")
        if obsid:
            return f"https://www.lamost.org/dr7/v2.0/spectrum/view?obsid={obsid}"

    if "galah" in survey:
        sobj = row.get("sobject_id") or row.get("sobjectid") or row.get("SOBJECT_ID")
        if sobj:
            return f"https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR4/spectra/{sobj}.fits"

    if "rave" in survey:
        raveid = row.get("RAVEID") or row.get("raveid") or row.get("PPMXL")
        if raveid:
            return f"https://www.rave-survey.org/ravedr6/dr6/spectrum/{raveid}"
        return "https://www.rave-survey.org/ravedr6/"

    if "sixdf" in survey or "6df" in survey:
        sid = row.get("SeqNum") or row.get("Name") or row.get("6dFGS")
        if sid:
            return f"https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=VII/259/6dfgs&Name={sid}"

    if "2df" in survey:
        name = row.get("Name") or row.get("name") or row.get("TGS")
        if name:
            return f"https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=VII/250/2dfgrs&Name={name}"

    if "manga" in survey:
        plateifu = row.get("PLATEIFU") or row.get("plateifu") or row.get("MANGAID")
        if plateifu:
            return f"https://www.sdss.org/dr17/manga/manga-data/data-cube/?plateifu={plateifu}"
        return "https://www.sdss.org/dr17/manga/"

    if "apogee" in survey:
        apogee_id = row.get("APOGEE_ID") or row.get("apogee_id")
        if apogee_id:
            return f"https://www.sdss.org/dr17/apogee/spectrum/?id={apogee_id}"
        return "https://www.sdss.org/dr17/apogee/"

    if "milliquas" in survey:
        name = row.get("Name") or row.get("provenance_name")
        if name:
            return f"https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=VII/294/catalog&Name={name}"

    if "simbad" in survey:
        ident = row.get("provenance_name") or row.get("simbad_main_id")
        if ident:
            from urllib.parse import quote

            return f"https://simbad.cds.unistra.fr/simbad/sim-id?Ident={quote(str(ident).strip())}"

    if "ned" in survey:
        if pd.notna(row.get("ra_deg")) and pd.notna(row.get("dec_deg")):
            return (
                f"https://ned.ipac.caltech.edu/cgi-bin/objsearch?"
                f"search_type=Near+Position+Search&lon={float(row['ra_deg']):f}&lat={float(row['dec_deg']):f}&radius=0.1"
            )

    if "osc" in survey:
        name = row.get("provenance_name")
        if name:
            from urllib.parse import quote

            return f"https://sne.space/{quote(str(name).strip())}"

    if "gaia" in survey:
        source_id = row.get("source_id") or row.get("SOURCE_ID") or row.get("Source")
        if source_id:
            return f"https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=I/355&Source={source_id}"

    if "cks" in survey:
        return "https://california-planet-search.github.io/cks-website/"

    if "vipers" in survey:
        return "https://archive.eso.org/scienceportal/home?data_collection=VIPERS"

    if "vandels" in survey:
        return "https://archive.eso.org/scienceportal/home?data_collection=VANDELS"

    if "vvds" in survey:
        return "https://archive.eso.org/scienceportal/home?data_collection=VVDS"

    if "zcosmos" in survey:
        return "https://archive.eso.org/scienceportal/home?data_collection=zCOSMOS"

    if "deep2" in survey:
        return "https://deep.ps.uci.edu/DR4/home.html"

    if "wigglez" in survey:
        return "https://wigglez.swin.edu.au/"

    if "gama" in survey:
        return "https://www.gama-survey.org/"

    if "3d_hst" in survey:
        return "https://archive.stsci.edu/prepds/3d-hst/"

    if "s5" in survey:
        return "https://s5collab.github.io/"

    if "efeds" in survey:
        return "https://erosita.mpe.mpg.de/edr/"

    if "ozdes" in survey:
        return "https://ozdes.org/"

    if row.get("link"):
        existing = str(row.get("link") or "").strip()
        if existing:
            return existing

    return None


def _parquet_safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce heterogeneous VizieR catalog columns so parquet export succeeds."""
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype("string")
    return out


def _write_spectra_parquet(df: pd.DataFrame, path: Path, *, compression: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _parquet_safe_frame(df).to_parquet(path, index=False, compression=compression)


def _normalize_spectra_long(spectra_long: pd.DataFrame) -> pd.DataFrame:
    if spectra_long.empty:
        return spectra_long

    out = spectra_long.copy()
    out["candidate_id"] = out["candidate_id"].astype(str)
    if "sep_arcsec" in out.columns:
        out["sep_arcsec"] = pd.to_numeric(out["sep_arcsec"], errors="coerce")

    if "link" not in out.columns:
        out["link"] = out.apply(_generate_link, axis=1)
    else:
        generated = out.apply(_generate_link, axis=1)
        out["link"] = out["link"].fillna("").astype(str)
        out.loc[out["link"].eq(""), "link"] = generated.loc[out["link"].eq("")]

    if "spectrum_redshift" not in out.columns:
        out["spectrum_redshift"] = _first_numeric_column(out, SPECTRA_REDSHIFT_COLUMNS)
    else:
        fallback = _first_numeric_column(out, SPECTRA_REDSHIFT_COLUMNS)
        out["spectrum_redshift"] = pd.to_numeric(out["spectrum_redshift"], errors="coerce").fillna(fallback)

    if "spectrum_spectral_type" not in out.columns:
        out["spectrum_spectral_type"] = _first_text_column(out, SPECTRA_TYPE_COLUMNS)
    else:
        fallback = _first_text_column(out, SPECTRA_TYPE_COLUMNS)
        missing = out["spectrum_spectral_type"].fillna("").astype(str).str.strip().eq("")
        out.loc[missing, "spectrum_spectral_type"] = fallback.loc[missing]

    keep_cols = [
        c
        for c in [
            "candidate_id",
            "survey",
            "catalog",
            "sep_arcsec",
            "link",
            "spectrum_redshift",
            "spectrum_spectral_type",
            "provenance_name",
            "transient_n_spectra",
            "transient_instrument",
        ]
        if c in out.columns
    ]
    keep_cols += [c for c in out.columns if c not in keep_cols]
    return out[keep_cols]


def _summarize_spectra_long(coords: pd.DataFrame, spectra_long: pd.DataFrame) -> pd.DataFrame:
    summary = coords[["candidate_id"]].copy()
    if spectra_long.empty:
        summary["has_spectrum"] = False
        summary["spectrum_sources"] = ""
        summary["spectrum_links"] = ""
        summary["spectrum_redshift"] = np.nan
        summary["spectrum_spectral_type"] = ""
        summary["spectrum_sep_arcsec"] = np.nan
        return summary

    work = spectra_long.copy()
    work["sep_arcsec"] = pd.to_numeric(work.get("sep_arcsec"), errors="coerce")
    work = work.sort_values(["candidate_id", "sep_arcsec"], na_position="last")
    nearest = work.drop_duplicates(subset=["candidate_id"], keep="first")

    by_id = work.groupby("candidate_id")
    sources = by_id["survey"].apply(lambda s: ",".join(sorted({str(x) for x in s.dropna() if str(x)}))).rename("spectrum_sources")
    links_agg = by_id["link"].apply(
        lambda s: ",".join(sorted({str(x) for x in s.dropna() if str(x).strip()}))
    ).rename("spectrum_links")

    summary = summary.merge(sources, on="candidate_id", how="left")
    summary = summary.merge(links_agg, on="candidate_id", how="left")
    summary = summary.merge(
        nearest[["candidate_id", "spectrum_redshift", "spectrum_spectral_type", "sep_arcsec"]].rename(
            columns={"sep_arcsec": "spectrum_sep_arcsec"}
        ),
        on="candidate_id",
        how="left",
    )

    summary["spectrum_sources"] = summary["spectrum_sources"].fillna("")
    summary["spectrum_links"] = summary["spectrum_links"].fillna("")
    summary["spectrum_spectral_type"] = summary["spectrum_spectral_type"].fillna("")
    summary["has_spectrum"] = summary["spectrum_sources"].str.len() > 0
    return summary


def _query_all_catalogs(
    coords_todo: pd.DataFrame,
    *,
    catalogs: dict[str, object],
    radius_arcsec: float,
    chunk_size: int,
    show_progress: bool,
    tap_timeout: float = SPECTRA_TAP_TIMEOUT,
    tap_chunk_size: int = SPECTRA_TAP_CHUNK_SIZE,
) -> pd.DataFrame:
    from malca.enrich.spectra_catalogs import SpectraCatalogSpec

    if isinstance(next(iter(catalogs.values()), None), SpectraCatalogSpec):
        resolved = catalogs  # type: ignore[assignment]
    else:
        resolved = resolve_spectra_catalogs(catalogs)  # type: ignore[arg-type]

    groups = grouped_catalog_queries(resolved)
    frames: list[pd.DataFrame] = []

    group_items = list(groups.items())
    group_iter = tqdm(group_items, desc="Spectra catalogs", disable=not show_progress)
    for _group_id, (rep_key, rep_spec) in group_iter:
        if rep_spec.query_group:
            survey_specs = {
                key: spec
                for key, spec in resolved.items()
                if spec.query_group == rep_spec.query_group and spec.vizier_id == rep_spec.vizier_id
            }
        else:
            survey_specs = {rep_key: rep_spec}

        res = query_spectra_catalog_group(
            coords_todo,
            survey_specs=survey_specs,
            representative=rep_spec,
            radius_arcsec=radius_arcsec,
            chunk_size=chunk_size,
            show_progress=show_progress,
            progress_desc=f"spectra:{rep_key}",
            tap_timeout=tap_timeout,
            tap_chunk_size=tap_chunk_size,
        )
        if not res.empty:
            frames.append(res)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
    merge_provenance_from_input: bool = True,
    run_transient_spectra: bool = False,
    tns_api_key: str | None = None,
    pessto_catalog: Path | None = None,
    ztf_bts_catalog: Path | None = None,
    tap_timeout: float = SPECTRA_TAP_TIMEOUT,
    tap_chunk_size: int = SPECTRA_TAP_CHUNK_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_map = catalogs or DEFAULT_SPECTRA_CATALOGS

    df_use = _ensure_candidate_id(df)
    if not {"candidate_id", "ra_deg", "dec_deg"}.issubset(df_use.columns):
        empty = pd.DataFrame()
        empty.to_parquet(out_dir / "spectra_long.parquet", index=False, compression="zstd")
        empty.to_parquet(out_dir / "spectra_summary.parquet", index=False, compression="zstd")
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
                print(f"[spectra] Loaded checkpoint: {len(cached_ids)} candidates already processed")
        except Exception:
            ckpt_df = pd.DataFrame()

    coords_todo = coords[~coords["candidate_id"].isin(cached_ids)] if cached_ids else coords

    cache_df = pd.DataFrame()
    if not cached_ids and cache_file and Path(cache_file).exists():
        try:
            cache_df = pd.read_parquet(cache_file)
        except Exception:
            cache_df = pd.DataFrame()

    fresh = pd.DataFrame()
    if not coords_todo.empty:
        fresh = _query_all_catalogs(
            coords_todo,
            catalogs=catalog_map,
            radius_arcsec=radius_arcsec,
            chunk_size=chunk_size,
            show_progress=show_progress,
            tap_timeout=tap_timeout,
            tap_chunk_size=tap_chunk_size,
        )
    elif cached_ids:
        print(f"[spectra] All {len(coords)} candidates already in checkpoint, skipping queries")

    parts = [p for p in [ckpt_df, cache_df, fresh] if not p.empty]
    spectra_long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    spectra_long = _normalize_spectra_long(spectra_long)

    if merge_provenance_from_input:
        spectra_long = merge_external_spectral_provenance(df_use, spectra_long)
        spectra_long = _normalize_spectra_long(spectra_long)

    if run_transient_spectra:
        transient = run_transient_spectra_enrichment(
            df_use,
            radius_arcsec=radius_arcsec,
            tns_api_key=tns_api_key,
            pessto_catalog=pessto_catalog,
            ztf_bts_catalog=ztf_bts_catalog,
            show_progress=show_progress,
        )
        if not transient.empty:
            spectra_long = pd.concat([spectra_long, _normalize_spectra_long(transient)], ignore_index=True)

    if cache_file and not spectra_long.empty:
        _write_spectra_parquet(spectra_long, cache_file, compression="snappy")

    summary = _summarize_spectra_long(coords, spectra_long)

    if checkpoint_path and not spectra_long.empty:
        _write_spectra_parquet(spectra_long, checkpoint_path, compression="snappy")

    _write_spectra_parquet(spectra_long, out_dir / "spectra_long.parquet", compression="zstd")
    summary.to_parquet(out_dir / "spectra_summary.parquet", index=False, compression="zstd")

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
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--all-candidates", action="store_true", help="Query all input rows instead of only failed_any=False passers")
    parser.add_argument("--transient-spectra", action="store_true", help="Query OSC / TNS spectrum metadata")
    parser.add_argument("--pessto-catalog", type=Path, default=None)
    parser.add_argument("--ztf-bts-catalog", type=Path, default=None)
    parser.add_argument("--tns-api-key", type=str, default=None, help="TNS API key (or set TNS_API_KEY env)")
    parser.add_argument("--eso-username", type=str, default=None, help="ESO archive username (or set ESO_USERNAME env)")
    parser.add_argument("--eso-password", type=str, default=None, help="ESO archive password (or set ESO_PASSWORD env)")
    parser.add_argument("--download-spectra", action="store_true", help="Also download flux arrays into spectra/ cache")
    parser.add_argument("--tap-timeout", type=float, default=SPECTRA_TAP_TIMEOUT, help="Wall-clock timeout (seconds) per TAP catalog query")
    parser.add_argument("--tap-chunk-size", type=int, default=SPECTRA_TAP_CHUNK_SIZE, help="Max upload chunk size for TAP crossmatches")
    args = parser.parse_args()

    df = read_feature_table(args.input)
    if not getattr(args, "all_candidates", False):
        df = select_passing_candidates_if_present(df, printer=print)

    import os
    tns_key = args.tns_api_key or os.environ.get("TNS_API_KEY")

    long_df, summary = run_spectra_availability(
        df,
        out_dir=args.out_dir,
        radius_arcsec=args.radius_arcsec,
        chunk_size=args.chunk_size,
        cache_file=args.cache,
        checkpoint_path=args.checkpoint,
        run_transient_spectra=args.transient_spectra,
        tns_api_key=tns_key,
        pessto_catalog=args.pessto_catalog,
        ztf_bts_catalog=args.ztf_bts_catalog,
        tap_timeout=args.tap_timeout,
        tap_chunk_size=args.tap_chunk_size,
    )
    print(f"Spectra enrichment written to {args.out_dir}")

    if args.download_spectra and not long_df.empty:
        from malca.enrich.spectrum_config import load_spectrum_fetch_config
        from malca.enrich.spectrum_fetch import prefetch_spectra

        config = load_spectrum_fetch_config(
            eso_username=args.eso_username,
            eso_password=args.eso_password,
            tns_api_key=tns_key,
        )
        cache_dir = Path(args.out_dir) / "spectra"
        prefetch_spectra(long_df, cache_dir=cache_dir, config=config, show_progress=True)
        print(f"Spectrum flux cache written to {cache_dir}")


if __name__ == "__main__":
    main()
