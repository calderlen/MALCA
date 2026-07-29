from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from malca.enrich.apogee import (
    APOGEE_SUMMARY_COLUMNS,
    apogee_summary_column,
    apogee_summary_columns,
    normalize_apogee_metadata_columns,
)
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
SPECTRA_QUERY_STATUS_COLUMNS: tuple[str, ...] = (
    "catalog",
    "survey_keys",
    "query_group",
    "mode",
    "status",
    "attempted",
    "matched",
    "chunk_start",
    "chunk_stop",
    "error_message",
)
SPECTRUM_RECORD_ID_COLUMNS: tuple[str, ...] = (
    "SpecObjID", "SpObjID", "specobjid", "ObsID", "obsid", "obsID",
    "spectrum_id", "visit_id", "exposure_id", "PLATEIFU", "plateifu",
    "sobject_id", "SOBJECT_ID", "RAVEID", "APOGEE_ID", "ID",
    "TARGETID", "TargetID", "targetid", "source_id", "SOURCE_ID", "Source",
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
        targetid = row.get("TARGETID") or row.get("TargetID") or row.get("targetid") or row.get("TARGET_ID")
        if targetid:
            return f"https://www.desi.lbl.gov/documents/etc/DESI-DR1-target-{targetid}/"
        return "https://data.desi.lbl.gov/documents/"

    if "lamost" in survey:
        obsid = row.get("ObsID") or row.get("obsid") or row.get("obsID")
        if obsid:
            return f"https://www.lamost.org/dr7/v2.0/spectrum/view?obsid={obsid}"

    if "galah" in survey:
        sobj = row.get("sobject_id") or row.get("sobjectid") or row.get("SOBJECT_ID") or row.get("GALAH")
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
        apogee_id = row.get("APOGEE_ID") or row.get("apogee_id") or row.get("ID") or row.get("id")
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

    existing_link = row.get("link")
    if existing_link is not None and pd.notna(existing_link):
        existing = str(existing_link or "").strip()
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


def _spectra_query_status_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=SPECTRA_QUERY_STATUS_COLUMNS)
    out = pd.DataFrame(rows)
    for col in SPECTRA_QUERY_STATUS_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[list(SPECTRA_QUERY_STATUS_COLUMNS) + [col for col in out.columns if col not in SPECTRA_QUERY_STATUS_COLUMNS]]


def _spectrum_record_identity(row: pd.Series) -> tuple[str, str]:
    object_id = ""
    for column in SPECTRUM_RECORD_ID_COLUMNS:
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
            # Prefer the strongest survey-specific record identifier.  Using
            # every populated identifier makes a key unstable when a cache
            # refresh merely adds (for example) a source_id column.
            object_id = f"{column}:{text}"
            break
    if not object_id:
        for column in ("provenance_name", "link"):
            value = row.get(column)
            if value is not None and pd.notna(value) and str(value).strip():
                object_id = f"{column}:{str(value).strip()}"
                break
    if not object_id:
        sep = pd.to_numeric(row.get("sep_arcsec"), errors="coerce")
        object_id = f"sep:{float(sep):.6f}" if np.isfinite(sep) else "unidentified"
    token = f"{row.get('survey', '')}|{row.get('catalog', '')}|{object_id}"
    return object_id, hashlib.sha1(token.encode("utf-8")).hexdigest()[:24]


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

    out = normalize_apogee_metadata_columns(out)

    identities = out.apply(_spectrum_record_identity, axis=1)
    out["spectrum_object_id"] = [item[0] for item in identities]
    out["spectrum_record_key"] = [item[1] for item in identities]
    provenance_only = out.get("catalog", pd.Series("", index=out.index)).fillna("").astype(str).str.startswith("provenance:")
    transient_count = pd.to_numeric(out.get("transient_n_spectra", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["spectrum_record_status"] = np.where(
        provenance_only & ~(transient_count > 0), "metadata_only", "available"
    )
    out = (
        out.sort_values(
            ["candidate_id", "survey", "spectrum_record_key", "sep_arcsec"],
            na_position="last",
            kind="mergesort",
        )
        .drop_duplicates(
            subset=["candidate_id", "survey", "spectrum_record_key"],
            keep="first",
        )
        .reset_index(drop=True)
    )

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
            "spectrum_object_id",
            "spectrum_record_key",
            "spectrum_record_status",
        ]
        if c in out.columns
    ]
    keep_cols += [c for c in out.columns if c not in keep_cols]
    return out[keep_cols]


def _summarize_spectra_long(coords: pd.DataFrame, spectra_long: pd.DataFrame) -> pd.DataFrame:
    summary = coords[["candidate_id"]].copy()
    if spectra_long.empty:
        summary["has_spectrum"] = False
        summary["has_spectral_metadata"] = False
        summary["spectrum_status"] = "no_match"
        summary["spectrum_n_unique_records"] = 0
        summary["spectrum_redshift_conflict"] = False
        summary["spectrum_spectral_type_conflict"] = False
        summary["spectrum_conflict_details_json"] = "{}"
        summary["spectrum_sources"] = ""
        summary["spectrum_links"] = ""
        summary["spectrum_redshift"] = np.nan
        summary["spectrum_spectral_type"] = ""
        summary["spectrum_sep_arcsec"] = np.nan
        for col in apogee_summary_columns():
            summary[col] = pd.NA
        return summary

    work = normalize_apogee_metadata_columns(spectra_long.copy())
    if "spectrum_record_key" not in work.columns or "spectrum_record_status" not in work.columns:
        work = _normalize_spectra_long(work)
    work["sep_arcsec"] = pd.to_numeric(work.get("sep_arcsec"), errors="coerce")
    work = work.sort_values(["candidate_id", "sep_arcsec"], na_position="last")
    nearest = work.drop_duplicates(subset=["candidate_id"], keep="first")
    redshift_rows = work.loc[pd.to_numeric(work["spectrum_redshift"], errors="coerce").notna()]
    redshift_nearest = redshift_rows.drop_duplicates(subset=["candidate_id"], keep="first")
    type_rows = work.loc[work["spectrum_spectral_type"].fillna("").astype(str).str.strip().ne("")]
    type_nearest = type_rows.drop_duplicates(subset=["candidate_id"], keep="first")

    by_id = work.groupby("candidate_id")
    sources = by_id["survey"].apply(lambda s: ",".join(sorted({str(x) for x in s.dropna() if str(x)}))).rename("spectrum_sources")
    links_agg = by_id["link"].apply(
        lambda s: ",".join(sorted({str(x) for x in s.dropna() if str(x).strip()}))
    ).rename("spectrum_links")

    summary = summary.merge(sources, on="candidate_id", how="left")
    summary = summary.merge(links_agg, on="candidate_id", how="left")
    summary = summary.merge(
        nearest[["candidate_id", "sep_arcsec"]].rename(columns={"sep_arcsec": "spectrum_sep_arcsec"}),
        on="candidate_id", how="left",
    )
    summary = summary.merge(
        redshift_nearest[["candidate_id", "spectrum_redshift"]], on="candidate_id", how="left",
    )
    summary = summary.merge(
        type_nearest[["candidate_id", "spectrum_spectral_type"]], on="candidate_id", how="left",
    )

    summary["spectrum_sources"] = summary["spectrum_sources"].fillna("")
    summary["spectrum_links"] = summary["spectrum_links"].fillna("")
    summary["spectrum_spectral_type"] = summary["spectrum_spectral_type"].fillna("")
    record_counts = work.groupby("candidate_id")["spectrum_record_key"].nunique().rename("spectrum_n_unique_records")
    available = (
        work.loc[work["spectrum_record_status"].astype(str) == "available", ["candidate_id"]]
        .drop_duplicates().assign(has_spectrum=True)
    )
    summary = summary.merge(record_counts, on="candidate_id", how="left")
    summary = summary.merge(available, on="candidate_id", how="left")
    summary["spectrum_n_unique_records"] = summary["spectrum_n_unique_records"].fillna(0).astype(int)
    summary["has_spectral_metadata"] = summary["spectrum_n_unique_records"] > 0
    summary["has_spectrum"] = summary["has_spectrum"].astype("boolean").fillna(False).astype(bool)
    summary["spectrum_status"] = np.where(
        summary["has_spectrum"], "available",
        np.where(summary["has_spectral_metadata"], "metadata_only", "no_match"),
    )

    conflict_rows: list[dict[str, object]] = []
    for candidate_id, group in work.groupby("candidate_id"):
        redshifts = pd.to_numeric(group["spectrum_redshift"], errors="coerce").dropna().to_numpy(dtype=float)
        types = sorted({
            " ".join(str(value).strip().upper().split())
            for value in group["spectrum_spectral_type"].dropna()
            if str(value).strip()
        })
        z_conflict = False
        z_min = z_max = np.nan
        if redshifts.size:
            z_min, z_max = float(np.nanmin(redshifts)), float(np.nanmax(redshifts))
            tolerance = max(0.001, 0.05 * abs(float(np.nanmedian(redshifts))))
            z_conflict = (z_max - z_min) > tolerance
        type_conflict = len(types) > 1
        conflict_rows.append({
            "candidate_id": str(candidate_id),
            "spectrum_redshift_conflict": bool(z_conflict),
            "spectrum_spectral_type_conflict": bool(type_conflict),
            "spectrum_conflict_details_json": json.dumps(
                {
                    "redshift_min": None if not np.isfinite(z_min) else z_min,
                    "redshift_max": None if not np.isfinite(z_max) else z_max,
                    "spectral_types": types,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        })
    summary = summary.merge(pd.DataFrame(conflict_rows), on="candidate_id", how="left")
    summary["spectrum_redshift_conflict"] = summary["spectrum_redshift_conflict"].astype("boolean").fillna(False).astype(bool)
    summary["spectrum_spectral_type_conflict"] = summary["spectrum_spectral_type_conflict"].astype("boolean").fillna(False).astype(bool)
    summary["spectrum_conflict_details_json"] = summary["spectrum_conflict_details_json"].fillna("{}")

    apogee = work[work["survey"].astype(str).str.contains("apogee", case=False, na=False)]
    if not apogee.empty:
        apogee_nearest = apogee.drop_duplicates(subset=["candidate_id"], keep="first")
        apogee_cols = [col for col in APOGEE_SUMMARY_COLUMNS if col in apogee_nearest.columns]
        if apogee_cols:
            apogee_summary = apogee_nearest[["candidate_id", *apogee_cols]].rename(
                columns={col: apogee_summary_column(col) for col in apogee_cols}
            )
            summary = summary.merge(apogee_summary, on="candidate_id", how="left")
    for col in apogee_summary_columns():
        if col not in summary.columns:
            summary[col] = pd.NA
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
    status_rows: list[dict] | None = None,
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
            status_rows=status_rows,
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
        _spectra_query_status_frame([]).to_parquet(out_dir / "spectra_query_status.parquet", index=False, compression="zstd")
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
    status_rows: list[dict] = []
    if not coords_todo.empty:
        fresh = _query_all_catalogs(
            coords_todo,
            catalogs=catalog_map,
            radius_arcsec=radius_arcsec,
            chunk_size=chunk_size,
            show_progress=show_progress,
            tap_timeout=tap_timeout,
            tap_chunk_size=tap_chunk_size,
            status_rows=status_rows,
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
            spectra_long = _normalize_spectra_long(spectra_long)

    if cache_file and not spectra_long.empty:
        _write_spectra_parquet(spectra_long, cache_file, compression="snappy")

    summary = _summarize_spectra_long(coords, spectra_long)

    if checkpoint_path and not spectra_long.empty:
        _write_spectra_parquet(spectra_long, checkpoint_path, compression="snappy")

    _write_spectra_parquet(spectra_long, out_dir / "spectra_long.parquet", compression="zstd")
    summary.to_parquet(out_dir / "spectra_summary.parquet", index=False, compression="zstd")
    _write_spectra_parquet(_spectra_query_status_frame(status_rows), out_dir / "spectra_query_status.parquet", compression="zstd")

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
