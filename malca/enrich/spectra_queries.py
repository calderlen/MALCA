from __future__ import annotations

import pandas as pd

from malca.config import VIZIER_TAP_URL, SPECTRA_TAP_CHUNK_SIZE, SPECTRA_TAP_TIMEOUT
from malca.enrich.neighbor import _query_catalog_bulk
from malca.enrich.spectra_catalogs import SpectraCatalogSpec
from malca.core.utils import batch_tap_crossmatch


def _row_matches_filter(row: pd.Series, spec: SpectraCatalogSpec) -> bool:
    if spec.filter_col is None:
        return True
    if spec.filter_col not in row.index:
        return not getattr(spec, "filter_not_null", False) and spec.filter_values is None and spec.filter_contains is None
        
    val = row.get(spec.filter_col)
    if getattr(spec, "filter_not_null", False):
        import pandas as pd
        if pd.isna(val) or val == "" or str(val).strip().lower() in ("", "nan"):
            return False
            
    value = str(val or "").strip().lower()
    if not value:
        return False
    if spec.filter_values is not None:
        return value in {v.lower() for v in spec.filter_values}
    if spec.filter_contains is not None:
        return spec.filter_contains.lower() in value
    return True


def _tag_survey_rows(
    matches: pd.DataFrame,
    *,
    survey_specs: dict[str, SpectraCatalogSpec],
    catalog_id: str,
) -> pd.DataFrame:
    if matches.empty:
        return matches

    tagged_frames: list[pd.DataFrame] = []
    for survey_key, spec in survey_specs.items():
        if spec.vizier_id != catalog_id:
            continue
        if spec.filter_col is None and spec.filter_contains is None:
            tagged = matches.copy()
        else:
            mask = matches.apply(lambda row: _row_matches_filter(row, spec), axis=1)
            tagged = matches.loc[mask].copy()
        if tagged.empty:
            continue
        tagged["survey"] = survey_key
        tagged["catalog"] = catalog_id
        tagged_frames.append(tagged)

    if not tagged_frames:
        return pd.DataFrame()
    return pd.concat(tagged_frames, ignore_index=True)


def query_spectra_catalog_group(
    coords_df: pd.DataFrame,
    *,
    survey_specs: dict[str, SpectraCatalogSpec],
    representative: SpectraCatalogSpec,
    radius_arcsec: float,
    chunk_size: int,
    show_progress: bool = False,
    progress_desc: str | None = None,
    tap_timeout: float = SPECTRA_TAP_TIMEOUT,
    tap_chunk_size: int = SPECTRA_TAP_CHUNK_SIZE,
) -> pd.DataFrame:
    """Run one underlying catalog query and fan out rows to survey keys."""
    if coords_df.empty:
        return pd.DataFrame()

    if representative.mode == "tap":
        upload = coords_df[["candidate_id", "ra_deg", "dec_deg"]].copy()
        upload["_idx"] = upload["candidate_id"].astype(str)
        upload = upload.rename(columns={"ra_deg": "ra", "dec_deg": "dec"})
        tap_table = representative.tap_table or f'"{representative.vizier_id}"'
        result = batch_tap_crossmatch(
            upload[["_idx", "ra", "dec"]],
            tap_url=VIZIER_TAP_URL,
            catalog_table=tap_table,
            select_cols=representative.tap_select,
            ra_col=representative.ra_col,
            dec_col=representative.dec_col,
            match_radius_arcsec=radius_arcsec,
            chunk_size=min(int(chunk_size), int(tap_chunk_size)),
            n_workers=2,
            verbose=show_progress,
            desc=progress_desc or f"spectra-tap:{representative.vizier_id}",
            timeout=float(tap_timeout),
        )
        if result.empty:
            return pd.DataFrame()
        result = result.rename(columns={"_idx": "candidate_id"})
        result["candidate_id"] = result["candidate_id"].astype(str)
        matches = result
    else:
        matches = _query_catalog_bulk(
            coords_df,
            catalog=representative.vizier_id,
            radius_arcsec=radius_arcsec,
            chunk_size=chunk_size,
            show_progress=show_progress,
            progress_desc=progress_desc or f"spectra:{representative.vizier_id}",
        )

    if matches.empty:
        return pd.DataFrame()

    if len(survey_specs) == 1:
        only_key = next(iter(survey_specs))
        out = matches.copy()
        out["survey"] = only_key
        if "catalog" not in out.columns:
            out["catalog"] = representative.vizier_id
        return out

    return _tag_survey_rows(matches, survey_specs=survey_specs, catalog_id=representative.vizier_id)
