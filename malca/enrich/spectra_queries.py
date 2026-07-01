from __future__ import annotations

import pandas as pd
import numpy as np
from astroquery.utils.tap.core import TapPlus

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


def _angular_sep_arcsec(
    ra_deg: float,
    dec_deg: float,
    target_ra_deg: np.ndarray,
    target_dec_deg: np.ndarray,
) -> np.ndarray:
    ra = np.deg2rad(float(ra_deg))
    dec = np.deg2rad(float(dec_deg))
    target_ra = np.deg2rad(target_ra_deg.astype(float))
    target_dec = np.deg2rad(target_dec_deg.astype(float))
    cos_sep = (
        np.sin(dec) * np.sin(target_dec)
        + np.cos(dec) * np.cos(target_dec) * np.cos(ra - target_ra)
    )
    return np.rad2deg(np.arccos(np.clip(cos_sep, -1.0, 1.0))) * 3600.0


def _query_tap_cone_batches(
    coords_df: pd.DataFrame,
    *,
    representative: SpectraCatalogSpec,
    radius_arcsec: float,
    chunk_size: int,
    show_progress: bool,
    progress_desc: str | None,
    status_rows: list[dict] | None,
    status_context: dict,
) -> pd.DataFrame:
    """Query TAP with batched OR cone predicates and assign matches locally."""
    coords = coords_df[["candidate_id", "ra_deg", "dec_deg"]].copy()
    coords["candidate_id"] = coords["candidate_id"].astype(str)
    coords["ra_deg"] = pd.to_numeric(coords["ra_deg"], errors="coerce")
    coords["dec_deg"] = pd.to_numeric(coords["dec_deg"], errors="coerce")
    coords = coords.dropna(subset=["ra_deg", "dec_deg"])
    if coords.empty:
        return pd.DataFrame()

    tap_table = representative.tap_table or f'"{representative.vizier_id}"'
    select_cols = representative.tap_select.strip() or "*"
    select_expr = "c.*" if select_cols == "*" else select_cols
    radius_deg = float(radius_arcsec) / 3600.0
    step = max(1, min(int(chunk_size), 50))
    starts = range(0, len(coords), step)
    iterator = starts
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(starts, total=(len(coords) + step - 1) // step, desc=progress_desc or f"spectra-tap-cone:{representative.vizier_id}")

    frames: list[pd.DataFrame] = []
    tap = TapPlus(url=VIZIER_TAP_URL)

    for start in iterator:
        chunk = coords.iloc[start : start + step].copy()
        predicates = [
            (
                "1=CONTAINS("
                f"POINT('ICRS', c.{representative.ra_col}, c.{representative.dec_col}), "
                f"CIRCLE('ICRS', {float(row.ra_deg):.12g}, {float(row.dec_deg):.12g}, {radius_deg:.12g})"
                ")"
            )
            for row in chunk.itertuples(index=False)
        ]
        query = f"""
        SELECT {select_expr}
        FROM {tap_table} AS c
        WHERE {" OR ".join(predicates)}
        """
        status_base = {
            **status_context,
            "mode": "tap_cone",
            "attempted": int(len(chunk)),
            "matched": 0,
            "chunk_start": int(start),
            "chunk_stop": int(start + len(chunk)),
            "error_message": "",
        }
        try:
            result = tap.launch_job(query, verbose=False).get_results()
            result_df = result.to_pandas() if result is not None and len(result) else pd.DataFrame()
        except Exception as exc:
            if status_rows is not None:
                status_rows.append({**status_base, "status": "error", "error_message": str(exc)})
            continue

        if result_df.empty:
            if status_rows is not None:
                status_rows.append({**status_base, "status": "no_data"})
            continue

        if representative.ra_col not in result_df.columns or representative.dec_col not in result_df.columns:
            if status_rows is not None:
                status_rows.append({
                    **status_base,
                    "status": "error",
                    "error_message": f"TAP cone result missing {representative.ra_col}/{representative.dec_col}",
                })
            continue

        target_ra = chunk["ra_deg"].to_numpy(dtype=float)
        target_dec = chunk["dec_deg"].to_numpy(dtype=float)
        target_ids = chunk["candidate_id"].to_numpy(dtype=str)
        assigned: list[dict] = []
        for _, row in result_df.iterrows():
            sep = _angular_sep_arcsec(row[representative.ra_col], row[representative.dec_col], target_ra, target_dec)
            match_idx = np.flatnonzero(sep <= float(radius_arcsec))
            for idx in match_idx:
                record = row.to_dict()
                record["candidate_id"] = target_ids[idx]
                record["sep_arcsec"] = float(sep[idx])
                assigned.append(record)

        if assigned:
            out = pd.DataFrame(assigned)
            frames.append(out)
            if status_rows is not None:
                status_rows.append({**status_base, "status": "ok", "matched": int(len(out))})
        elif status_rows is not None:
            status_rows.append({**status_base, "status": "no_data"})

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
    status_rows: list[dict] | None = None,
) -> pd.DataFrame:
    """Run one underlying catalog query and fan out rows to survey keys."""
    if coords_df.empty:
        return pd.DataFrame()

    survey_keys = ",".join(sorted(survey_specs))
    status_context = {
        "catalog": representative.vizier_id,
        "survey_keys": survey_keys,
        "query_group": representative.query_group or "",
    }

    if representative.mode == "tap_cone":
        matches = _query_tap_cone_batches(
            coords_df,
            representative=representative,
            radius_arcsec=radius_arcsec,
            chunk_size=min(int(chunk_size), int(tap_chunk_size)),
            show_progress=show_progress,
            progress_desc=progress_desc,
            status_rows=status_rows,
            status_context=status_context,
        )
    elif representative.mode == "tap":
        upload = coords_df[["candidate_id", "ra_deg", "dec_deg"]].copy()
        upload["_idx"] = upload["candidate_id"].astype(str)
        upload = upload.rename(columns={"ra_deg": "ra", "dec_deg": "dec"})
        tap_table = representative.tap_table or f'"{representative.vizier_id}"'
        try:
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
        except Exception as exc:
            if status_rows is not None:
                status_rows.append({
                    **status_context,
                    "mode": "tap",
                    "status": "error",
                    "attempted": int(len(upload)),
                    "matched": 0,
                    "chunk_start": 0,
                    "chunk_stop": int(len(upload)),
                    "error_message": str(exc),
                })
            return pd.DataFrame()
        if result.empty:
            if status_rows is not None:
                status_rows.append({
                    **status_context,
                    "mode": "tap",
                    "status": "no_data",
                    "attempted": int(len(upload)),
                    "matched": 0,
                    "chunk_start": 0,
                    "chunk_stop": int(len(upload)),
                    "error_message": "",
                })
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
            status_rows=status_rows,
            status_context={
                "survey_keys": survey_keys,
                "query_group": representative.query_group or "",
            },
        )

    if matches.empty:
        return pd.DataFrame()

    if len(survey_specs) == 1:
        only_key = next(iter(survey_specs))
        out = matches.copy()
        out["survey"] = only_key
        if "catalog" not in out.columns:
            out["catalog"] = representative.vizier_id
        if representative.mode == "tap" and status_rows is not None:
            status_rows.append({
                **status_context,
                "mode": "tap",
                "status": "ok",
                "attempted": int(len(coords_df)),
                "matched": int(len(out)),
                "chunk_start": 0,
                "chunk_stop": int(len(coords_df)),
                "error_message": "",
            })
        return out

    out = _tag_survey_rows(matches, survey_specs=survey_specs, catalog_id=representative.vizier_id)
    if representative.mode == "tap" and status_rows is not None:
        status_rows.append({
            **status_context,
            "mode": "tap",
            "status": "ok" if not out.empty else "no_data",
            "attempted": int(len(coords_df)),
            "matched": int(len(out)),
            "chunk_start": 0,
            "chunk_stop": int(len(coords_df)),
            "error_message": "",
        })
    return out
