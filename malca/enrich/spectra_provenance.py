from __future__ import annotations

from urllib.parse import quote

import numpy as np
import pandas as pd

from malca.enrich.neighbor import _ensure_candidate_id


PROVENANCE_SOURCE_SPECS: tuple[dict[str, object], ...] = (
    {
        "survey": "tns",
        "name_col": "tns_name",
        "type_col": "tns_type",
        "z_col": "tns_redshift",
        "sep_col": None,
        "link_builder": lambda row: (
            f"https://www.wis-tns.org/object/{quote(str(row.get('tns_name') or '').strip())}"
            if str(row.get("tns_name") or "").strip()
            else None
        ),
    },
    {
        "survey": "simbad",
        "name_col": "simbad_main_id",
        "type_col": "simbad_otype",
        "z_col": "simbad_redshift",
        "sep_col": "simbad_sep_arcsec",
        "alt_type_col": "simbad_sp_type",
        "link_builder": lambda row: (
            f"https://simbad.cds.unistra.fr/simbad/sim-id?Ident={quote(str(row.get('simbad_main_id') or '').strip())}"
            if str(row.get("simbad_main_id") or "").strip()
            else None
        ),
    },
    {
        "survey": "milliquas",
        "name_col": "milliquas_name",
        "type_col": "milliquas_type",
        "z_col": "milliquas_z",
        "sep_col": "milliquas_sep_arcsec",
        "link_builder": lambda row: (
            "https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=VII/294/catalog"
        ),
    },
    {
        "survey": "ned",
        "name_col": "ned_name",
        "type_col": "ned_type",
        "z_col": "ned_redshift",
        "sep_col": "ned_sep_arcsec",
        "link_builder": lambda row: (
            f"https://ned.ipac.caltech.edu/cgi-bin/objsearch?search_type=Near+Position+Search&lon={float(row.get('ra_deg')):f}&lat={float(row.get('dec_deg')):f}&radius=0.1"
            if pd.notna(row.get("ra_deg")) and pd.notna(row.get("dec_deg"))
            else None
        ),
    },
    {
        "survey": "desi_clagn",
        "name_col": "clagn_name",
        "type_col": "clagn_type",
        "z_col": "clagn_redshift",
        "sep_col": "clagn_sep_arcsec",
        "link_builder": lambda row: "https://data.desi.lbl.gov/documents/",
    },
)


def _has_value(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    return text.ne("") & ~text.str.lower().isin({"nan", "none", "<na>"})


def merge_external_spectral_provenance(
    df: pd.DataFrame,
    spectra_long: pd.DataFrame,
) -> pd.DataFrame:
    """Append spectral metadata already present on the input frame (no new queries)."""
    frame = _ensure_candidate_id(df)
    if frame.empty:
        return spectra_long

    rows: list[dict[str, object]] = []
    for spec in PROVENANCE_SOURCE_SPECS:
        survey = str(spec["survey"])
        name_col = str(spec["name_col"])
        type_col = str(spec["type_col"])
        z_col = str(spec["z_col"])
        sep_col = spec.get("sep_col")
        alt_type_col = spec.get("alt_type_col")
        link_builder = spec["link_builder"]

        if name_col not in frame.columns and z_col not in frame.columns and type_col not in frame.columns:
            continue

        name_series = frame[name_col] if name_col in frame.columns else pd.Series("", index=frame.index)
        type_series = frame[type_col].fillna("").astype(str).str.strip() if type_col in frame.columns else pd.Series("", index=frame.index, dtype=object)
        if alt_type_col and alt_type_col in frame.columns:
            alt = frame[alt_type_col].fillna("").astype(str).str.strip()
            type_series = type_series.where(type_series.ne(""), alt)

        z_series = pd.to_numeric(frame[z_col], errors="coerce") if z_col in frame.columns else pd.Series(np.nan, index=frame.index)
        sep_series = (
            pd.to_numeric(frame[str(sep_col)], errors="coerce")
            if sep_col and str(sep_col) in frame.columns
            else pd.Series(np.nan, index=frame.index)
        )

        active = _has_value(name_series.astype(str)) | z_series.notna() | _has_value(type_series.astype(str))
        if not active.any():
            continue

        for idx in frame.index[active]:
            row = frame.loc[idx]
            name = str(row.get(name_col, "") or "").strip()
            spectral_type = str(type_col and row.get(type_col, "") or "").strip()
            if alt_type_col and not spectral_type:
                spectral_type = str(row.get(alt_type_col, "") or "").strip()
            z_val = pd.to_numeric(row.get(z_col), errors="coerce") if z_col in frame.columns else np.nan
            sep_val = pd.to_numeric(row.get(sep_col), errors="coerce") if sep_col and sep_col in frame.columns else np.nan
            link = link_builder(row) if callable(link_builder) else None
            if not name and not spectral_type and pd.isna(z_val):
                continue
            rows.append(
                {
                    "candidate_id": str(row["candidate_id"]),
                    "survey": survey,
                    "catalog": f"provenance:{survey}",
                    "sep_arcsec": sep_val,
                    "link": link,
                    "spectrum_redshift": z_val,
                    "spectrum_spectral_type": spectral_type,
                    "provenance_name": name,
                }
            )

    if not rows:
        return spectra_long

    provenance = pd.DataFrame(rows)
    if spectra_long.empty:
        return provenance
    return pd.concat([spectra_long, provenance], ignore_index=True)
