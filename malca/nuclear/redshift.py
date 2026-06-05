from __future__ import annotations

import re

import numpy as np
import pandas as pd


REDSHIFT_ALIASES: tuple[tuple[str, str], ...] = (
    ("redshift", "input"),
    ("z", "input"),
    ("spec_z", "spectrum"),
    ("spectrum_redshift", "spectrum"),
    ("sdss_z", "SDSS"),
    ("desi_z", "DESI"),
    ("sixdf_z", "6dF"),
    ("6df_z", "6dF"),
    ("ned_redshift", "NED"),
    ("simbad_redshift", "SIMBAD"),
    ("tns_redshift", "TNS"),
)

SPECTRAL_TYPE_ALIASES: tuple[tuple[str, str], ...] = (
    ("spectral_type", "input"),
    ("spectrum_spectral_type", "spectrum"),
    ("host_spectral_type", "input"),
    ("host_spectral_class", "input"),
    ("sdss_class", "SDSS"),
    ("sdss_subclass", "SDSS"),
    ("desi_spectype", "DESI"),
    ("desi_class", "DESI"),
    ("sixdf_class", "6dF"),
    ("6df_class", "6dF"),
    ("ned_type", "NED"),
    ("simbad_otype", "SIMBAD"),
    ("tns_type", "TNS"),
)

AGN_RE = re.compile(r"\b(?:agn|qso|quasar|seyfert|liner|blazar|bllac|broad[-\s]?line|type\s*1)\b", re.IGNORECASE)
BROAD_LINE_RE = re.compile(r"\b(?:broad[-\s]?line|type\s*1|seyfert\s*1|qso|quasar|blr)\b", re.IGNORECASE)
QUIESCENT_RE = re.compile(r"\b(?:quiescent|passive|post[-\s]?starburst|e\+a|k\+a|elliptical|early[-\s]?type)\b", re.IGNORECASE)
STAR_FORMING_RE = re.compile(r"\b(?:star[-\s]?forming|hii|emission[-\s]?line|spiral|late[-\s]?type|sf)\b", re.IGNORECASE)
STAR_RE = re.compile(r"\b(?:star|stellar|variable star|eb|rr|mira|cv|nova)\b", re.IGNORECASE)


def _text(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series("", index=frame.index, dtype=object)
    return frame[col].fillna("").astype(str)


def _first_numeric_with_source(frame: pd.DataFrame, aliases: tuple[tuple[str, str], ...]) -> tuple[pd.Series, pd.Series]:
    values = pd.Series(np.nan, index=frame.index, dtype=float)
    sources = pd.Series("", index=frame.index, dtype=object)
    for col, source in aliases:
        if col not in frame.columns:
            continue
        candidate = pd.to_numeric(frame[col], errors="coerce")
        mask = values.isna() & candidate.notna()
        values.loc[mask] = candidate.loc[mask]
        sources.loc[mask] = source
    return values, sources


def _first_text_with_source(frame: pd.DataFrame, aliases: tuple[tuple[str, str], ...]) -> tuple[pd.Series, pd.Series]:
    values = pd.Series("", index=frame.index, dtype=object)
    sources = pd.Series("", index=frame.index, dtype=object)
    for col, source in aliases:
        if col not in frame.columns:
            continue
        candidate = frame[col].fillna("").astype(str).str.strip()
        mask = values.eq("") & candidate.ne("") & ~candidate.str.lower().isin({"nan", "none", "<na>"})
        values.loc[mask] = candidate.loc[mask]
        sources.loc[mask] = source
    return values, sources


def classify_host_spectrum(text: object) -> str:
    """Map heterogeneous catalog spectral labels into coarse nuclear classes."""
    value = str(text or "").strip()
    if not value:
        return "unknown"
    if BROAD_LINE_RE.search(value):
        return "broad_line_agn"
    if AGN_RE.search(value):
        return "agn"
    if QUIESCENT_RE.search(value):
        return "quiescent_or_poststarburst"
    if STAR_FORMING_RE.search(value):
        return "star_forming"
    if STAR_RE.search(value):
        return "stellar"
    if re.search(r"\bgal(?:axy)?\b", value, re.IGNORECASE):
        return "galaxy"
    return "unknown"


def resolve_redshift_spectral_types(df: pd.DataFrame) -> pd.DataFrame:
    """Add redshift and coarse spectral-context fields from available catalog columns."""
    out = df.copy()

    redshift, redshift_source = _first_numeric_with_source(out, REDSHIFT_ALIASES)
    if "redshift" in out.columns:
        out["redshift_input"] = out["redshift"]
    out["redshift"] = redshift
    out["redshift_source"] = redshift_source

    spectral_type, spectral_source = _first_text_with_source(out, SPECTRAL_TYPE_ALIASES)
    out["spectral_type"] = spectral_type
    out["spectral_type_source"] = spectral_source
    out["host_spectral_class"] = spectral_type.map(classify_host_spectrum)
    out["prior_agn_spectrum_flag"] = spectral_type.str.contains(AGN_RE, na=False) | out["host_spectral_class"].isin({"agn", "broad_line_agn"})
    out["broad_line_flag"] = spectral_type.str.contains(BROAD_LINE_RE, na=False) | out["host_spectral_class"].eq("broad_line_agn")

    if "spectrum_sources" not in out.columns:
        source_values: list[str] = []
        for idx in out.index:
            row_sources: list[str] = []
            for source in (redshift_source.loc[idx], spectral_source.loc[idx]):
                text = str(source or "").strip()
                if text and text not in row_sources:
                    row_sources.append(text)
            source_values.append(",".join(row_sources))
        sources = pd.Series(source_values, index=out.index, dtype=object)
        out["spectrum_sources"] = sources
    if "spectrum_links" not in out.columns:
        out["spectrum_links"] = ""

    return out
