from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


APOGEE_IDENTITY_COLUMNS: tuple[str, ...] = (
    "APOGEE_ID",
    "TELESCOPE",
    "FIELD",
)

APOGEE_STELLAR_PARAMETER_COLUMNS: tuple[str, ...] = (
    "TEFF",
    "LOGG",
    "M_H",
    "FE_H",
    "VHELIO_AVG",
    "VERR",
    "VSCATTER",
    "SNR",
    "NVISITS",
    "STARFLAG",
    "ASPCAPFLAG",
    "VSINI",
)

APOGEE_ELEMENT_ABUNDANCE_COLUMNS: tuple[str, ...] = (
    "C_FE",
    "CI_FE",
    "N_FE",
    "O_FE",
    "NA_FE",
    "MG_FE",
    "AL_FE",
    "SI_FE",
    "P_FE",
    "S_FE",
    "K_FE",
    "CA_FE",
    "TI_FE",
    "TIII_FE",
    "V_FE",
    "CR_FE",
    "MN_FE",
    "CO_FE",
    "NI_FE",
    "CU_FE",
    "GE_FE",
    "RB_FE",
    "CE_FE",
    "ND_FE",
    "YB_FE",
)

APOGEE_FETCH_COLUMNS: tuple[str, ...] = (
    *APOGEE_IDENTITY_COLUMNS,
    *APOGEE_STELLAR_PARAMETER_COLUMNS,
    *APOGEE_ELEMENT_ABUNDANCE_COLUMNS,
)

APOGEE_SUMMARY_COLUMNS: tuple[str, ...] = (
    *APOGEE_STELLAR_PARAMETER_COLUMNS,
    *APOGEE_ELEMENT_ABUNDANCE_COLUMNS,
)

APOGEE_METADATA_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "APOGEE_ID": ("APOGEE_ID", "apogee_id", "ID", "id"),
    "TELESCOPE": ("TELESCOPE", "telescope", "Tel"),
    "FIELD": ("FIELD", "field", "Field"),
    "TEFF": ("TEFF", "teff", "Teff", "TEFF_SPEC", "RV_TEFF"),
    "LOGG": ("LOGG", "logg", "Logg", "LOGG_SPEC", "RV_LOGG"),
    "M_H": ("M_H", "m_h", "[M/H]", "M_H_SPEC"),
    "FE_H": ("FE_H", "fe_h", "[Fe/H]", "FE_H_SPEC"),
    "VHELIO_AVG": ("VHELIO_AVG", "vhelio_avg", "VHELIO", "HRV", "RV"),
    "VERR": ("VERR", "verr", "e_HRV", "VERR_MED", "RV_ERR"),
    "VSCATTER": ("VSCATTER", "vscatter", "s_HRV", "RV_SCATTER"),
    "SNR": ("SNR", "snr", "SNREV"),
    "NVISITS": ("NVISITS", "nvisits", "Nvisits", "Nvis", "Nv"),
    "STARFLAG": ("STARFLAG", "starflag", "SFlag", "STARFLAGS"),
    "ASPCAPFLAG": ("ASPCAPFLAG", "aspcapflag", "AFlag", "ASPCAPFLAGS"),
    "VSINI": ("VSINI", "vsini", "Vsini", "V_SINI"),
}

_APOGEE_BRACKET_ABUNDANCE_ALIASES: dict[str, str] = {
    "C_FE": "[C/Fe]",
    "CI_FE": "[CI/Fe]",
    "N_FE": "[N/Fe]",
    "O_FE": "[O/Fe]",
    "NA_FE": "[Na/Fe]",
    "MG_FE": "[Mg/Fe]",
    "AL_FE": "[Al/Fe]",
    "SI_FE": "[Si/Fe]",
    "P_FE": "[P/Fe]",
    "S_FE": "[S/Fe]",
    "K_FE": "[K/Fe]",
    "CA_FE": "[Ca/Fe]",
    "TI_FE": "[Ti/Fe]",
    "TIII_FE": "[TiII/Fe]",
    "V_FE": "[V/Fe]",
    "CR_FE": "[Cr/Fe]",
    "MN_FE": "[Mn/Fe]",
    "CO_FE": "[Co/Fe]",
    "NI_FE": "[Ni/Fe]",
    "CU_FE": "[Cu/Fe]",
    "GE_FE": "[Ge/Fe]",
    "RB_FE": "[Rb/Fe]",
    "CE_FE": "[Ce/Fe]",
    "ND_FE": "[Nd/Fe]",
    "YB_FE": "[Yb/Fe]",
}

for _column in APOGEE_ELEMENT_ABUNDANCE_COLUMNS:
    APOGEE_METADATA_COLUMN_ALIASES.setdefault(
        _column,
        (_column, _column.lower(), _APOGEE_BRACKET_ABUNDANCE_ALIASES.get(_column, _column)),
    )


def is_apogee_survey(survey: Any) -> bool:
    return "apogee" in str(survey or "").lower()


def apogee_summary_column(column: str) -> str:
    return f"apogee_{column.lower()}"


def apogee_summary_columns() -> tuple[str, ...]:
    return tuple(apogee_summary_column(column) for column in APOGEE_SUMMARY_COLUMNS)


def _value_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        is_missing = pd.isna(value)
    except Exception:
        is_missing = False
    if isinstance(is_missing, bool) and is_missing:
        return False
    try:
        if float(value) <= -9999.0:
            return False
    except (TypeError, ValueError):
        pass
    if str(value).strip().lower() in {"", "nan", "none", "<na>"}:
        return False
    return True


def _mapping_value(mapping: Mapping[str, Any] | pd.Series, key: str) -> Any:
    if isinstance(mapping, pd.Series):
        return mapping.get(key)
    return mapping.get(key)


def apogee_metadata_from_mapping(*mappings: Mapping[str, Any] | pd.Series | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    valid_mappings = [mapping for mapping in mappings if mapping is not None]
    for column in APOGEE_FETCH_COLUMNS:
        for mapping in valid_mappings:
            for alias in APOGEE_METADATA_COLUMN_ALIASES.get(column, (column,)):
                value = _mapping_value(mapping, alias)
                if _value_present(value):
                    metadata[column] = value
                    break
            if column in metadata:
                break
    return metadata


def normalize_apogee_metadata_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "survey" not in frame.columns:
        return frame

    apogee_mask = frame["survey"].astype(str).str.contains("apogee", case=False, na=False)
    if not apogee_mask.any():
        return frame

    out = frame.copy()
    for column in APOGEE_FETCH_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
        missing = out[column].isna()
        for alias in APOGEE_METADATA_COLUMN_ALIASES.get(column, (column,)):
            if alias not in out.columns or alias == column:
                continue
            update_mask = apogee_mask & missing & out[alias].notna()
            if update_mask.any():
                out.loc[update_mask, column] = out.loc[update_mask, alias]
                missing = out[column].isna()
    return out
