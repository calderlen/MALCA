from __future__ import annotations

from typing import Any

import pandas as pd


REVIEW_METADATA_FIELDS: list[tuple[str, str]] = [
    ("ASAS-SN ID", "asas_sn_id"),
    ("Path", "path"),
    ("VSX class", "vsx_class"),
    ("VSX sep (arcsec)", "vsx_sep_arcsec"),
    ("Trigger", "trigger_type"),
    ("periodic_flag", "periodic_flag"),
    ("catalog_match", "catalog_match"),
    ("high_ruwe_flag", "high_ruwe_flag"),
    ("periodicity_score", "periodicity_score"),
    ("lsp_power", "lsp_power"),
    ("lsp_period", "lsp_period"),
    ("lsp_bootstrap_sig", "lsp_bootstrap_sig"),
    ("dip_best_log_bf", "dip_best_log_bf"),
    ("jump_best_log_bf", "jump_best_log_bf"),
    ("dip_best_morph", "dip_best_morph"),
    ("jump_best_morph", "jump_best_morph"),
    ("ruwe", "ruwe"),
    ("teff_gspphot", "teff_gspphot"),
    ("logg_gspphot", "logg_gspphot"),
    ("mh_gspphot", "mh_gspphot"),
    ("distance_gspphot", "distance_gspphot"),
    ("A_v_3d", "A_v_3d"),
    ("ebv_3d", "ebv_3d"),
    ("yso_class", "yso_class"),
    ("population", "population"),
    ("age50", "age50"),
    ("mass50", "mass50"),
    ("banyan_field_prob", "banyan_field_prob"),
    ("banyan_best_assoc", "banyan_best_assoc"),
]


def normalize_vsx_record(record: dict[str, Any]) -> dict[str, Any]:
    return dict(record)


def normalize_vsx_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "vsx_class" not in out.columns:
        out["vsx_class"] = pd.NA

    if "vsx_sep_arcsec" not in out.columns:
        out["vsx_sep_arcsec"] = pd.NA
    return out


def extract_review_metadata(payload: dict[str, Any]) -> list[tuple[str, Any]]:
    p = normalize_vsx_record(payload)
    rows: list[tuple[str, Any]] = []
    for label, key in REVIEW_METADATA_FIELDS:
        if key not in p:
            continue
        val = p.get(key)
        if val is None:
            continue
        if isinstance(val, float) and pd.isna(val):
            continue
        if isinstance(val, str) and not val.strip():
            continue
        rows.append((label, val))
    return rows
