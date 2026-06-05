from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping
import re

import numpy as np
import pandas as pd


NAME_ALIASES = ("name", "object", "object_name", "source_name", "target", "target_name", "iau_name")
RA_ALIASES = ("ra", "ra_deg", "RA", "RAJ2000", "RA_ICRS")
DEC_ALIASES = ("dec", "dec_deg", "DEC", "DEJ2000", "DE_ICRS")
REDSHIFT_ALIASES = ("redshift", "z", "spec_z")
TYPE_ALIASES = ("clagn_type", "type", "class", "classification", "spectral_change")


def _first_col(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    lower = {str(col).lower(): col for col in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        found = lower.get(alias.lower())
        if found is not None:
            return found
    return None


def _read_catalog(path: str | Path) -> pd.DataFrame:
    file_path = Path(path).expanduser()
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(file_path, sep="\t")
    return pd.read_csv(file_path)


def _clean_token(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def _normalize_catalog_frame(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    name_col = _first_col(df, NAME_ALIASES)
    ra_col = _first_col(df, RA_ALIASES)
    dec_col = _first_col(df, DEC_ALIASES)
    z_col = _first_col(df, REDSHIFT_ALIASES)
    type_col = _first_col(df, TYPE_ALIASES)

    out = pd.DataFrame(index=df.index)
    out["known_clagn_name"] = df[name_col].fillna("").astype(str).str.strip() if name_col else ""
    out["known_clagn_source"] = str(source)
    out["known_clagn_ra_deg"] = pd.to_numeric(df[ra_col], errors="coerce") if ra_col else np.nan
    out["known_clagn_dec_deg"] = pd.to_numeric(df[dec_col], errors="coerce") if dec_col else np.nan
    out["known_clagn_redshift"] = pd.to_numeric(df[z_col], errors="coerce") if z_col else np.nan
    out["known_clagn_type"] = df[type_col].fillna("").astype(str).str.strip() if type_col else ""
    out["known_clagn_label"] = "known_clagn"
    out["known_clagn_name_key"] = out["known_clagn_name"].map(_clean_token)
    return out


def load_known_clagn_catalogs(
    catalog_paths: Mapping[str, str | Path] | Iterable[str | Path] | None,
) -> pd.DataFrame:
    """Load DESI/6dF/literature CLAGN catalog exports into a common schema."""
    if catalog_paths is None:
        return pd.DataFrame(
            columns=[
                "known_clagn_name",
                "known_clagn_source",
                "known_clagn_ra_deg",
                "known_clagn_dec_deg",
                "known_clagn_redshift",
                "known_clagn_type",
                "known_clagn_label",
                "known_clagn_name_key",
            ]
        )

    if isinstance(catalog_paths, Mapping):
        items = list(catalog_paths.items())
    else:
        items = [(Path(path).stem, path) for path in catalog_paths]

    frames: list[pd.DataFrame] = []
    for source, path in items:
        file_path = Path(path).expanduser()
        if not file_path.exists():
            continue
        raw = _read_catalog(file_path)
        frames.append(_normalize_catalog_frame(raw, source=str(source)))

    if not frames:
        return load_known_clagn_catalogs(None)
    return pd.concat(frames, ignore_index=True)


def _angular_sep_arcsec(
    ra1: pd.Series,
    dec1: pd.Series,
    ra2: pd.Series,
    dec2: pd.Series,
) -> np.ndarray:
    r1 = np.deg2rad(pd.to_numeric(ra1, errors="coerce").to_numpy(dtype=float))
    d1 = np.deg2rad(pd.to_numeric(dec1, errors="coerce").to_numpy(dtype=float))
    r2 = np.deg2rad(pd.to_numeric(ra2, errors="coerce").to_numpy(dtype=float))
    d2 = np.deg2rad(pd.to_numeric(dec2, errors="coerce").to_numpy(dtype=float))
    sin_ddec = np.sin((d2 - d1) / 2.0)
    sin_dra = np.sin((r2 - r1) / 2.0)
    a = sin_ddec**2 + np.cos(d1) * np.cos(d2) * sin_dra**2
    return np.rad2deg(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))) * 3600.0


def match_known_clagn_catalogs(
    targets: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    radius_arcsec: float = 3.0,
) -> pd.DataFrame:
    """Attach known-CLAGN matches by name and coordinates."""
    out = targets.copy()
    for col, default in (
        ("known_clagn_match", False),
        ("known_clagn_source", ""),
        ("known_clagn_type", ""),
        ("known_clagn_name", ""),
        ("known_clagn_sep_arcsec", np.nan),
        ("known_clagn_training_label", ""),
    ):
        if col not in out.columns:
            out[col] = default

    if catalog is None or catalog.empty or out.empty:
        return out

    catalog = catalog.copy()
    name_lookup = {
        str(row["known_clagn_name_key"]): row
        for _idx, row in catalog.iterrows()
        if str(row.get("known_clagn_name_key", "")).strip()
    }
    target_name_keys = pd.Series("", index=out.index, dtype=object)
    for col in ("candidate_id", "name", "asas_sn_id", "clagn_id", "aliases"):
        if col not in out.columns:
            continue
        keys = out[col].fillna("").astype(str).map(_clean_token)
        target_name_keys = target_name_keys.mask(target_name_keys.eq("") & keys.ne(""), keys)

    for idx, key in target_name_keys.items():
        if key not in name_lookup:
            continue
        row = name_lookup[key]
        out.loc[idx, "known_clagn_match"] = True
        out.loc[idx, "known_clagn_source"] = row.get("known_clagn_source", "")
        out.loc[idx, "known_clagn_type"] = row.get("known_clagn_type", "")
        out.loc[idx, "known_clagn_name"] = row.get("known_clagn_name", "")
        out.loc[idx, "known_clagn_sep_arcsec"] = 0.0
        out.loc[idx, "known_clagn_training_label"] = "known_clagn"

    coord_targets = out[["ra_deg", "dec_deg"]].copy() if {"ra_deg", "dec_deg"}.issubset(out.columns) else pd.DataFrame()
    coord_catalog = catalog.dropna(subset=["known_clagn_ra_deg", "known_clagn_dec_deg"]).copy()
    if coord_targets.empty or coord_catalog.empty:
        return out

    for idx, target in coord_targets.dropna(subset=["ra_deg", "dec_deg"]).iterrows():
        sep = _angular_sep_arcsec(
            pd.Series([target["ra_deg"]] * len(coord_catalog)),
            pd.Series([target["dec_deg"]] * len(coord_catalog)),
            coord_catalog["known_clagn_ra_deg"],
            coord_catalog["known_clagn_dec_deg"],
        )
        if len(sep) == 0 or not np.isfinite(sep).any():
            continue
        best_pos = int(np.nanargmin(sep))
        best_sep = float(sep[best_pos])
        if best_sep > float(radius_arcsec):
            continue
        row = coord_catalog.iloc[best_pos]
        existing_sep = pd.to_numeric(pd.Series([out.loc[idx, "known_clagn_sep_arcsec"]]), errors="coerce").iloc[0]
        if bool(out.loc[idx, "known_clagn_match"]) and np.isfinite(existing_sep) and existing_sep <= best_sep:
            continue
        out.loc[idx, "known_clagn_match"] = True
        out.loc[idx, "known_clagn_source"] = row.get("known_clagn_source", "")
        out.loc[idx, "known_clagn_type"] = row.get("known_clagn_type", "")
        out.loc[idx, "known_clagn_name"] = row.get("known_clagn_name", "")
        out.loc[idx, "known_clagn_sep_arcsec"] = best_sep
        out.loc[idx, "known_clagn_training_label"] = "known_clagn"

    return out
