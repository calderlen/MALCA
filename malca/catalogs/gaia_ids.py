"""Helpers for handling Gaia source IDs without losing integer precision."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
import numbers
from pathlib import Path
from typing import Iterable, Sequence
import warnings

import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype

from malca.config import (
    GAIA_AIP_TAP_URL,
    GAIA_ID_MAPPING_CACHE,
    GAIA_LOCAL_CATALOG,
    LEGACY_GAIA_CACHE_FILE,
    LEGACY_GAIA_LOCAL_CATALOG,
    PARQUET_CACHE_COMPRESSION,
)


MAX_EXACT_FLOAT_INT = 2**53 - 1
GAIA_ID_MAPPING_COLUMNS = (
    "input_gaia_id",
    "source_id",
    "gaia_id",
    "gaia_dr2_id",
    "gaia_id_release",
    "gaia_id_mapping_status",
    "dr2_dr3_angular_distance_mas",
    "dr2_dr3_magnitude_difference",
)
GAIA_ID_PROVENANCE_COLUMNS = (
    "gaia_dr2_id",
    "gaia_id_release",
    "gaia_id_mapping_status",
    "dr2_dr3_angular_distance_mas",
    "dr2_dr3_magnitude_difference",
)
_MISSING_TEXT = {"", "nan", "none", "<na>", "null"}


def parse_gaia_source_id(value: object) -> str | None:
    """Parse a Gaia source ID-like value to an exact integer string.

    Large Gaia IDs cannot be represented exactly as floating point numbers. If
    a large ID has already become a float, reject it instead of querying a
    rounded source_id.
    """
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        if value is None:
            return None

    if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        fval = float(value)
        if not math.isfinite(fval) or not fval.is_integer():
            return None
        if abs(fval) > MAX_EXACT_FLOAT_INT:
            return None

    text = str(value).strip()
    if not text:
        return None

    try:
        source_id = Decimal(text)
    except (InvalidOperation, ValueError):
        return None

    if source_id != source_id.to_integral_value():
        return None

    return format(source_id.to_integral_value(), "f")


def normalize_gaia_source_ids(values: Iterable[object]) -> list[str]:
    """Return unique Gaia source IDs as exact digit strings, preserving order."""
    normalized: list[str] = []
    seen: set[str] = set()

    for value in values:
        source_id = parse_gaia_source_id(value)
        if source_id is None or source_id in seen:
            continue
        seen.add(source_id)
        normalized.append(source_id)

    return normalized


def normalize_gaia_source_id_series(series: pd.Series) -> pd.Series:
    """Normalize a pandas Series of Gaia source IDs without float upcasts."""
    if is_integer_dtype(series.dtype):
        return series.astype("Int64").astype("string")

    if is_float_dtype(series.dtype):
        values = pd.to_numeric(series, errors="coerce")
        valid = (
            values.notna()
            & (values % 1 == 0)
            & (values.abs() <= MAX_EXACT_FLOAT_INT)
        )
        out = pd.Series(pd.NA, index=series.index, dtype="string")
        if bool(valid.any()):
            out.loc[valid] = values.loc[valid].astype("Int64").astype("string")
        return out

    return series.map(parse_gaia_source_id).astype("string")


def _empty_gaia_id_mapping_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(GAIA_ID_MAPPING_COLUMNS))


def _coerce_gaia_id_mapping_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize persisted or TAP-returned mapping tables to the public schema."""
    if df is None or df.empty:
        return _empty_gaia_id_mapping_frame()

    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    rename_map = {
        "dr2_source_id": "input_gaia_id",
        "dr3_source_id": "source_id",
        "angular_distance": "dr2_dr3_angular_distance_mas",
        "magnitude_difference": "dr2_dr3_magnitude_difference",
    }
    out = out.rename(columns={old: new for old, new in rename_map.items() if old in out.columns})

    if "input_gaia_id" in out.columns:
        out["input_gaia_id"] = out["input_gaia_id"].map(parse_gaia_source_id)
    if "source_id" in out.columns:
        out["source_id"] = out["source_id"].map(parse_gaia_source_id)
    if "gaia_id" not in out.columns and "source_id" in out.columns:
        out["gaia_id"] = out["source_id"]
    if "gaia_id" in out.columns:
        out["gaia_id"] = out["gaia_id"].map(parse_gaia_source_id)
    if "gaia_dr2_id" in out.columns:
        out["gaia_dr2_id"] = out["gaia_dr2_id"].map(parse_gaia_source_id)

    for column in ("dr2_dr3_angular_distance_mas", "dr2_dr3_magnitude_difference"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    for column in GAIA_ID_MAPPING_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA

    out = out.loc[out["input_gaia_id"].notna(), list(GAIA_ID_MAPPING_COLUMNS)].copy()
    return out.drop_duplicates(subset=["input_gaia_id"], keep="last")


def _read_gaia_id_mapping_cache(mapping_cache_path: Path | str | None) -> pd.DataFrame:
    if mapping_cache_path is None:
        return _empty_gaia_id_mapping_frame()
    path = Path(mapping_cache_path)
    if not path.exists():
        return _empty_gaia_id_mapping_frame()
    try:
        return _coerce_gaia_id_mapping_frame(pd.read_parquet(path))
    except Exception as exc:
        warnings.warn(f"Ignoring unreadable Gaia ID mapping cache at {path}: {exc}", RuntimeWarning)
        return _empty_gaia_id_mapping_frame()


def _write_gaia_id_mapping_cache(mapping_cache_path: Path | str | None, rows: pd.DataFrame) -> None:
    if mapping_cache_path is None or rows.empty:
        return

    path = Path(mapping_cache_path)
    new_rows = _coerce_gaia_id_mapping_frame(rows)
    if new_rows.empty:
        return

    existing = _read_gaia_id_mapping_cache(path)
    combined = new_rows if existing.empty else pd.concat([existing, new_rows], ignore_index=True)
    combined = _coerce_gaia_id_mapping_frame(combined)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False, compression=PARQUET_CACHE_COMPRESSION)


def _local_gaia_catalog_paths(gaia_cache_path: Path | str | None) -> list[Path]:
    candidates: list[Path] = []
    for value in (gaia_cache_path, GAIA_LOCAL_CATALOG, LEGACY_GAIA_LOCAL_CATALOG, LEGACY_GAIA_CACHE_FILE):
        if value is None:
            continue
        path = Path(value)
        if path not in candidates:
            candidates.append(path)
    return candidates


def _ids_present_in_local_dr3_cache(
    gaia_ids: Sequence[str],
    *,
    gaia_cache_path: Path | str | None,
) -> set[str]:
    requested = set(gaia_ids)
    if not requested:
        return set()

    found: set[str] = set()
    for path in _local_gaia_catalog_paths(gaia_cache_path):
        if not path.exists():
            continue
        try:
            cache = pd.read_parquet(path, columns=["source_id"])
        except Exception:
            continue
        if "source_id" not in cache.columns:
            continue
        source_ids = cache["source_id"].map(parse_gaia_source_id).dropna()
        found.update(set(source_ids) & requested)
        if found == requested:
            break
    return found


def _chunked(values: Sequence[str], chunk_size: int) -> Iterable[list[str]]:
    size = max(1, int(chunk_size))
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def query_dr2_neighbourhood_mappings(
    dr2_ids: Sequence[str | int],
    *,
    tap_url: str = GAIA_AIP_TAP_URL,
    chunk_size: int = 5000,
    tap_service: object | None = None,
) -> pd.DataFrame:
    """Query Gaia DR3's official DR2-neighbourhood table for DR2 source IDs."""
    ids = normalize_gaia_source_ids(dr2_ids)
    if not ids:
        return _empty_gaia_id_mapping_frame()

    from astropy.table import Table
    import pyvo

    service = tap_service if tap_service is not None else pyvo.dal.TAPService(tap_url)
    query = """
    SELECT
        n.dr2_source_id,
        n.dr3_source_id,
        n.angular_distance,
        n.magnitude_difference
    FROM TAP_UPLOAD.upload_table AS u
    JOIN gaiadr3.dr2_neighbourhood AS n
        ON n.dr2_source_id = u.source_id
    """

    frames: list[pd.DataFrame] = []
    for chunk in _chunked(ids, chunk_size):
        upload_table = Table({"source_id": [int(value) for value in chunk]})
        result = service.run_async(query, uploads={"upload_table": upload_table})
        frame = result.to_table().to_pandas()
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return _empty_gaia_id_mapping_frame()

    out = _coerce_gaia_id_mapping_frame(pd.concat(frames, ignore_index=True))
    if out.empty:
        return out

    out["gaia_id"] = out["source_id"]
    same_id = out["source_id"].eq(out["input_gaia_id"])
    out.loc[same_id, "gaia_dr2_id"] = pd.NA
    out.loc[~same_id, "gaia_dr2_id"] = out.loc[~same_id, "input_gaia_id"]
    out.loc[same_id, "gaia_id_release"] = "dr3"
    out.loc[~same_id, "gaia_id_release"] = "dr2_translated"
    out.loc[same_id, "gaia_id_mapping_status"] = "dr3"
    out.loc[~same_id, "gaia_id_mapping_status"] = "dr2_translated"
    out["_distance_sort"] = pd.to_numeric(out["dr2_dr3_angular_distance_mas"], errors="coerce")
    out["_mag_sort"] = pd.to_numeric(out["dr2_dr3_magnitude_difference"], errors="coerce").abs()
    out = out.sort_values(["input_gaia_id", "_distance_sort", "_mag_sort"], na_position="last")
    out = out.drop_duplicates(subset=["input_gaia_id"], keep="first")
    return out.loc[:, list(GAIA_ID_MAPPING_COLUMNS)].copy()


def query_dr3_source_ids(
    source_ids: Sequence[str | int],
    *,
    tap_url: str = GAIA_AIP_TAP_URL,
    chunk_size: int = 5000,
    tap_service: object | None = None,
) -> set[str]:
    """Return IDs that exist in ``gaiadr3.gaia_source``."""
    ids = normalize_gaia_source_ids(source_ids)
    if not ids:
        return set()

    from astropy.table import Table
    import pyvo

    service = tap_service if tap_service is not None else pyvo.dal.TAPService(tap_url)
    query = """
    SELECT g.source_id
    FROM TAP_UPLOAD.upload_table AS u
    JOIN gaiadr3.gaia_source AS g
        ON g.source_id = u.source_id
    """

    found: set[str] = set()
    for chunk in _chunked(ids, chunk_size):
        upload_table = Table({"source_id": [int(value) for value in chunk]})
        result = service.run_async(query, uploads={"upload_table": upload_table})
        frame = result.to_table().to_pandas()
        if not frame.empty and "source_id" in frame.columns:
            found.update(frame["source_id"].map(parse_gaia_source_id).dropna().tolist())
    return found


def canonicalize_gaia_ids(
    gaia_ids: Iterable[object],
    *,
    gaia_cache_path: Path | str | None = None,
    mapping_cache_path: Path | str | None = GAIA_ID_MAPPING_CACHE,
    write_mapping_cache: bool = True,
    tap_url: str = GAIA_AIP_TAP_URL,
    query_tap: bool = True,
    chunk_size: int = 5000,
    warn: bool = True,
) -> pd.DataFrame:
    """Return a mapping table whose ``source_id``/``gaia_id`` values are DR3 IDs.

    The lookup is intentionally conservative: IDs already present in the local
    Gaia DR3 cache are accepted as DR3, remaining IDs are looked up in the
    cached/official DR2-neighbourhood table, and IDs with no mapping are left
    unchanged with a status marker.
    """
    ids = normalize_gaia_source_ids(gaia_ids)
    if not ids:
        return _empty_gaia_id_mapping_frame()

    out = pd.DataFrame(
        {
            "input_gaia_id": ids,
            "source_id": ids,
            "gaia_id": ids,
            "gaia_dr2_id": pd.NA,
            "gaia_id_release": pd.NA,
            "gaia_id_mapping_status": "pending",
            "dr2_dr3_angular_distance_mas": pd.NA,
            "dr2_dr3_magnitude_difference": pd.NA,
        },
        columns=list(GAIA_ID_MAPPING_COLUMNS),
    )

    local_dr3_ids = _ids_present_in_local_dr3_cache(ids, gaia_cache_path=gaia_cache_path)
    if local_dr3_ids:
        local_mask = out["input_gaia_id"].isin(local_dr3_ids)
        out.loc[local_mask, "gaia_id_release"] = "dr3"
        out.loc[local_mask, "gaia_id_mapping_status"] = "dr3"

    pending_mask = out["gaia_id_mapping_status"].eq("pending")
    if pending_mask.any() and query_tap:
        pending_ids = out.loc[pending_mask, "input_gaia_id"].tolist()
        try:
            remote_dr3_ids = query_dr3_source_ids(
                pending_ids,
                tap_url=tap_url,
                chunk_size=chunk_size,
            )
        except Exception as exc:
            remote_dr3_ids = set()
            if warn:
                warnings.warn(f"Gaia DR3 source_id existence lookup failed: {exc}", RuntimeWarning)
        if remote_dr3_ids:
            remote_mask = out["input_gaia_id"].isin(remote_dr3_ids) & out["gaia_id_mapping_status"].eq("pending")
            out.loc[remote_mask, "gaia_id_release"] = "dr3"
            out.loc[remote_mask, "gaia_id_mapping_status"] = "dr3"

    pending_mask = out["gaia_id_mapping_status"].eq("pending")
    cache = _read_gaia_id_mapping_cache(mapping_cache_path)
    if pending_mask.any() and not cache.empty:
        cache_lookup = cache.set_index("input_gaia_id")
        pending_ids = out.loc[pending_mask, "input_gaia_id"]
        cached_mask = pending_ids.isin(cache_lookup.index)
        if cached_mask.any():
            cached_indices = pending_ids.loc[cached_mask].index
            for column in GAIA_ID_MAPPING_COLUMNS:
                if column == "input_gaia_id":
                    continue
                out.loc[cached_indices, column] = pending_ids.loc[cached_indices].map(cache_lookup[column])

    pending_mask = out["gaia_id_mapping_status"].eq("pending")
    tap_failed = False
    if pending_mask.any() and query_tap:
        pending_ids = out.loc[pending_mask, "input_gaia_id"].tolist()
        try:
            tap_mappings = query_dr2_neighbourhood_mappings(
                pending_ids,
                tap_url=tap_url,
                chunk_size=chunk_size,
            )
        except Exception as exc:
            tap_failed = True
            if warn:
                warnings.warn(f"Gaia DR2->DR3 mapping lookup failed: {exc}", RuntimeWarning)
            tap_mappings = _empty_gaia_id_mapping_frame()

        if not tap_mappings.empty:
            tap_lookup = tap_mappings.set_index("input_gaia_id")
            ids_series = out.loc[pending_mask, "input_gaia_id"]
            mapped_mask = ids_series.isin(tap_lookup.index)
            if mapped_mask.any():
                mapped_indices = ids_series.loc[mapped_mask].index
                for column in GAIA_ID_MAPPING_COLUMNS:
                    if column == "input_gaia_id":
                        continue
                    out.loc[mapped_indices, column] = ids_series.loc[mapped_indices].map(tap_lookup[column])

    pending_mask = out["gaia_id_mapping_status"].eq("pending")
    if pending_mask.any():
        status = "lookup_failed" if tap_failed else ("unchecked" if not query_tap else "unmapped")
        out.loc[pending_mask, "gaia_id_mapping_status"] = status
        out.loc[pending_mask, "gaia_id_release"] = pd.NA
        if warn and status == "unmapped":
            sample = ", ".join(out.loc[pending_mask, "input_gaia_id"].head(5).tolist())
            more = "..." if int(pending_mask.sum()) > 5 else ""
            warnings.warn(
                f"No Gaia DR3 DR2-neighbourhood mapping found for {int(pending_mask.sum())} Gaia ID(s): {sample}{more}",
                RuntimeWarning,
            )

    cacheable = out[out["gaia_id_mapping_status"].isin({"dr3", "dr2_translated", "unmapped"})]
    if write_mapping_cache:
        try:
            _write_gaia_id_mapping_cache(mapping_cache_path, cacheable)
        except Exception as exc:
            if warn:
                warnings.warn(f"Could not write Gaia ID mapping cache: {exc}", RuntimeWarning)
    return _coerce_gaia_id_mapping_frame(out)


def _missing_like_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return series.isna() | text.isin(_MISSING_TEXT)


def canonicalize_gaia_ids_in_frame(
    df: pd.DataFrame,
    *,
    gaia_col: str = "gaia_id",
    source_col: str = "source_id",
    gaia_cache_path: Path | str | None = None,
    mapping_cache_path: Path | str | None = GAIA_ID_MAPPING_CACHE,
    write_mapping_cache: bool = True,
    tap_url: str = GAIA_AIP_TAP_URL,
    query_tap: bool = True,
    chunk_size: int = 5000,
    warn: bool = True,
) -> pd.DataFrame:
    """Canonicalize a candidate table's Gaia IDs to DR3 and attach provenance."""
    if df.empty or (gaia_col not in df.columns and source_col not in df.columns):
        return df

    out = df.copy()
    if gaia_col in out.columns:
        raw_ids = out[gaia_col]
    else:
        raw_ids = out[source_col]

    normalized_ids = raw_ids.map(parse_gaia_source_id)
    valid_mask = normalized_ids.notna()
    if not valid_mask.any():
        return out

    mapping = canonicalize_gaia_ids(
        normalized_ids.loc[valid_mask].tolist(),
        gaia_cache_path=gaia_cache_path,
        mapping_cache_path=mapping_cache_path,
        write_mapping_cache=write_mapping_cache,
        tap_url=tap_url,
        query_tap=query_tap,
        chunk_size=chunk_size,
        warn=warn,
    )
    if mapping.empty:
        return out

    lookup = mapping.set_index("input_gaia_id")
    canonical = normalized_ids.map(lookup["source_id"])
    status = normalized_ids.map(lookup["gaia_id_mapping_status"])
    canonical_mask = canonical.notna() & status.isin({"dr3", "dr2_translated", "unmapped"})

    if gaia_col not in out.columns:
        out[gaia_col] = pd.NA
    out.loc[canonical_mask, gaia_col] = canonical.loc[canonical_mask]
    out[gaia_col] = out[gaia_col].astype("string")

    if source_col not in out.columns:
        out[source_col] = pd.NA
    source_fill_mask = canonical.notna() & status.isin({"dr3", "dr2_translated"})
    out.loc[source_fill_mask, source_col] = canonical.loc[source_fill_mask]
    out[source_col] = out[source_col].astype("string")

    for column in GAIA_ID_PROVENANCE_COLUMNS:
        values = normalized_ids.map(lookup[column])
        if column not in out.columns:
            out[column] = pd.NA
        if column in {"gaia_id_mapping_status", "gaia_id_release"}:
            fill_mask = values.notna()
        else:
            fill_mask = values.notna() & ~_missing_like_mask(values)
        if fill_mask.any():
            out.loc[fill_mask, column] = values.loc[fill_mask]

    return out
