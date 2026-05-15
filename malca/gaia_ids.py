"""Helpers for handling Gaia source IDs without losing integer precision."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
import numbers
from typing import Iterable

import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype


MAX_EXACT_FLOAT_INT = 2**53 - 1


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
