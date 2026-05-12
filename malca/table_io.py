"""Parquet-only helpers for MALCA-owned tabular artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.config import PARQUET_OUTPUT_COMPRESSION


def require_parquet_path(path: str | Path, *, kind: str = "table") -> Path:
    """Return ``path`` as a Path and reject non-Parquet internal artifacts."""
    out = Path(path).expanduser()
    if out.suffix.lower() != ".parquet":
        raise ValueError(f"Internal {kind} must be a .parquet file: {out}")
    return out


def read_parquet_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a MALCA-owned Parquet table."""
    table = pd.read_parquet(require_parquet_path(path), **kwargs)
    return table.to_frame() if isinstance(table, pd.Series) else table


def write_parquet_table(
    df: pd.DataFrame,
    path: str | Path,
    *,
    compression: str = PARQUET_OUTPUT_COMPRESSION,
    **kwargs,
) -> None:
    """Write a MALCA-owned Parquet table."""
    out = require_parquet_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression=compression, **kwargs)
