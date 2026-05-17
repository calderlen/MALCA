"""Parquet-only helpers for MALCA-owned tabular artifacts."""

from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd

from malca.config import PARQUET_OUTPUT_COMPRESSION

PARQUET_WRITE_CHUNK_ROWS = 250_000


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


def read_passing_parquet_table(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    failed_col: str = "failed_any",
    **kwargs,
) -> pd.DataFrame:
    """Read rows that pass filtering without materializing full audit tables."""
    out = require_parquet_path(path)
    read_kwargs = dict(kwargs)
    if columns is not None:
        read_kwargs["columns"] = columns

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pq.read_schema(out)
        if failed_col not in schema.names:
            return read_parquet_table(out, **read_kwargs)

        field_type = schema.field(failed_col).type
        if pa.types.is_boolean(field_type):
            filters = [(failed_col, "==", False)]
        elif pa.types.is_integer(field_type):
            filters = [(failed_col, "==", 0)]
        else:
            filters = None

        if filters is not None:
            table = pd.read_parquet(out, filters=filters, **read_kwargs)
            return table.to_frame() if isinstance(table, pd.Series) else table
    except Exception:
        # Fall back to a normal read below; callers still get correct rows.
        pass

    fallback_kwargs = dict(read_kwargs)
    requested_columns = fallback_kwargs.get("columns")
    if requested_columns is not None and failed_col not in requested_columns:
        fallback_kwargs["columns"] = list(requested_columns) + [failed_col]
    table = read_parquet_table(out, **fallback_kwargs)
    if failed_col not in table.columns:
        return table

    failed = table[failed_col]
    if pd.api.types.is_bool_dtype(failed):
        mask = ~failed.fillna(False).astype(bool)
    elif pd.api.types.is_numeric_dtype(failed):
        mask = failed.fillna(0).astype(float) == 0.0
    else:
        lowered = failed.astype("string").str.strip().str.lower()
        mask = ~lowered.isin({"1", "true", "t", "yes", "y"}).fillna(False)

    table = table.loc[mask].copy()
    if requested_columns is not None and failed_col not in requested_columns:
        table = table.drop(columns=[failed_col])
    return table


def write_parquet_table(
    df: pd.DataFrame,
    path: str | Path,
    *,
    compression: str = PARQUET_OUTPUT_COMPRESSION,
    chunk_rows: int | None = PARQUET_WRITE_CHUNK_ROWS,
    **kwargs,
) -> None:
    """Write a MALCA-owned Parquet table atomically.

    Large frames are streamed to Parquet in row chunks to avoid the extra
    whole-table Arrow/pandas allocation that can exceed memory on audit tables.
    """
    out = require_parquet_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        effective_chunk_rows = int(chunk_rows) if chunk_rows is not None else None
        if (
            effective_chunk_rows is None
            or effective_chunk_rows <= 0
            or len(df) <= effective_chunk_rows
        ):
            df.to_parquet(tmp_path, index=False, compression=compression, **kwargs)
        else:
            if kwargs:
                df.to_parquet(tmp_path, index=False, compression=compression, **kwargs)
            else:
                import pyarrow as pa
                import pyarrow.parquet as pq

                chunk_size = effective_chunk_rows
                schema_source = df.iloc[:chunk_size]
                extra_schema_labels: list[object] = []
                for col in df.columns:
                    if not pd.api.types.is_object_dtype(df[col]):
                        continue
                    if not bool(schema_source[col].isna().all()):
                        continue
                    first_valid = df[col].first_valid_index()
                    if first_valid is not None and first_valid not in schema_source.index:
                        extra_schema_labels.append(first_valid)
                if extra_schema_labels:
                    extra_schema_labels = list(dict.fromkeys(extra_schema_labels))
                    schema_source = pd.concat(
                        [schema_source, df.loc[extra_schema_labels]],
                        copy=False,
                    )
                schema = pa.Table.from_pandas(schema_source, preserve_index=False).schema
                writer: pq.ParquetWriter | None = None
                try:
                    for start in range(0, len(df), chunk_size):
                        chunk = df.iloc[start : start + chunk_size]
                        table = pa.Table.from_pandas(
                            chunk,
                            schema=schema,
                            preserve_index=False,
                        )
                        if writer is None:
                            writer = pq.ParquetWriter(
                                tmp_path,
                                schema,
                                compression=compression,
                            )
                        writer.write_table(table)
                finally:
                    if writer is not None:
                        writer.close()
        tmp_path.replace(out)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
