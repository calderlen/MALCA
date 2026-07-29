"""Parquet-only helpers for MALCA-owned tabular artifacts."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Iterable

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


def _normalize_columns(columns: Iterable[str] | None) -> list[str] | None:
    if columns is None:
        return None
    return [str(col) for col in columns]


def _parquet_schema_names(path: str | Path) -> list[str]:
    out = require_parquet_path(path)
    try:
        import pyarrow.parquet as pq

        return list(pq.read_schema(out).names)
    except Exception:
        return list(pd.read_parquet(out).columns)


def is_layer_first_frame(df: pd.DataFrame) -> bool:
    """Return True when a DataFrame uses canonical three-layer feature columns."""
    if not isinstance(df, pd.DataFrame):
        return False
    from malca.products.feature_layers import FEATURE_LAYER_COLUMNS

    return set(FEATURE_LAYER_COLUMNS).issubset(set(map(str, df.columns)))


def is_layer_first_table(path: str | Path) -> bool:
    """Return True when a parquet table has canonical three-layer feature columns."""
    from malca.products.feature_layers import FEATURE_LAYER_COLUMNS

    names = set(_parquet_schema_names(path))
    return set(FEATURE_LAYER_COLUMNS).issubset(names)


def _feature_read_columns(
    schema_names: list[str],
    requested_columns: list[str] | None,
) -> list[str] | None:
    if requested_columns is None:
        return None

    from malca.products.feature_layers import FEATURE_LAYER_COLUMNS, feature_layer_for_column, is_layer_path, split_layer_path

    schema_set = set(schema_names)
    read_cols: list[str] = []
    for col in requested_columns:
        if col in schema_set and col not in read_cols:
            read_cols.append(col)
        elif is_layer_path(col):
            layer, _key = split_layer_path(col)
            if layer in schema_set and layer not in read_cols:
                read_cols.append(layer)
        elif feature_layer_for_column(col) is not None:
            path = f"{feature_layer_for_column(col)}.{col}"
            raise ValueError(
                f"Flat feature column '{col}' is not supported at product boundaries; "
                f"use canonical layer path '{path}'"
            )
        else:
            # Let pandas raise the same missing-column error callers expect.
            read_cols.append(col)
    return read_cols


def read_feature_table(
    path: str | Path,
    *,
    columns: Iterable[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Read a canonical layer-first candidate/product table.

    Raw/cache parquet readers should keep using :func:`read_parquet_table`.
    """
    out = require_parquet_path(path)
    requested_columns = _normalize_columns(columns)
    if not is_layer_first_table(out):
        raise ValueError(
            f"Feature table is not layer-first: {out}. Run 'malca migrate' before using runtime commands."
        )

    schema_names = _parquet_schema_names(out)
    read_kwargs = dict(kwargs)
    read_cols = _feature_read_columns(schema_names, requested_columns)
    if read_cols is not None:
        read_kwargs["columns"] = read_cols
    table = read_parquet_table(out, **read_kwargs)
    if requested_columns is None:
        return table

    from malca.products.feature_layers import is_layer_path, feature_value_series

    projected = pd.DataFrame(index=table.index)
    for col in requested_columns:
        if col in table.columns:
            projected[col] = table[col]
        elif is_layer_path(col):
            projected[col] = feature_value_series(table, col)
    remaining_missing = [col for col in requested_columns if col not in projected.columns]
    if remaining_missing:
        raise KeyError(
            "Requested columns not found in feature table: " + ", ".join(remaining_missing)
        )
    return projected[requested_columns].copy()


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


def _passing_mask(series: pd.Series) -> pd.Series:
    from malca.products.candidates import passing_candidates_mask

    return passing_candidates_mask(
        pd.DataFrame({"failed_any": series}),
        failed_col="failed_any",
        require_failed_col=True,
    )


def read_passing_feature_table(
    path: str | Path,
    *,
    columns: Iterable[str] | None = None,
    failed_col: str = "derived_stats.failed_any",
    **kwargs,
) -> pd.DataFrame:
    """Read passing rows from a canonical layer-first candidate/product table."""
    out = require_parquet_path(path)
    requested_columns = _normalize_columns(columns)
    if not is_layer_first_table(out):
        raise ValueError(
            f"Feature table is not layer-first: {out}. Run 'malca migrate' before using runtime commands."
        )

    read_columns = requested_columns
    if read_columns is not None and failed_col not in read_columns:
        read_columns = [*read_columns, failed_col]
    table = read_feature_table(out, columns=read_columns, **kwargs)
    if failed_col not in table.columns:
        from malca.products.feature_layers import feature_value_series, is_layer_path

        if is_layer_path(failed_col):
            table = table.copy()
            table[failed_col] = feature_value_series(table, failed_col, default=pd.NA)
        else:
            raise ValueError(
                f"Cannot select passing rows from {out}: required feature {failed_col!r} is missing"
            )

    table = table.loc[_passing_mask(table[failed_col])].copy()
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


def _write_layer_first_feature_table_chunked(
    df: pd.DataFrame,
    path: str | Path,
    *,
    compression: str,
    layer_chunk_rows: int,
    **kwargs,
) -> None:
    """Convert a flat feature table to layer-first form and stream it to Parquet."""
    from malca.products.feature_layers import to_layer_first_frame

    import pyarrow as pa
    import pyarrow.parquet as pq

    out = require_parquet_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        writer: pq.ParquetWriter | None = None
        schema = None
        chunk_size = max(1, int(layer_chunk_rows))
        for start in range(0, len(df), chunk_size):
            layered_chunk = to_layer_first_frame(df.iloc[start : start + chunk_size])
            if writer is None:
                table = pa.Table.from_pandas(layered_chunk, preserve_index=False)
                schema = table.schema
                writer = pq.ParquetWriter(tmp_path, schema, compression=compression)
                writer.write_table(table)
            else:
                table = pa.Table.from_pandas(
                    layered_chunk,
                    schema=schema,
                    preserve_index=False,
                )
                writer.write_table(table)
        if writer is None:
            layered = to_layer_first_frame(df)
            layered.to_parquet(tmp_path, index=False, compression=compression, **kwargs)
        else:
            writer.close()
        tmp_path.replace(out)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def write_feature_table(
    df: pd.DataFrame,
    path: str | Path,
    *,
    compression: str = PARQUET_OUTPUT_COMPRESSION,
    chunk_rows: int | None = PARQUET_WRITE_CHUNK_ROWS,
    layer_chunk_rows: int | None = None,
    **kwargs,
) -> None:
    """Write a candidate/product table as canonical layer-first parquet."""
    from malca.products.feature_layers import is_layer_first_frame, to_layer_first_frame

    if is_layer_first_frame(df):
        write_parquet_table(
            df,
            path,
            compression=compression,
            chunk_rows=chunk_rows,
            **kwargs,
        )
        return

    encode_chunk_rows = (
        int(layer_chunk_rows)
        if layer_chunk_rows is not None
        else int(chunk_rows)
        if chunk_rows is not None
        else None
    )
    if (
        encode_chunk_rows is None
        or encode_chunk_rows <= 0
        or len(df) <= encode_chunk_rows
    ):
        layered = to_layer_first_frame(df)
        write_parquet_table(
            layered,
            path,
            compression=compression,
            chunk_rows=chunk_rows,
            **kwargs,
        )
        return

    _write_layer_first_feature_table_chunked(
        df,
        path,
        compression=compression,
        layer_chunk_rows=encode_chunk_rows,
        **kwargs,
    )
