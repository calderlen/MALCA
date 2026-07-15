#!/usr/bin/env python
"""Download the full AAVSO VSX catalog and build MALCA's local Parquet copy.

The CDS/VizieR mirror stores VSX as a fixed-width table.  MALCA expects the raw
file at ``input/vsx/vsxcat.090525.csv`` and the full, unfiltered catalog at
``input/vsx/vsx_all.parquet``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from malca.config import PARQUET_OUTPUT_COMPRESSION, VSX_ALL_CATALOG_PATH, VSX_RAW_CATALOG_PATH
from malca.vsx.filter import (
    coerce_vsx_catalog_columns,
    colspecs,
    normalize_vsx_catalog,
    vsx_columns,
)


DEFAULT_VSX_URL = "https://cdsarc.cds.unistra.fr/ftp/B/vsx/vsx.dat"
DEFAULT_CHUNK_ROWS = 250_000
DEFAULT_DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
PARQUET_COMPLETE_MARKER_SUFFIX = ".complete.json"


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{value} B"


def _remote_size(session: requests.Session, url: str, *, timeout: float) -> int | None:
    try:
        response = session.head(url, allow_redirects=True, timeout=timeout)
    except requests.RequestException:
        return None
    if not response.ok:
        return None
    try:
        return int(response.headers.get("Content-Length", ""))
    except ValueError:
        return None


def download_vsx_raw(
    *,
    url: str = DEFAULT_VSX_URL,
    raw_path: Path = VSX_RAW_CATALOG_PATH,
    force: bool = False,
    timeout: float = 120.0,
    chunk_bytes: int = DEFAULT_DOWNLOAD_CHUNK_BYTES,
) -> Path:
    """Download ``vsx.dat`` to ``raw_path``, resuming a partial file when possible."""
    raw_path = Path(raw_path).expanduser()
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if force and raw_path.exists():
        raw_path.unlink()

    with requests.Session() as session:
        total_size = _remote_size(session, url, timeout=timeout)
        existing_size = raw_path.stat().st_size if raw_path.exists() else 0

        if total_size is not None and existing_size == total_size:
            print(f"VSX raw file already complete: {raw_path} ({_format_bytes(existing_size)})")
            return raw_path
        if total_size is not None and existing_size > total_size:
            raise RuntimeError(
                f"Existing raw file is larger than remote file: {raw_path} "
                f"({_format_bytes(existing_size)} > {_format_bytes(total_size)}). "
                "Use --force-download to replace it."
            )

        headers: dict[str, str] = {}
        mode = "wb"
        if existing_size:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"
            print(
                f"Resuming VSX download at {_format_bytes(existing_size)} "
                f"of {_format_bytes(total_size)}"
            )
        else:
            print(f"Downloading VSX catalog to {raw_path} ({_format_bytes(total_size)})")

        response = session.get(
            url,
            headers=headers,
            stream=True,
            timeout=timeout,
            allow_redirects=True,
        )
        if existing_size and response.status_code == 200:
            response.close()
            raise RuntimeError(
                "Server did not honor the resume request. "
                "Use --force-download to restart from byte 0."
            )
        if response.status_code == 416 and total_size is not None and existing_size == total_size:
            response.close()
            print(f"VSX raw file already complete: {raw_path} ({_format_bytes(existing_size)})")
            return raw_path
        response.raise_for_status()

        downloaded = existing_size
        t0 = time.perf_counter()
        last_report = downloaded
        with raw_path.open(mode) as handle:
            for block in response.iter_content(chunk_size=int(chunk_bytes)):
                if not block:
                    continue
                handle.write(block)
                downloaded += len(block)
                if downloaded - last_report >= 64 * 1024 * 1024:
                    elapsed = max(time.perf_counter() - t0, 1e-6)
                    rate = (downloaded - existing_size) / elapsed
                    print(
                        "  downloaded "
                        f"{_format_bytes(downloaded)} / {_format_bytes(total_size)} "
                        f"at {_format_bytes(int(rate))}/s",
                        flush=True,
                    )
                    last_report = downloaded

    print(f"Downloaded VSX raw file: {raw_path} ({_format_bytes(raw_path.stat().st_size)})")
    if total_size is not None and raw_path.stat().st_size != total_size:
        raise RuntimeError(
            f"Downloaded VSX raw file is incomplete: {raw_path} "
            f"({_format_bytes(raw_path.stat().st_size)} / {_format_bytes(total_size)})"
        )
    return raw_path


def verify_vsx_raw_complete(
    *,
    raw_path: Path,
    url: str = DEFAULT_VSX_URL,
    timeout: float = 120.0,
) -> None:
    """Raise if an existing raw VSX file is known to be only partially downloaded."""
    raw_path = Path(raw_path).expanduser()
    if not raw_path.exists():
        raise FileNotFoundError(f"VSX raw file not found: {raw_path}")
    with requests.Session() as session:
        total_size = _remote_size(session, url, timeout=timeout)
    if total_size is None:
        print(
            "Warning: could not verify remote VSX size; converting existing raw file as-is",
            flush=True,
        )
        return

    local_size = raw_path.stat().st_size
    if local_size != total_size:
        raise RuntimeError(
            f"Existing VSX raw file is incomplete: {raw_path} "
            f"({_format_bytes(local_size)} / {_format_bytes(total_size)}). "
            "Run without --skip-download to resume it, or pass --allow-partial-raw "
            "only for a deliberate test conversion."
        )


def _complete_marker_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}{PARQUET_COMPLETE_MARKER_SUFFIX}")


def _write_complete_marker(*, output_path: Path, raw_path: Path, row_count: int) -> None:
    marker_path = _complete_marker_path(output_path)
    marker = {
        "output_path": str(output_path),
        "raw_path": str(raw_path),
        "row_count": int(row_count),
        "raw_size_bytes": int(raw_path.stat().st_size) if raw_path.exists() else None,
        "output_size_bytes": int(output_path.stat().st_size) if output_path.exists() else None,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iter_normalized_vsx_chunks(raw_path: Path, *, chunk_rows: int):
    for chunk in pd.read_fwf(
        raw_path,
        colspecs=colspecs,
        names=vsx_columns,
        dtype=str,
        chunksize=int(chunk_rows),
    ):
        coerced = coerce_vsx_catalog_columns(chunk)
        normalized = normalize_vsx_catalog(coerced)
        if not normalized.empty:
            yield normalized


def convert_vsx_raw_to_parquet(
    raw_path: Path = VSX_RAW_CATALOG_PATH,
    output_path: Path = VSX_ALL_CATALOG_PATH,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    overwrite: bool = False,
) -> int:
    """Convert the fixed-width VSX raw file to ``vsx_all.parquet`` in chunks."""
    raw_path = Path(raw_path).expanduser()
    output_path = Path(output_path).expanduser()
    if not raw_path.exists():
        raise FileNotFoundError(f"VSX raw file not found: {raw_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    marker_path = _complete_marker_path(output_path)
    if marker_path.exists():
        marker_path.unlink()

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    t0 = time.perf_counter()
    try:
        for chunk_index, chunk in enumerate(
            _iter_normalized_vsx_chunks(raw_path, chunk_rows=max(1, int(chunk_rows))),
            start=1,
        ):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    tmp_path,
                    table.schema,
                    compression=PARQUET_OUTPUT_COMPRESSION,
                )
            writer.write_table(table)
            total_rows += int(len(chunk))
            print(
                f"  parsed chunk {chunk_index}: {len(chunk):,} usable row(s); "
                f"{total_rows:,} total",
                flush=True,
            )
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        empty = coerce_vsx_catalog_columns(pd.DataFrame(columns=vsx_columns))
        table = pa.Table.from_pandas(empty, preserve_index=False)
        pq.write_table(table, tmp_path, compression=PARQUET_OUTPUT_COMPRESSION)

    tmp_path.replace(output_path)
    _write_complete_marker(output_path=output_path, raw_path=raw_path, row_count=total_rows)
    print(
        f"Wrote full VSX parquet: {output_path} "
        f"({total_rows:,} row(s) in {time.perf_counter() - t0:.1f}s)"
    )
    return total_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the full AAVSO VSX catalog and build input/vsx/vsx_all.parquet.",
    )
    parser.add_argument("--url", default=DEFAULT_VSX_URL, help="CDS/VizieR VSX fixed-width catalog URL")
    parser.add_argument("--raw-path", type=Path, default=VSX_RAW_CATALOG_PATH, help="Raw fixed-width VSX output path")
    parser.add_argument("--output", type=Path, default=VSX_ALL_CATALOG_PATH, help="Full VSX Parquet output path")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS, help="Rows per fixed-width parse chunk")
    parser.add_argument("--timeout", type=float, default=120.0, help="Network timeout in seconds")
    parser.add_argument("--skip-download", action="store_true", help="Use an existing raw file and only build Parquet")
    parser.add_argument("--download-only", action="store_true", help="Download the raw file without building Parquet")
    parser.add_argument("--force-download", action="store_true", help="Replace any existing raw file instead of resuming")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing Parquet output")
    parser.add_argument(
        "--allow-partial-raw",
        action="store_true",
        help="With --skip-download, allow converting a raw file whose size does not match the remote VSX file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    raw_path = args.raw_path.expanduser()
    output_path = args.output.expanduser()

    if not args.skip_download:
        download_vsx_raw(
            url=args.url,
            raw_path=raw_path,
            force=bool(args.force_download),
            timeout=float(args.timeout),
        )
    elif args.allow_partial_raw:
        if not raw_path.exists():
            raise FileNotFoundError(f"--skip-download requested, but raw file is missing: {raw_path}")
    else:
        verify_vsx_raw_complete(raw_path=raw_path, url=args.url, timeout=float(args.timeout))

    if args.download_only:
        return 0

    convert_vsx_raw_to_parquet(
        raw_path=raw_path,
        output_path=output_path,
        chunk_rows=int(args.chunk_rows),
        overwrite=bool(args.overwrite),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
