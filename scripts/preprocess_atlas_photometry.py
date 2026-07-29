#!/usr/bin/env python3
"""Preprocess reduced/direct ATLAS forced-photometry Parquet files.

The input files are never modified.  For every ``atlas_lc_*.parquet`` input,
the script writes:

* ``<output>/clean/<relative path>``: only accepted c/o detections, with
  conventional AB magnitudes recomputed from positive flux.
* ``<output>/flagged/<relative path>``: every original row, with quality flags,
  rejection reasons, flux S/N, and clean-magnitude columns.
* ``<output>/atlas_preprocess_summary.json``: batch settings and row counts.

The ATLAS FAQ measurement-quality cuts are:

    duJy < 10000
    err == 0
    100 < x < 10460
    100 < y < 10460
    1.6 < maj < 5
    1.6 < min < 5
    -1 < apfit < -0.1
    mag5sig > 17
    Sky > 17

These cuts do not by themselves make the signed ATLAS ``m`` field an ordinary
magnitude.  This preprocessor therefore also requires reduced/direct image
photometry, a selected filter, positive finite ``uJy`` and ``duJy``, and a
configurable minimum ``uJy / duJy``.  It calculates ``m_clean`` and
``dm_clean`` from accepted fluxes without changing the raw ``m`` or ``dm``.

Example
-------
Process the currently downloaded July 1 ATLAS light curves:

    conda run -n malca python scripts/preprocess_atlas_photometry.py \
        output/runs/dat3-full-extended_2026-07-01-v4/results/external_lcs \
        --output-dir \
        output/runs/dat3-full-extended_2026-07-01-v4/results/atlas_preprocessed
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from malca.enrichment.atlas_forced_photometry import (
    ATLAS_PREPROCESS_DEFAULT_FILTERS as DEFAULT_FILTERS,
    ATLAS_PREPROCESS_DEFAULT_SNR_MIN as DEFAULT_SNR_MIN,
    ATLAS_PREPROCESS_VERSION,
    atlas_science_view,
    preprocess_atlas_frame,
)

DEFAULT_PATTERN = "atlas_lc_*.parquet"


def _rejection_counts(flagged: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    rejected = flagged.loc[~flagged["atlas_good"], "atlas_reject_reason"]
    for value in rejected:
        for reason in str(value).split(";"):
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _discover_inputs(input_path: Path, pattern: str) -> tuple[Path, list[Path]]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".parquet":
            raise ValueError(f"ATLAS input must be Parquet: {input_path}")
        return input_path.parent, [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)

    paths = sorted(path for path in input_path.rglob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(
            f"No ATLAS Parquet files matching {pattern!r} under {input_path}"
        )
    return input_path, paths


def _validate_output_location(input_path: Path, output_dir: Path) -> None:
    if input_path.is_file():
        if output_dir == input_path.parent:
            raise ValueError(
                "output_dir must differ from the input file's directory so raw "
                "and preprocessed products cannot be confused"
            )
        return

    if output_dir == input_path or input_path in output_dir.parents:
        raise ValueError(
            "output_dir must be outside the input tree so reruns cannot ingest "
            "their own preprocessed products"
        )


def run_preprocessing(
    input_path: Path,
    output_dir: Path,
    *,
    snr_min: float = DEFAULT_SNR_MIN,
    filters: Iterable[str] = DEFAULT_FILTERS,
    pattern: str = DEFAULT_PATTERN,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Preprocess one file or a directory tree and return the batch summary."""
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_root, sources = _discover_inputs(input_path, pattern)
    _validate_output_location(input_path, output_dir)

    filters = tuple(
        dict.fromkeys(str(value).strip().lower() for value in filters if str(value).strip())
    )
    destinations = []
    for source in sources:
        relative = source.relative_to(source_root)
        destinations.append(
            (
                source,
                output_dir / "clean" / relative,
                output_dir / "flagged" / relative,
            )
        )
    summary_path = output_dir / "atlas_preprocess_summary.json"

    if not overwrite and not dry_run:
        existing = [
            path
            for _source, clean_path, flagged_path in destinations
            for path in (clean_path, flagged_path)
            if path.exists()
        ]
        if summary_path.exists():
            existing.append(summary_path)
        if existing:
            examples = ", ".join(str(path) for path in existing[:3])
            raise FileExistsError(
                f"Refusing to overwrite {len(existing)} existing output(s); "
                f"examples: {examples}. Pass --overwrite to replace them."
            )

    file_summaries: list[dict[str, object]] = []
    total_rows = 0
    total_good = 0
    total_faq_good = 0
    aggregate_rejections: dict[str, int] = {}

    for source, clean_path, flagged_path in destinations:
        raw = pd.read_parquet(source)
        flagged = preprocess_atlas_frame(raw, snr_min=snr_min, filters=filters)
        clean = atlas_science_view(raw, snr_min=snr_min, filters=filters)

        if not dry_run:
            _atomic_write_parquet(clean, clean_path)
            _atomic_write_parquet(flagged, flagged_path)

        rejection_counts = _rejection_counts(flagged)
        for reason, count in rejection_counts.items():
            aggregate_rejections[reason] = aggregate_rejections.get(reason, 0) + count

        rows_total = int(len(flagged))
        rows_good = int(flagged["atlas_good"].sum())
        rows_faq_good = int(flagged["atlas_faq_good"].sum())
        total_rows += rows_total
        total_good += rows_good
        total_faq_good += rows_faq_good
        file_summaries.append(
            {
                "input": str(source),
                "clean_output": None if dry_run else str(clean_path),
                "flagged_output": None if dry_run else str(flagged_path),
                "rows_total": rows_total,
                "rows_faq_good": rows_faq_good,
                "rows_good": rows_good,
                "rows_rejected": rows_total - rows_good,
                "rejection_counts": rejection_counts,
            }
        )

    summary: dict[str, object] = {
        "atlas_preprocess_version": ATLAS_PREPROCESS_VERSION,
        "input": str(input_path),
        "output_dir": None if dry_run else str(output_dir),
        "dry_run": bool(dry_run),
        "pattern": pattern,
        "filters": list(filters),
        "snr_min": float(snr_min),
        "files_processed": len(file_summaries),
        "rows_total": total_rows,
        "rows_faq_good": total_faq_good,
        "rows_good": total_good,
        "rows_rejected": total_rows - total_good,
        "fraction_good": (total_good / total_rows) if total_rows else None,
        "rejection_counts": dict(
            sorted(
                aggregate_rejections.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "files": file_summaries,
    }
    if not dry_run:
        _atomic_write_json(summary, summary_path)
    return summary


def _parse_filters(value: str) -> tuple[str, ...]:
    filters = tuple(
        dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip())
    )
    if not filters:
        raise argparse.ArgumentTypeError(
            "Use a comma-separated filter list such as c,o"
        )
    return filters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="An ATLAS Parquet file or directory containing atlas_lc_*.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Separate destination root for clean, flagged, and summary products",
    )
    parser.add_argument(
        "--snr-min",
        type=float,
        default=DEFAULT_SNR_MIN,
        help=f"Minimum positive-flux S/N for a detection (default: {DEFAULT_SNR_MIN:g})",
    )
    parser.add_argument(
        "--filters",
        type=_parse_filters,
        default=DEFAULT_FILTERS,
        help="Comma-separated retained filters (default: c,o)",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Filename pattern for directory input (default: {DEFAULT_PATTERN})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing preprocessed outputs; raw inputs remain untouched",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and summarize every input without writing output files",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_preprocessing(
        args.input,
        args.output_dir,
        snr_min=args.snr_min,
        filters=args.filters,
        pattern=args.pattern,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
