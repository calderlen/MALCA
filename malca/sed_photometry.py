"""CLI for building normalized SED photometry tables."""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from malca.candidates import select_passing_candidates_if_present
from malca.feature_layers import with_feature_columns
from malca.sed_model import (
    SED_MODEL_CURVE_COLUMNS,
    SED_MODEL_FIT_COLUMNS,
    fit_sed_models,
    upsert_sed_model_results,
)
from malca.review.sed import (
    ALL_CATALOG_SOURCES,
    DEFAULT_PIPELINE_SED_SOURCES,
    SED_COLUMNS,
    fetch_sed_photometry,
    resolve_sed_sources,
    upsert_sed_rows,
)
from malca.review.store import db_connect
from malca.table_io import read_feature_table, write_parquet_table


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malca sed-photometry",
        description="Fetch and normalize broadband SED photometry for review candidates.",
    )
    parser.add_argument("input", type=Path, help="Input candidate table (.parquet, .csv, or .txt)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output SED Parquet path (default: <input>_sed_photometry.parquet)",
    )
    parser.add_argument(
        "--review-db",
        type=Path,
        default=None,
        help="Optional review SQLite DB to upsert SED rows into",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="default",
        help=(
            "Comma-separated source keys, 'default', 'all', or 'far-ir'. "
            "Default is the broad-classification set; far-IR sources are explicit. "
            f"Default: {', '.join(DEFAULT_PIPELINE_SED_SOURCES)}. "
            f"Available: {', '.join(ALL_CATALOG_SOURCES)}"
        ),
    )
    parser.add_argument(
        "--fit-atmosphere",
        dest="fit_atmosphere",
        action="store_true",
        default=True,
        help="Fit mandatory pystellibs Castelli/Kurucz atmosphere models after photometry (default).",
    )
    parser.add_argument(
        "--no-fit-atmosphere",
        dest="fit_atmosphere",
        action="store_false",
        help="Only write SED photometry; skip Castelli/Kurucz atmosphere fitting.",
    )
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="Fetch SED photometry for all input rows instead of only failed_any=False passers.",
    )
    return parser


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_sed_photometry.parquet")


def _default_model_output_paths(output_path: Path) -> tuple[Path, Path]:
    stem = output_path.stem
    if stem == "sed_photometry":
        prefix = ""
    elif stem.endswith("_sed_photometry"):
        prefix = stem[: -len("_sed_photometry")]
    else:
        prefix = stem
    prefix_text = f"{prefix}_" if prefix else ""
    return (
        output_path.with_name(f"{prefix_text}sed_model_fits.parquet"),
        output_path.with_name(f"{prefix_text}sed_model_curves.parquet"),
    )


def _is_sqlite_input(path: Path) -> bool:
    return path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}


def _ensure_candidate_id(df: pd.DataFrame) -> pd.DataFrame:
    if "candidate_id" in df.columns:
        return df
    if "asas_sn_id" not in df.columns:
        return df
    out = df.copy()
    out["candidate_id"] = out["asas_sn_id"].astype(str)
    return out


def _is_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    try:
        return not bool(pd.isna(value))
    except Exception:
        return True


def _expand_sqlite_candidate_payloads(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "payload_json" not in df.columns:
        return df

    records: list[dict] = []
    for raw in df.to_dict("records"):
        payload_raw = raw.get("payload_json")
        if isinstance(payload_raw, str) and payload_raw.strip():
            try:
                payload = json.loads(payload_raw)
            except Exception:
                payload = {}
        else:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        merged = dict(payload) if payload else {}
        for key, value in raw.items():
            if _is_present(value):
                merged[key] = value
            elif key not in merged:
                merged[key] = value
        records.append(merged)
    return pd.DataFrame.from_records(records)


def _read_candidate_table(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(input_path)
    if _is_sqlite_input(input_path):
        with closing(sqlite3.connect(input_path)) as conn:
            has_candidates = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'candidates'"
            ).fetchone()
            if has_candidates is None:
                raise ValueError(f"SQLite input {input_path} does not contain a candidates table")
            return _expand_sqlite_candidate_payloads(pd.read_sql_query("SELECT * FROM candidates", conn))
    return read_feature_table(input_path)


def run(args: argparse.Namespace) -> Path:
    input_path = args.input.expanduser()
    output_path = (args.output or _default_output_path(input_path)).expanduser()
    fit_atmosphere = bool(getattr(args, "fit_atmosphere", True))
    review_db_path = (
        args.review_db.expanduser()
        if args.review_db
        else (input_path if _is_sqlite_input(input_path) else None)
    )

    df = _read_candidate_table(input_path)
    df = _ensure_candidate_id(df)
    df = with_feature_columns(df, ["failed_any", "ra", "dec", "gaia_id"])
    if not getattr(args, "all_candidates", False):
        df = select_passing_candidates_if_present(df, printer=print)
    requested_sources = args.sources
    print(f"Loaded {len(df)} candidates from {input_path}")
    print(f"SED sources: {', '.join(resolve_sed_sources(requested_sources))}")

    rows = fetch_sed_photometry(
        df,
        sources=requested_sources,
        progress_callback=lambda msg: print(msg, flush=True),
    )
    for col in SED_COLUMNS:
        if col not in rows.columns:
            rows[col] = None
    rows = rows[SED_COLUMNS]
    write_parquet_table(rows, output_path)
    print(f"Saved {len(rows)} SED rows to {output_path}")

    if not rows.empty and "source" in rows.columns:
        counts = rows.groupby("source", dropna=False).size().sort_index()
        print("\nRows by source:")
        for source, count in counts.items():
            print(f"  {source}: {count}")

    fits = pd.DataFrame(columns=SED_MODEL_FIT_COLUMNS)
    curves = pd.DataFrame(columns=SED_MODEL_CURVE_COLUMNS)
    fit_output_path, curve_output_path = _default_model_output_paths(output_path)
    if fit_atmosphere:
        fits, curves = fit_sed_models(
            df,
            rows,
            progress_callback=lambda msg: print(msg, flush=True),
        )
        for col in SED_MODEL_FIT_COLUMNS:
            if col not in fits.columns:
                fits[col] = None
        for col in SED_MODEL_CURVE_COLUMNS:
            if col not in curves.columns:
                curves[col] = None
        fits = fits[SED_MODEL_FIT_COLUMNS]
        curves = curves[SED_MODEL_CURVE_COLUMNS]
        write_parquet_table(fits, fit_output_path)
        write_parquet_table(curves, curve_output_path)
        n_ok = int((fits["status"].astype(str) == "ok").sum()) if "status" in fits.columns else 0
        print(f"Saved {len(fits)} SED model fit rows to {fit_output_path} ({n_ok} ok)")
        print(f"Saved {len(curves)} SED model curve rows to {curve_output_path}")

    if review_db_path:
        with closing(db_connect(review_db_path)) as conn:
            updated = upsert_sed_rows(conn, rows)
            if fit_atmosphere:
                n_fits, n_curves = upsert_sed_model_results(conn, fits, curves)
            else:
                n_fits, n_curves = 0, 0
        print(f"\nUpserted {updated} SED rows into {review_db_path}")
        if fit_atmosphere:
            print(f"Upserted {n_fits} SED model fit rows and {n_curves} curve rows into {review_db_path}")

    return output_path


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
