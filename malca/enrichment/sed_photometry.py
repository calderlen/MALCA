"""CLI for building normalized SED photometry tables."""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from malca.products.candidates import select_passing_candidates_if_present
from malca.products.feature_layers import with_feature_columns
from malca.enrichment.sed_model import (
    SED_MODEL_CURVE_COLUMNS,
    SED_MODEL_FIT_COLUMNS,
    SED_MODEL_POINT_COLUMNS,
    fit_sed_models,
    upsert_sed_model_results,
)
from malca.enrichment.sed_alpha import (
    SED_ALPHA_COLUMNS,
    compute_sed_alpha_features,
    upsert_sed_alpha_results,
)
from malca.enrichment.sed_archive import (
    ARCHIVE_QUERY_TIMEOUT_SECONDS,
    ARCHIVE_DISCOVERY_SOURCE_KEYS,
    discover_sed_archive_products,
    resolve_archive_discovery_source_keys,
)
from malca.review.sed import (
    ALL_CATALOG_SOURCES,
    CANONICAL_SED_COLUMNS,
    DEFAULT_PIPELINE_SED_SOURCES,
    SED_FETCH_CHUNK_SIZE,
    SED_FETCH_MANIFEST_ATTR,
    SED_FETCH_MAX_ATTEMPTS,
    SED_FETCH_RETRY_BASE_SECONDS,
    build_sed_fetch_manifest,
    fetch_sed_photometry,
    resolve_sed_sources,
    upsert_sed_rows,
    validate_sed_fetch_manifest,
)
from malca.review.store import db_connect
from malca.review.sed_storage import (
    SED_ARCHIVE_COVERAGE_COLUMNS,
    SED_ARCHIVE_PRODUCT_COLUMNS,
    SED_IMAGE_JOB_COLUMNS,
    enqueue_sed_image_jobs,
    ensure_sed_storage_schema,
    upsert_sed_archive_coverage,
    upsert_sed_archive_products,
)
from malca.io.table_io import read_feature_table, read_parquet_table, write_parquet_table


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
            "Comma-separated source keys, 'default'/'all', 'broad', or 'far-ir'. "
            "Default uses the bounded payload/IRSA-AllWISE/PS1/SkyMapper/SDSS profile; "
            "'all' explicitly fetches every registered SED catalog. "
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
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help="Limit acquisition/discovery to this candidate ID; repeat as needed.",
    )
    parser.add_argument(
        "--fit-workers",
        type=int,
        default=1,
        help="Parallel threads for atmosphere fitting (default: 1).",
    )
    parser.add_argument(
        "--fetch-chunk-size",
        type=int,
        default=SED_FETCH_CHUNK_SIZE,
        help=f"Candidates checkpointed per source fetch chunk (default: {SED_FETCH_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--fetch-max-attempts",
        type=int,
        default=SED_FETCH_MAX_ATTEMPTS,
        help=f"Attempts for retryable source errors (default: {SED_FETCH_MAX_ATTEMPTS}).",
    )
    parser.add_argument(
        "--fetch-retry-base-seconds",
        type=float,
        default=SED_FETCH_RETRY_BASE_SECONDS,
        help=(
            "Initial exponential-backoff delay for retryable source errors "
            f"(default: {SED_FETCH_RETRY_BASE_SECONDS:g} s)."
        ),
    )
    parser.add_argument(
        "--no-alpha",
        dest="compute_alpha",
        action="store_false",
        default=True,
        help="Skip 2-24 micron SED spectral-index feature calculation.",
    )
    parser.add_argument(
        "--discover-archive-products",
        dest="discover_archive_products",
        action="store_true",
        default=True,
        help=(
            "Discover Spitzer SEIP, Herschel HSA, and APEX products for the "
            "selected archive-backed sources (default)."
        ),
    )
    parser.add_argument(
        "--no-discover-archive-products",
        dest="discover_archive_products",
        action="store_false",
        help="Skip archive coverage/product discovery; catalog photometry is still fetched.",
    )
    parser.add_argument(
        "--archive-coverage-output",
        type=Path,
        default=None,
        help="Coverage ledger Parquet path (default: beside the SED output).",
    )
    parser.add_argument(
        "--archive-products-output",
        type=Path,
        default=None,
        help="Archive product ledger Parquet path (default: beside the SED output).",
    )
    parser.add_argument(
        "--image-jobs-output",
        type=Path,
        default=None,
        help="Image-measurement job ledger Parquet path (default: beside the SED output).",
    )
    parser.add_argument(
        "--archive-query-timeout-seconds",
        type=float,
        default=ARCHIVE_QUERY_TIMEOUT_SECONDS,
        help=(
            "Connection/read timeout for each archive-discovery request "
            f"(default: {ARCHIVE_QUERY_TIMEOUT_SECONDS:g} s)."
        ),
    )
    parser.add_argument(
        "--archive-checkpoint-size",
        type=int,
        default=25,
        help="Completed archive target queries per atomic ledger checkpoint (default: 25).",
    )
    parser.add_argument(
        "--refresh-archive-products",
        action="store_true",
        help=(
            "Ignore and replace existing archive-ledger checkpoints. By default, "
            "valid ledgers are resumed and completed target/source queries are reused."
        ),
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


def _default_model_point_output_path(output_path: Path) -> Path:
    stem = output_path.stem
    if stem == "sed_photometry":
        prefix = ""
    elif stem.endswith("_sed_photometry"):
        prefix = stem[: -len("_sed_photometry")]
    else:
        prefix = stem
    prefix_text = f"{prefix}_" if prefix else ""
    return output_path.with_name(f"{prefix_text}sed_model_points.parquet")


def _default_alpha_output_path(output_path: Path) -> Path:
    stem = output_path.stem
    if stem == "sed_photometry":
        prefix = ""
    elif stem.endswith("_sed_photometry"):
        prefix = stem[: -len("_sed_photometry")]
    else:
        prefix = stem
    prefix_text = f"{prefix}_" if prefix else ""
    return output_path.with_name(f"{prefix_text}sed_alpha.parquet")


def _default_fetch_manifest_output_path(output_path: Path) -> Path:
    stem = output_path.stem
    if stem == "sed_photometry":
        prefix = ""
    elif stem.endswith("_sed_photometry"):
        prefix = stem[: -len("_sed_photometry")]
    else:
        prefix = stem
    prefix_text = f"{prefix}_" if prefix else ""
    return output_path.with_name(f"{prefix_text}sed_fetch_manifest.parquet")


def _default_archive_output_paths(output_path: Path) -> tuple[Path, Path, Path]:
    stem = output_path.stem
    if stem == "sed_photometry":
        prefix = ""
    elif stem.endswith("_sed_photometry"):
        prefix = stem[: -len("_sed_photometry")]
    else:
        prefix = stem
    prefix_text = f"{prefix}_" if prefix else ""
    return (
        output_path.with_name(f"{prefix_text}sed_archive_coverage.parquet"),
        output_path.with_name(f"{prefix_text}sed_archive_products.parquet"),
        output_path.with_name(f"{prefix_text}sed_image_measurement_jobs.parquet"),
    )


def _empty_archive_ledgers() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(columns=SED_ARCHIVE_COVERAGE_COLUMNS),
        pd.DataFrame(columns=SED_ARCHIVE_PRODUCT_COLUMNS),
        pd.DataFrame(columns=SED_IMAGE_JOB_COLUMNS),
    )


def _canonical_archive_ledger(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    key_column: str,
) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = None
    if key_column in out.columns:
        out = out.drop_duplicates(subset=[key_column], keep="last")
    return out[columns].reset_index(drop=True)


def _load_archive_checkpoint(
    *,
    coverage_path: Path,
    products_path: Path,
    jobs_path: Path,
    candidates: pd.DataFrame,
    sources: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool, bool]:
    """Load a structurally valid exact-scope archive checkpoint, if present."""

    paths = (coverage_path, products_path, jobs_path)
    exists = tuple(path.exists() for path in paths)
    if not any(exists):
        return (*_empty_archive_ledgers(), False, False)
    if not all(exists):
        missing = [str(path) for path, present in zip(paths, exists) if not present]
        raise RuntimeError(
            "Archive checkpoint is incomplete; refusing to mix old and new ledgers. "
            "Missing: "
            + ", ".join(missing)
            + ". Restore the missing ledger or pass --refresh-archive-products."
        )

    coverage = read_parquet_table(coverage_path)
    products = read_parquet_table(products_path)
    jobs = read_parquet_table(jobs_path)
    required = (
        (
            "coverage",
            coverage,
            {"coverage_id", "candidate_id", "source_key", "coverage_status", "discovery_signature"},
        ),
        (
            "products",
            products,
            {"product_id", "coverage_id", "candidate_id", "source_key"},
        ),
        (
            "jobs",
            jobs,
            {"job_id", "coverage_id", "candidate_id", "source_key"},
        ),
    )
    schema_errors: list[str] = []
    for label, frame, columns in required:
        missing_columns = sorted(columns - set(frame.columns))
        if missing_columns:
            schema_errors.append(f"{label} missing columns {missing_columns}")
    if schema_errors:
        raise RuntimeError(
            "Archive checkpoint schema is invalid: "
            + "; ".join(schema_errors)
            + ". Pass --refresh-archive-products to replace it."
        )

    expected_ids = set(candidates["candidate_id"].astype(str))
    expected_sources = set(resolve_archive_discovery_source_keys(sources))
    for label, frame, _ in required:
        actual_ids = set(frame["candidate_id"].dropna().astype(str))
        actual_sources = set(frame["source_key"].dropna().astype(str))
        extra_ids = actual_ids - expected_ids
        extra_sources = actual_sources - expected_sources
        if extra_ids or extra_sources:
            details: list[str] = []
            if extra_ids:
                details.append(f"{len(extra_ids)} candidate IDs outside this selection")
            if extra_sources:
                details.append(f"unexpected sources {sorted(extra_sources)}")
            raise RuntimeError(
                f"Archive {label} checkpoint has the wrong scope ("
                + ", ".join(details)
                + "). Use a different output path or pass --refresh-archive-products."
            )

    if (
        coverage["discovery_signature"].isna()
        | coverage["discovery_signature"].astype(str).str.strip().eq("")
    ).any():
        raise RuntimeError(
            "Archive coverage checkpoint contains blank discovery signatures; "
            "pass --refresh-archive-products to replace it."
        )
    coverage_ids = set(coverage["coverage_id"].dropna().astype(str))
    invalid_product_refs = set(products["coverage_id"].dropna().astype(str)) - coverage_ids
    invalid_job_refs = set(jobs["coverage_id"].dropna().astype(str)) - coverage_ids
    if invalid_product_refs or invalid_job_refs:
        raise RuntimeError(
            "Archive checkpoint contains product/job rows without matching coverage "
            f"({len(invalid_product_refs)} product refs, {len(invalid_job_refs)} job refs); "
            "pass --refresh-archive-products to replace it."
        )

    canonical_coverage = _canonical_archive_ledger(
        coverage,
        columns=SED_ARCHIVE_COVERAGE_COLUMNS,
        key_column="coverage_id",
    )
    canonical_products = _canonical_archive_ledger(
        products,
        columns=SED_ARCHIVE_PRODUCT_COLUMNS,
        key_column="product_id",
    )
    canonical_jobs = _canonical_archive_ledger(
        jobs,
        columns=SED_IMAGE_JOB_COLUMNS,
        key_column="job_id",
    )
    normalized = (
        len(canonical_coverage) != len(coverage)
        or len(canonical_products) != len(products)
        or len(canonical_jobs) != len(jobs)
    )
    return (
        canonical_coverage,
        canonical_products,
        canonical_jobs,
        True,
        normalized,
    )


def _completed_archive_candidates_by_source(
    coverage: pd.DataFrame,
) -> dict[str, set[str]]:
    """Return target/source queries with a terminal non-error coverage result."""

    completed: dict[str, set[str]] = {}
    if coverage.empty:
        return completed
    for (source_key, candidate_id), group in coverage.groupby(
        ["source_key", "candidate_id"],
        dropna=False,
    ):
        statuses = {
            str(value).strip().casefold()
            for value in group["coverage_status"]
            if str(value).strip()
        }
        if statuses - {"query_error"}:
            completed.setdefault(str(source_key), set()).add(str(candidate_id))
    return completed


def _merge_archive_checkpoint_rows(
    current: pd.DataFrame,
    rows: pd.DataFrame | list[dict],
    *,
    columns: list[str],
    key_column: str,
) -> pd.DataFrame:
    if rows is None:
        return current
    incoming = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if incoming.empty:
        return current
    return _canonical_archive_ledger(
        pd.concat([current, incoming], ignore_index=True, sort=False),
        columns=columns,
        key_column=key_column,
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
    if review_db_path:
        with closing(db_connect(review_db_path)) as conn:
            schema_changed = ensure_sed_storage_schema(conn)
            conn.commit()
        if schema_changed:
            print(f"Initialized SED storage schema in {review_db_path}")

    df = _read_candidate_table(input_path)
    df = _ensure_candidate_id(df)
    df = with_feature_columns(df, ["failed_any", "ra", "dec", "gaia_id"])
    selected_candidate_ids = {
        str(value).strip()
        for value in getattr(args, "candidate_id", [])
        if str(value).strip()
    }
    if selected_candidate_ids:
        available = set(df["candidate_id"].astype(str))
        missing = sorted(selected_candidate_ids - available)
        if missing:
            raise KeyError(
                "Requested candidate ID(s) are absent from the input: "
                + ", ".join(missing)
            )
        df = df[df["candidate_id"].astype(str).isin(selected_candidate_ids)].copy()
    elif not getattr(args, "all_candidates", False):
        df = select_passing_candidates_if_present(df, printer=print)
    requested_sources = args.sources
    print(f"Loaded {len(df)} candidates from {input_path}")
    print(f"SED sources: {', '.join(resolve_sed_sources(requested_sources))}")

    rows = fetch_sed_photometry(
        df,
        sources=requested_sources,
        progress_callback=lambda msg: print(msg, flush=True),
        fetch_chunk_size=max(int(getattr(args, "fetch_chunk_size", SED_FETCH_CHUNK_SIZE)), 1),
        max_attempts=max(int(getattr(args, "fetch_max_attempts", SED_FETCH_MAX_ATTEMPTS)), 1),
        retry_base_seconds=max(
            float(getattr(args, "fetch_retry_base_seconds", SED_FETCH_RETRY_BASE_SECONDS)),
            0.0,
        ),
    )
    requested = resolve_sed_sources(requested_sources)
    manifest = rows.attrs.get(SED_FETCH_MANIFEST_ATTR)
    if not isinstance(manifest, pd.DataFrame):
        manifest = build_sed_fetch_manifest(df, sources=requested, fetched_rows=rows)
    # The manifest is a DataFrame stored as transient fetch metadata.  Detach it
    # before slicing/writing rows: pandas compares DataFrame.attrs during the
    # chunked writer's schema concat, and DataFrame-valued attrs cannot be
    # reduced to a single truth value.
    rows.attrs.pop(SED_FETCH_MANIFEST_ATTR, None)
    for col in CANONICAL_SED_COLUMNS:
        if col not in rows.columns:
            rows[col] = None
    rows = rows[CANONICAL_SED_COLUMNS]
    write_parquet_table(rows, output_path)
    print(f"Saved {len(rows)} SED rows to {output_path}")

    manifest_output_path = _default_fetch_manifest_output_path(output_path)
    write_parquet_table(manifest, manifest_output_path)
    manifest_counts = (
        manifest.groupby(["source_key", "status"], dropna=False).size().unstack(fill_value=0)
        if not manifest.empty
        else pd.DataFrame()
    )
    n_incomplete = int((~manifest["is_complete"].astype(bool)).sum()) if not manifest.empty else 0
    print(
        f"Saved {len(manifest)} candidate-source fetch statuses to {manifest_output_path} "
        f"({n_incomplete} incomplete)"
    )
    if not manifest_counts.empty:
        print("\nFetch status by source:")
        print(manifest_counts.to_string())

    archive_coverage = pd.DataFrame()
    archive_products = pd.DataFrame()
    image_jobs = pd.DataFrame()
    archive_sources = set(requested) & ARCHIVE_DISCOVERY_SOURCE_KEYS
    if bool(getattr(args, "discover_archive_products", True)) and archive_sources:
        default_coverage_path, default_products_path, default_jobs_path = (
            _default_archive_output_paths(output_path)
        )
        coverage_path = (
            getattr(args, "archive_coverage_output", None) or default_coverage_path
        ).expanduser()
        products_path = (
            getattr(args, "archive_products_output", None) or default_products_path
        ).expanduser()
        jobs_path = (
            getattr(args, "image_jobs_output", None) or default_jobs_path
        ).expanduser()
        refresh_archive = bool(getattr(args, "refresh_archive_products", False))
        if refresh_archive:
            archive_coverage, archive_products, image_jobs = _empty_archive_ledgers()
            checkpoint_loaded = False
            checkpoint_normalized = False
            print("Refreshing archive discovery; existing ledger checkpoints will be replaced.")
        else:
            (
                archive_coverage,
                archive_products,
                image_jobs,
                checkpoint_loaded,
                checkpoint_normalized,
            ) = _load_archive_checkpoint(
                coverage_path=coverage_path,
                products_path=products_path,
                jobs_path=jobs_path,
                candidates=df,
                sources=requested,
            )
            if checkpoint_normalized:
                write_parquet_table(archive_coverage, coverage_path)
                write_parquet_table(archive_products, products_path)
                write_parquet_table(image_jobs, jobs_path)
                print(
                    "Normalized duplicate stable IDs in the archive checkpoint "
                    "before resuming."
                )

        completed_by_source = _completed_archive_candidates_by_source(archive_coverage)
        resolved_archive_sources = resolve_archive_discovery_source_keys(requested)
        expected_archive_queries = len(df) * len(resolved_archive_sources)
        completed_archive_queries = sum(
            len(completed_by_source.get(source_key, set()))
            for source_key in resolved_archive_sources
        )
        if checkpoint_loaded:
            print(
                "Loaded archive checkpoint: "
                f"{completed_archive_queries}/{expected_archive_queries} "
                "target/source queries complete"
            )

        checkpoint_size = max(int(getattr(args, "archive_checkpoint_size", 25)), 1)
        pending_checkpoint_targets = 0

        def persist_archive_checkpoint(*, force: bool = False) -> None:
            nonlocal pending_checkpoint_targets
            if pending_checkpoint_targets <= 0 and not force:
                return
            write_parquet_table(archive_coverage, coverage_path)
            write_parquet_table(archive_products, products_path)
            write_parquet_table(image_jobs, jobs_path)
            if pending_checkpoint_targets > 0:
                print(
                    "[SED archive] checkpointed "
                    f"{pending_checkpoint_targets} completed target queries",
                    flush=True,
                )
            pending_checkpoint_targets = 0

        def archive_checkpoint_callback(
            source_key: str,
            coverage_rows: list[dict],
            product_rows: list[dict],
            job_rows: list[dict],
        ) -> None:
            nonlocal archive_coverage, archive_products, image_jobs
            nonlocal pending_checkpoint_targets
            archive_coverage = _merge_archive_checkpoint_rows(
                archive_coverage,
                coverage_rows,
                columns=SED_ARCHIVE_COVERAGE_COLUMNS,
                key_column="coverage_id",
            )
            archive_products = _merge_archive_checkpoint_rows(
                archive_products,
                product_rows,
                columns=SED_ARCHIVE_PRODUCT_COLUMNS,
                key_column="product_id",
            )
            image_jobs = _merge_archive_checkpoint_rows(
                image_jobs,
                job_rows,
                columns=SED_IMAGE_JOB_COLUMNS,
                key_column="job_id",
            )
            pending_checkpoint_targets += 1
            if pending_checkpoint_targets >= checkpoint_size:
                persist_archive_checkpoint()

        if completed_archive_queries < expected_archive_queries:
            try:
                (
                    discovered_coverage,
                    discovered_products,
                    discovered_jobs,
                ) = discover_sed_archive_products(
                    df,
                    sources=requested,
                    progress_callback=lambda msg: print(msg, flush=True),
                    checkpoint_callback=archive_checkpoint_callback,
                    completed_candidate_ids_by_source=completed_by_source,
                    query_timeout_seconds=max(
                        float(
                            getattr(
                                args,
                                "archive_query_timeout_seconds",
                                ARCHIVE_QUERY_TIMEOUT_SECONDS,
                            )
                        ),
                        0.0,
                    ),
                )
                archive_coverage = _merge_archive_checkpoint_rows(
                    archive_coverage,
                    discovered_coverage,
                    columns=SED_ARCHIVE_COVERAGE_COLUMNS,
                    key_column="coverage_id",
                )
                archive_products = _merge_archive_checkpoint_rows(
                    archive_products,
                    discovered_products,
                    columns=SED_ARCHIVE_PRODUCT_COLUMNS,
                    key_column="product_id",
                )
                image_jobs = _merge_archive_checkpoint_rows(
                    image_jobs,
                    discovered_jobs,
                    columns=SED_IMAGE_JOB_COLUMNS,
                    key_column="job_id",
                )
            except BaseException:
                persist_archive_checkpoint(force=pending_checkpoint_targets > 0)
                raise
            persist_archive_checkpoint(force=True)
        else:
            print("[SED archive] Reusing complete archive ledgers; no remote discovery needed.")

        completed_by_source = _completed_archive_candidates_by_source(archive_coverage)
        completed_archive_queries = sum(
            len(completed_by_source.get(source_key, set()))
            for source_key in resolved_archive_sources
        )
        incomplete_archive_queries = expected_archive_queries - completed_archive_queries
        print(
            "Archive ledgers ready: "
            f"{len(archive_coverage)} coverage rows, "
            f"{len(archive_products)} product rows, "
            f"{len(image_jobs)} image jobs "
            f"({incomplete_archive_queries} retryable target/source queries)"
        )
        if review_db_path:
            with closing(db_connect(review_db_path)) as conn:
                n_coverage = upsert_sed_archive_coverage(conn, archive_coverage)
                n_products = upsert_sed_archive_products(conn, archive_products)
                n_jobs = enqueue_sed_image_jobs(conn, image_jobs)
            print(
                f"Upserted archive ledgers into {review_db_path}: "
                f"{n_coverage} coverage, {n_products} products, {n_jobs} jobs"
            )

    fetch_complete, manifest_errors = validate_sed_fetch_manifest(
        manifest,
        df,
        sources=requested,
    )
    if not fetch_complete:
        raise RuntimeError(
            "SED fetch is incomplete; resumable photometry and manifest were saved. "
            + "; ".join(manifest_errors)
        )

    if not rows.empty and "source" in rows.columns:
        counts = rows.groupby("source", dropna=False).size().sort_index()
        print("\nRows by source:")
        for source, count in counts.items():
            print(f"  {source}: {count}")

    fits = pd.DataFrame(columns=SED_MODEL_FIT_COLUMNS)
    curves = pd.DataFrame(columns=SED_MODEL_CURVE_COLUMNS)
    model_points = pd.DataFrame(columns=SED_MODEL_POINT_COLUMNS)
    alpha_rows = pd.DataFrame(columns=SED_ALPHA_COLUMNS)
    alpha_output_path = _default_alpha_output_path(output_path)
    if bool(getattr(args, "compute_alpha", True)):
        alpha_rows = compute_sed_alpha_features(df, rows)
        for col in SED_ALPHA_COLUMNS:
            if col not in alpha_rows.columns:
                alpha_rows[col] = None
        alpha_rows = alpha_rows[SED_ALPHA_COLUMNS]
        write_parquet_table(alpha_rows, alpha_output_path)
        n_ok_alpha = (
            int((alpha_rows["sed_alpha_status"].astype(str) == "ok").sum())
            if "sed_alpha_status" in alpha_rows.columns
            else 0
        )
        print(f"Saved {len(alpha_rows)} SED alpha rows to {alpha_output_path} ({n_ok_alpha} ok)")

    fit_output_path, curve_output_path = _default_model_output_paths(output_path)
    point_output_path = _default_model_point_output_path(output_path)
    if fit_atmosphere:
        fits, curves, model_points = fit_sed_models(
            df,
            rows,
            progress_callback=lambda msg: print(msg, flush=True),
            return_points=True,
            workers=max(int(getattr(args, "fit_workers", 1)), 1),
        )
        for col in SED_MODEL_FIT_COLUMNS:
            if col not in fits.columns:
                fits[col] = None
        for col in SED_MODEL_CURVE_COLUMNS:
            if col not in curves.columns:
                curves[col] = None
        for col in SED_MODEL_POINT_COLUMNS:
            if col not in model_points.columns:
                model_points[col] = None
        fits = fits[SED_MODEL_FIT_COLUMNS]
        curves = curves[SED_MODEL_CURVE_COLUMNS]
        model_points = model_points[SED_MODEL_POINT_COLUMNS]
        write_parquet_table(fits, fit_output_path)
        write_parquet_table(curves, curve_output_path)
        write_parquet_table(model_points, point_output_path)
        n_ok = int((fits["status"].astype(str) == "ok").sum()) if "status" in fits.columns else 0
        print(f"Saved {len(fits)} SED model fit rows to {fit_output_path} ({n_ok} ok)")
        print(f"Saved {len(curves)} SED model curve rows to {curve_output_path}")
        print(f"Saved {len(model_points)} SED model point rows to {point_output_path}")

    if review_db_path:
        with closing(db_connect(review_db_path)) as conn:
            updated = upsert_sed_rows(conn, rows)
            n_alpha = (
                upsert_sed_alpha_results(conn, alpha_rows)
                if bool(getattr(args, "compute_alpha", True))
                else 0
            )
            if fit_atmosphere:
                n_fits, n_curves = upsert_sed_model_results(conn, fits, curves, model_points)
            else:
                n_fits, n_curves = 0, 0
        print(f"\nUpserted {updated} SED rows into {review_db_path}")
        if bool(getattr(args, "compute_alpha", True)):
            print(f"Upserted {n_alpha} SED alpha rows into {review_db_path}")
        if fit_atmosphere:
            print(f"Upserted {n_fits} SED model fit rows and {n_curves} curve rows into {review_db_path}")

    return output_path


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
