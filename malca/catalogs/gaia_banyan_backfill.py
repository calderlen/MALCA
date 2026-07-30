"""Targeted Gaia DR3 astrometry and BANYAN Sigma backfill for review cohorts."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from malca.catalogs.gaia_fetch import fetch_gaia_catalog
from malca.config import BANYAN_MIN_ASSOC_PROB, GAIA_LOCAL_CATALOG
from malca.enrichment.banyan import BANYAN_OUTPUT_COLUMNS, compute_banyan_membership
from malca.enrichment.characterize import (
    gaia_identifier_series,
    merge_gaia_catalog_rows,
    query_gaia_by_ids,
)
from malca.review.store import (
    db_connect,
    ensure_review_db_schema,
    get_candidate_payload,
    replace_candidate_payload_fields,
    validate_review_db_integrity,
)


GAIA_BACKFILL_COLUMNS = tuple(dict.fromkeys((
    "gaia_id",
    "source_id",
    "ra",
    "dec",
    "ref_epoch",
    "parallax",
    "parallax_error",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "radial_velocity",
    "radial_velocity_error",
    "astrometric_params_solved",
    "ruwe",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "bp_rp",
    "gaia_fetch_schema_version",
    "gaia_fetch_updated_at",
    "gaia_enrichment_status",
    "gaia_enrichment_source",
    "gaia_astrometry_complete",
    "gaia_banyan_input_complete",
    "gaia_missing_fields_json",
    "gaia_enrichment_updated_at",
    *BANYAN_OUTPUT_COLUMNS,
)))


def _readonly_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def reviewed_candidate_ids(
    conn: sqlite3.Connection,
    *,
    cohort: str,
    morphology: str,
) -> list[str]:
    """Select a stable review cohort without relying on candidate payload JSON."""
    if cohort == "all-candidates":
        rows = conn.execute("SELECT candidate_id FROM candidates ORDER BY candidate_id").fetchall()
    elif cohort == "all-reviewed":
        rows = conn.execute(
            """
            SELECT c.candidate_id
            FROM candidates c
            JOIN reviews r ON r.candidate_id = c.candidate_id
            WHERE lower(coalesce(r.workflow_status, r.status, '')) = 'reviewed'
            ORDER BY c.candidate_id
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT c.candidate_id
            FROM candidates c
            JOIN reviews r ON r.candidate_id = c.candidate_id
            WHERE lower(coalesce(r.workflow_status, r.status, '')) = 'reviewed'
              AND lower(coalesce(r.morphology_primary, '')) = lower(?)
            ORDER BY c.candidate_id
            """,
            (morphology,),
        ).fetchall()
    return [str(row[0]) for row in rows]


def load_review_cohort(
    review_db: Path,
    *,
    cohort: str = "reviewed-dippers",
    morphology: str = "dimming_event",
) -> pd.DataFrame:
    """Load merged candidate payloads for the selected cohort read-only."""
    with _readonly_connection(review_db) as conn:
        candidate_ids = reviewed_candidate_ids(conn, cohort=cohort, morphology=morphology)
        records: list[dict[str, object]] = []
        for candidate_id in candidate_ids:
            record = get_candidate_payload(conn, candidate_id)
            record["candidate_id"] = candidate_id
            records.append(record)
    return pd.DataFrame(records)


def _finite_pair_count(frame: pd.DataFrame, left: str, right: str) -> int:
    if left not in frame.columns or right not in frame.columns:
        return 0
    a = pd.to_numeric(frame[left], errors="coerce")
    b = pd.to_numeric(frame[right], errors="coerce")
    return int((a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)).sum())


def _summary(frame: pd.DataFrame) -> dict[str, object]:
    identifiers = gaia_identifier_series(frame)
    status_counts = (
        frame["banyan_status"].fillna("<missing>").astype(str).value_counts().to_dict()
        if "banyan_status" in frame.columns
        else {}
    )
    return {
        "rows": int(len(frame)),
        "gaia_ids": int(identifiers.notna().sum()),
        "parallax": _finite_pair_count(frame, "parallax", "parallax"),
        "proper_motion": _finite_pair_count(frame, "pmra", "pmdec"),
        "proper_motion_errors": _finite_pair_count(frame, "pmra_error", "pmdec_error"),
        "banyan_input_complete": int(
            frame.get("gaia_banyan_input_complete", pd.Series(False, index=frame.index))
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "banyan_field_probability": _finite_pair_count(
            frame, "banyan_field_prob", "banyan_field_prob"
        ),
        "banyan_status_counts": status_counts,
    }


def backfill_frame(
    frame: pd.DataFrame,
    *,
    gaia_cache: Path,
    fetch_gaia: bool,
    run_banyan: bool,
    allow_partial_fetch: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Refresh Gaia rows, merge by Gaia ID, and compute explicit BANYAN results."""
    before = _summary(frame)
    identifiers = gaia_identifier_series(frame)
    gaia_ids = identifiers.dropna().astype(str).drop_duplicates().tolist()
    if fetch_gaia and gaia_ids:
        fetch_gaia_catalog(
            gaia_ids,
            output_path=gaia_cache,
            allow_partial=allow_partial_fetch,
        )

    if gaia_ids:
        gaia_rows = query_gaia_by_ids(gaia_ids, cache_file=str(gaia_cache))
        enriched = merge_gaia_catalog_rows(frame, gaia_rows)
    else:
        enriched = frame.copy()

    if run_banyan:
        enriched = compute_banyan_membership(
            enriched,
            association_threshold=BANYAN_MIN_ASSOC_PROB,
        )
    after = _summary(enriched)
    return enriched, {"before": before, "after": after}


def _json_scalar(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def backup_review_db(review_db: Path, backup_path: Path | None = None) -> Path:
    """Create a transaction-consistent SQLite backup before an in-place update."""
    if backup_path is None:
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        backup_path = review_db.with_name(f"{review_db.name}.pre-gaia-banyan-{stamp}.bak")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(review_db)) as source, sqlite3.connect(str(backup_path)) as target:
        source.backup(target)
    return backup_path


def apply_review_backfill(review_db: Path, frame: pd.DataFrame) -> int:
    """Persist only Gaia/BANYAN fields for the selected candidates."""
    ensure_review_db_schema(review_db)
    updated = 0
    with db_connect(review_db, initialize_if_missing=False) as conn:
        with conn:
            for _, row in frame.iterrows():
                candidate_id = str(row["candidate_id"])
                updates = {
                    column: _json_scalar(row.get(column))
                    for column in GAIA_BACKFILL_COLUMNS
                    if column in frame.columns
                }
                updated += int(
                    replace_candidate_payload_fields(
                        conn,
                        candidate_id,
                        updates,
                        commit=False,
                    )
                )
    return updated


def _write_outputs(
    output_dir: Path,
    frame: pd.DataFrame,
    report: dict[str, object],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "gaia_banyan_backfill.parquet"
    report_path = output_dir / "gaia_banyan_backfill.report.json"
    frame.to_parquet(table_path, index=False)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return table_path, report_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Backfill Gaia DR3 astrometry and BANYAN Sigma for a review cohort."
    )
    parser.add_argument("--review-db", type=Path, required=True)
    parser.add_argument("--gaia-cache", type=Path, default=GAIA_LOCAL_CATALOG)
    parser.add_argument(
        "--cohort",
        choices=("reviewed-dippers", "all-reviewed", "all-candidates"),
        default="reviewed-dippers",
    )
    parser.add_argument(
        "--morphology",
        default="dimming_event",
        help="Review morphology used by the reviewed-dippers cohort.",
    )
    parser.add_argument("--fetch-gaia", action="store_true", help="Refresh requested IDs from Gaia DR3 TAP.")
    parser.add_argument("--no-banyan", action="store_true", help="Merge Gaia only; do not run BANYAN Sigma.")
    parser.add_argument(
        "--input-sidecar",
        type=Path,
        default=None,
        help="Reuse a previously validated gaia_banyan_backfill.parquet instead of recomputing.",
    )
    parser.add_argument("--allow-partial-fetch", action="store_true", help="Permit a cache update after failed TAP chunks.")
    parser.add_argument("--apply", action="store_true", help="Update the selected candidates in review.db.")
    parser.add_argument("--backup", type=Path, default=None, help="Explicit pre-update SQLite backup path.")
    parser.add_argument("--no-backup", action="store_true", help="Skip the automatic pre-update SQLite backup.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    review_db = args.review_db.expanduser()
    gaia_cache = args.gaia_cache.expanduser()
    if not review_db.exists():
        raise FileNotFoundError(review_db)
    if args.backup is not None and not args.apply:
        parser.error("--backup is only meaningful with --apply")
    if args.no_backup and args.backup is not None:
        parser.error("Choose either --backup or --no-backup")
    if args.input_sidecar is not None and (args.fetch_gaia or args.no_banyan):
        parser.error("--input-sidecar cannot be combined with --fetch-gaia or --no-banyan")

    frame = load_review_cohort(
        review_db,
        cohort=args.cohort,
        morphology=args.morphology,
    )
    if frame.empty:
        raise RuntimeError("The selected review cohort contains no candidates")

    if args.input_sidecar is not None:
        input_sidecar = args.input_sidecar.expanduser()
        enriched = pd.read_parquet(input_sidecar)
        if "candidate_id" not in enriched.columns:
            raise ValueError(f"Input sidecar lacks candidate_id: {input_sidecar}")
        expected_ids = set(frame["candidate_id"].astype(str))
        sidecar_ids = set(enriched["candidate_id"].astype(str))
        if sidecar_ids != expected_ids:
            raise ValueError(
                "Input sidecar candidate set does not exactly match the selected cohort: "
                f"missing={len(expected_ids - sidecar_ids)}, extra={len(sidecar_ids - expected_ids)}"
            )
        report = {
            "before": _summary(frame),
            "after": _summary(enriched),
            "input_sidecar": str(input_sidecar.resolve()),
        }
    else:
        enriched, report = backfill_frame(
            frame,
            gaia_cache=gaia_cache,
            fetch_gaia=bool(args.fetch_gaia),
            run_banyan=not bool(args.no_banyan),
            allow_partial_fetch=bool(args.allow_partial_fetch),
        )
    report.update(
        {
            "review_db": str(review_db.resolve()),
            "gaia_cache": str(gaia_cache.resolve()),
            "cohort": args.cohort,
            "morphology": args.morphology,
            "fetch_gaia": bool(args.fetch_gaia),
            "run_banyan": not bool(args.no_banyan),
            "applied": False,
        }
    )

    if args.apply:
        backup = None
        if not args.no_backup:
            backup = backup_review_db(
                review_db,
                args.backup.expanduser() if args.backup is not None else None,
            )
            print(f"Created SQLite backup: {backup}")
        updated = apply_review_backfill(review_db, enriched)
        integrity = validate_review_db_integrity(review_db)
        report.update(
            applied=True,
            candidates_updated=updated,
            backup_path=str(backup) if backup is not None else None,
            integrity=integrity,
        )

    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else review_db.parent / "gaia_banyan_backfill"
    )
    table_path, report_path = _write_outputs(output_dir, enriched, report)
    mode = "applied" if args.apply else "dry-run"
    print(f"Gaia/BANYAN backfill {mode}: {json.dumps(report['after'], sort_keys=True)}")
    print(f"Candidate sidecar: {table_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
