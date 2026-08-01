"""Strict validation gate and input export for future R24 model comparison.

This module does not pretend that a local R24 grid or compatible ``sedfitter``
installation exists.  It prepares the exact, explicitly validated infrared
measurement set that an R24 backend is allowed to consume.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from malca.io.table_io import write_parquet_table
from malca.review.sed_storage import (
    ensure_sed_storage_schema,
    load_r24_ready_sed_measurements,
    load_sed_measurements,
    store_sed_measurement_validations,
)
from malca.review.store import db_connect


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malca sed-r24-inputs",
        description=(
            "Record explicit measurement validation decisions and export only "
            "R24-eligible infrared SED points."
        ),
    )
    parser.add_argument("review_db", type=Path, help="Review SQLite database")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Validated R24-input Parquet (default: beside the review DB).",
    )
    parser.add_argument(
        "--candidate-id",
        default=None,
        help="Limit the export to one candidate.",
    )
    parser.add_argument(
        "--accept",
        action="append",
        default=[],
        metavar="MEASUREMENT_ID",
        help=(
            "Explicitly accept one measurement for R24. Repeat as needed; "
            "--validator is required."
        ),
    )
    parser.add_argument(
        "--reject",
        action="append",
        default=[],
        metavar="MEASUREMENT_ID",
        help="Explicitly reject one measurement. Repeat as needed.",
    )
    parser.add_argument(
        "--validation-version",
        default="manual-r24-v1",
        help=(
            "Immutable validation recipe/version (default: manual-r24-v1). "
            "Use a new version to supersede an earlier decision."
        ),
    )
    parser.add_argument(
        "--validator",
        default=None,
        help="Person or process responsible for --accept/--reject decisions.",
    )
    parser.add_argument(
        "--validation-method",
        default="image-and-counterpart-review",
        help="Validation method recorded in provenance.",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Notes applied to validation records created in this invocation.",
    )
    parser.add_argument(
        "--min-wavelength-micron",
        type=float,
        default=1.0,
        help="Minimum wavelength exported to the R24 handoff (default: 1 micron).",
    )
    return parser


def _provenance_dict(value: object) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def prepare_r24_handoff(rows: pd.DataFrame) -> pd.DataFrame:
    """Add aperture/model-input fields without weakening validation semantics."""

    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        return frame
    provenance = frame["measurement_provenance_json"].map(_provenance_dict)
    frame["wavelength_micron"] = (
        pd.to_numeric(frame["plot_lambda_angstrom"], errors="coerce") / 10_000.0
    )
    frame["aperture_radius_arcsec"] = provenance.map(
        lambda item: item.get("aperture_radius_arcsec")
    )
    frame["beam_fwhm_arcsec"] = provenance.map(
        lambda item: item.get("beam_fwhm_arcsec")
    )
    frame["upper_limit_sigma"] = frame["is_upper_limit"].map(
        lambda value: 3.0 if bool(value) else None
    )
    frame["measurement_method"] = provenance.map(
        lambda item: item.get("measurement_version")
        or item.get("catalog")
        or "validated_catalog"
    )
    frame["r24_input_status"] = "validated"
    return frame


def _validate_measurement_ids(conn, args: argparse.Namespace) -> int:
    accepted = [str(value) for value in args.accept]
    rejected = [str(value) for value in args.reject]
    overlap = sorted(set(accepted) & set(rejected))
    if overlap:
        raise ValueError(
            "The same measurement cannot be accepted and rejected together: "
            + ", ".join(overlap)
        )
    requested = accepted + rejected
    if not requested:
        return 0
    if not str(args.validator or "").strip():
        raise ValueError("--validator is required when recording validation decisions")
    known = load_sed_measurements(conn, measurement_ids=requested)
    known_ids = set(known["measurement_id"].astype(str))
    missing = sorted(set(requested) - known_ids)
    if missing:
        raise KeyError("Unknown SED measurement ID(s): " + ", ".join(missing))
    created_at = datetime.now(timezone.utc).isoformat()
    decisions = []
    for measurement_id in accepted:
        decisions.append(
            {
                "measurement_id": measurement_id,
                "validation_version": str(args.validation_version),
                "validation_status": "accepted",
                "r24_eligible": True,
                "validator": str(args.validator),
                "validation_method": str(args.validation_method),
                "notes": args.notes,
                "provenance_json": {
                    "command": "malca sed-r24-inputs",
                    "decision": "explicit_accept",
                },
                "created_at": created_at,
            }
        )
    for measurement_id in rejected:
        decisions.append(
            {
                "measurement_id": measurement_id,
                "validation_version": str(args.validation_version),
                "validation_status": "rejected",
                "r24_eligible": False,
                "validator": str(args.validator),
                "validation_method": str(args.validation_method),
                "notes": args.notes,
                "provenance_json": {
                    "command": "malca sed-r24-inputs",
                    "decision": "explicit_reject",
                },
                "created_at": created_at,
            }
        )
    return store_sed_measurement_validations(conn, decisions)


def run(args: argparse.Namespace) -> Path:
    review_db = args.review_db.expanduser()
    output_path = (
        args.output.expanduser()
        if args.output
        else review_db.with_name(f"{review_db.stem}_sed_r24_validated_inputs.parquet")
    )
    min_wavelength_angstrom = max(float(args.min_wavelength_micron), 0.0) * 10_000.0
    with closing(db_connect(review_db)) as conn:
        ensure_sed_storage_schema(conn)
        conn.commit()
        n_validations = _validate_measurement_ids(conn, args)
        ready = load_r24_ready_sed_measurements(
            conn,
            candidate_id=args.candidate_id,
            min_wavelength_angstrom=min_wavelength_angstrom,
        )
    handoff = prepare_r24_handoff(ready)
    write_parquet_table(handoff, output_path)
    if n_validations:
        print(f"Stored {n_validations} immutable measurement validation decision(s)")
    print(
        f"Saved {len(handoff)} explicitly validated R24 input point(s) "
        f"to {output_path}"
    )
    if not handoff.empty:
        summary = (
            handoff.groupby("candidate_id")
            .agg(
                n_validated_points=("measurement_id", "size"),
                longest_wavelength_micron=("wavelength_micron", "max"),
                n_upper_limits=("is_upper_limit", "sum"),
            )
            .sort_index()
        )
        print(summary.to_string())
    return output_path


def main(argv: list[str] | None = None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
