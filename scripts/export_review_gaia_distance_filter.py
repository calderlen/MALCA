#!/usr/bin/env python
"""Export keep/drop lists for review candidates missing Gaia IDs or distances.

This is a non-destructive helper.  It does not delete review DB rows; it only
reports and optionally writes CSV lists that downstream exports can use.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from malca.catalogs.gaia_ids import parse_gaia_source_id  # noqa: E402
from malca.review.store import get_candidate_payload  # noqa: E402


DEFAULT_DB_PATH = Path("output/runs/runs_march18_bundle_all/review/review.taxonomy_filled.db")


def finite_number(value: Any) -> float | None:
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def positive_distance_from_payload(payload: dict[str, Any]) -> tuple[float | None, str | None]:
    for key in ("distance_gspphot", "bj_r_med_photogeo", "bj_r_med_geo", "distance_pc", "dist_pc"):
        value = finite_number(payload.get(key))
        if value is not None and value > 0:
            return value, key

    parallax = finite_number(payload.get("parallax"))
    if parallax is not None and parallax > 0:
        return 1000.0 / parallax, "parallax"

    return None, None


def load_review_payloads(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        ids = [
            str(row[0])
            for row in conn.execute("SELECT candidate_id FROM candidates ORDER BY candidate_id").fetchall()
        ]
        rows: list[dict[str, Any]] = []
        for candidate_id in ids:
            payload = get_candidate_payload(conn, candidate_id)
            source_id = parse_gaia_source_id(payload.get("source_id"))
            gaia_id = parse_gaia_source_id(payload.get("gaia_id"))
            distance_pc, distance_source = positive_distance_from_payload(payload)
            failed_missing_gaia_id = source_id is None and gaia_id is None
            failed_missing_distance = distance_pc is None
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "source_id": source_id,
                    "gaia_id": gaia_id,
                    "gaia_dr2_id": payload.get("gaia_dr2_id"),
                    "distance_pc": distance_pc,
                    "distance_source": distance_source,
                    "failed_missing_gaia_id": failed_missing_gaia_id,
                    "failed_missing_distance": failed_missing_distance,
                    "keep_gaia_distance": not failed_missing_gaia_id and not failed_missing_distance,
                }
            )
        return pd.DataFrame(rows)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Review DB path.")
    parser.add_argument("--write", action="store_true", help="Write keep/drop CSVs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for CSVs. Defaults to <review-db-parent>/gaia_distance_filter.",
    )
    args = parser.parse_args()

    db_path = args.db.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else db_path.parent / "gaia_distance_filter"
    )

    frame = load_review_payloads(db_path)
    if frame.empty:
        print("No review candidates found.")
        return

    keep = frame[frame["keep_gaia_distance"]].copy()
    drop = frame[~frame["keep_gaia_distance"]].copy()

    print(f"total candidates: {len(frame)}")
    print(f"keep with Gaia ID and usable distance: {len(keep)}")
    print(f"drop/exclude missing Gaia ID: {int(frame['failed_missing_gaia_id'].sum())}")
    print(f"drop/exclude missing usable distance: {int(frame['failed_missing_distance'].sum())}")
    print(f"drop/exclude total unique rows: {len(drop)}")
    print("distance sources:")
    print(frame["distance_source"].fillna("missing").value_counts(dropna=False).to_string())

    if not args.write:
        print("Dry run only. Rerun with --write to write keep/drop CSVs.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "gaia_distance_filter_all.csv", index=False)
    keep.to_csv(output_dir / "gaia_distance_filter_keep.csv", index=False)
    drop.to_csv(output_dir / "gaia_distance_filter_drop.csv", index=False)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
