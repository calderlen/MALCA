#!/usr/bin/env python
"""One-off review DB dust repair with Bailer-Jones distance fallback.

This script is intentionally not a public MALCA CLI command.  It updates only
dust/extinction fields for rows that already have a usable distance, and leaves
no-distance candidates in the review DB unchanged.
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

from malca.enrichment.characterize import get_dust_extinction  # noqa: E402
from malca.ltv.cmd import fetch_bailer_jones_distances  # noqa: E402
from malca.products.feature_layers import EXTERNAL_STATS_LAYER  # noqa: E402
from malca.review.store import db_connect, replace_candidate_payload_fields  # noqa: E402


DEFAULT_DB_PATH = Path("output/runs/runs_march18_bundle_all/review/review.taxonomy_filled.db")

DUST_BASE_COLUMNS = {
    "A_v_3d",
    "ebv_3d",
    "dust_sigma",
    "dust_max_dist_kpc",
}

PROVENANCE_COLUMNS = {
    "dust_distance_pc",
    "dust_distance_source",
    "bj_r_med_photogeo",
    "bj_r_med_geo",
}


def finite_number(value: Any) -> float | None:
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def finite_positive(value: Any) -> bool:
    value = finite_number(value)
    return value is not None and value > 0


def stale_dust(row: pd.Series) -> bool:
    av = finite_number(row.get("A_v_3d"))
    ebv = finite_number(row.get("ebv_3d"))
    return av is None or av == 0.0 or ebv is None


def clean_value(value: Any) -> Any | None:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float)):
        return float(value)
    return value


def direct_distance_source(row: pd.Series) -> tuple[float | None, str | None]:
    gsp = finite_number(row.get("distance_gspphot"))
    if gsp is not None and gsp > 0:
        return gsp, "distance_gspphot"

    parallax = finite_number(row.get("parallax"))
    if parallax is not None and parallax > 0:
        return 1000.0 / parallax, "parallax"

    return None, None


def choose_bailer_jones_distance(row: pd.Series) -> tuple[float | None, str | None]:
    photogeo = finite_number(row.get("bj_r_med_photogeo"))
    if photogeo is not None and photogeo > 0:
        return photogeo, "bj_r_med_photogeo"

    geo = finite_number(row.get("bj_r_med_geo"))
    if geo is not None and geo > 0:
        return geo, "bj_r_med_geo"

    return None, None


def translated_where_clause() -> str:
    return (
        "gaia_dr2_id IS NOT NULL "
        "AND trim(cast(gaia_dr2_id AS text)) NOT IN ('', 'nan', 'None', '<NA>')"
    )


def load_translated_rows(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(
            f"SELECT * FROM candidates WHERE {translated_where_clause()}",
            conn,
        )
    finally:
        conn.close()


def assign_direct_distance(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    distances: list[float | None] = []
    sources: list[str | None] = []
    for _, row in out.iterrows():
        distance_pc, source = direct_distance_source(row)
        distances.append(distance_pc)
        sources.append(source)
    out["dust_distance_pc"] = distances
    out["dust_distance_source"] = sources
    return out


def assign_bailer_jones_distance(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    fetched = fetch_bailer_jones_distances(
        frame.copy(),
        source_id_col="source_id",
        chunk_size=1000,
        n_workers=1,
        verbose=True,
    )

    distances: list[float | None] = []
    sources: list[str | None] = []
    for _, row in fetched.iterrows():
        distance_pc, source = choose_bailer_jones_distance(row)
        distances.append(distance_pc)
        sources.append(source)

    fetched["dust_distance_pc"] = distances
    fetched["dust_distance_source"] = sources
    valid = pd.Series(distances, index=fetched.index).map(lambda value: value is not None and value > 0)
    fetched = fetched.loc[valid].copy()

    # get_dust_extinction currently consumes distance_gspphot/parallax only.
    # Use distance_gspphot as a temporary input column and do not write it back.
    fetched["distance_gspphot"] = fetched["dust_distance_pc"]
    return fetched


def dust_output_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col in DUST_BASE_COLUMNS or col in PROVENANCE_COLUMNS or col.endswith("_dered")
    ]


def external_updates_from_row(row: pd.Series, columns: list[str]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for col in columns:
        value = clean_value(row.get(col))
        if value is not None:
            updates[col] = value
    return updates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write dust updates to the review DB.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Review DB path.")
    args = parser.parse_args()

    db_path = args.db.expanduser().resolve()
    translated = load_translated_rows(db_path)
    if translated.empty:
        print("No translated Gaia rows found.")
        return

    stale = translated[translated.apply(stale_dust, axis=1)].copy()
    finite_coords = stale["ra"].notna() & stale["dec"].notna()
    stale_with_coords = stale.loc[finite_coords].copy()

    direct = assign_direct_distance(stale_with_coords)
    direct = direct.loc[direct["dust_distance_pc"].map(finite_positive)].copy()

    direct_ids = set(direct["candidate_id"].astype(str))
    needs_fallback = stale_with_coords.loc[
        ~stale_with_coords["candidate_id"].astype(str).isin(direct_ids)
    ].copy()

    print(f"translated rows: {len(translated)}")
    print(f"stale dust rows: {len(stale)}")
    print(f"rows with finite coordinates: {len(stale_with_coords)}")
    print(f"dust from Gaia distance/parallax: {len(direct)}")
    print(f"rows needing Bailer-Jones fallback: {len(needs_fallback)}")

    bailer_jones = assign_bailer_jones_distance(needs_fallback)
    bj_ids = set(bailer_jones["candidate_id"].astype(str))
    no_distance = needs_fallback.loc[
        ~needs_fallback["candidate_id"].astype(str).isin(bj_ids)
    ].copy()

    print(f"dust from Bailer-Jones fallback: {len(bailer_jones)}")
    print(f"no usable distance: {len(no_distance)}")

    run_df = pd.concat([direct, bailer_jones], ignore_index=True)
    if run_df.empty:
        print("No rows have usable distances for dustmaps3d.")
        return

    dusted = get_dust_extinction(run_df)
    av = pd.to_numeric(dusted["A_v_3d"], errors="coerce")
    print(f"computed rows with finite A_v_3d: {int(av.notna().sum())}")
    print(f"computed rows with A_v_3d > 0: {int((av > 0).sum())}")

    for candidate_id in ("618475536448",):
        example = dusted[dusted["candidate_id"].astype(str).eq(candidate_id)]
        if not example.empty:
            cols = [
                col
                for col in (
                    "candidate_id",
                    "A_v_3d",
                    "ebv_3d",
                    "dust_sigma",
                    "dust_distance_pc",
                    "dust_distance_source",
                )
                if col in example.columns
            ]
            print(f"{candidate_id} computed:", example[cols].iloc[0].to_dict())

    if not args.write:
        print("Dry run only. Rerun with --write to update the DB.")
        return

    columns = dust_output_columns(dusted)
    with db_connect(db_path) as conn:
        updated = 0
        for _, row in dusted.iterrows():
            external_updates = external_updates_from_row(row, columns)
            if not external_updates:
                continue
            if replace_candidate_payload_fields(
                conn,
                str(row["candidate_id"]),
                {EXTERNAL_STATS_LAYER: external_updates},
                commit=False,
            ):
                updated += 1
        conn.commit()

    print(f"Updated {updated} review DB rows.")


if __name__ == "__main__":
    main()
