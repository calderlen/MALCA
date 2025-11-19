from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from reproduce_candidates import brayden_candidates

MANIFEST_PATH = Path("lc_manifest.csv")

# First five IDs were manually supplied before; keep them grouped explicitly.
PRIMARY_TARGET_IDS = [
    "J183153-284827",
    "J070519+061219",
    "J081523-385923",
    "J085816-430955",
    "J114712-621037",
]


def all_candidate_ids() -> list[str]:
    """Return every source_id from brayden_candidates as strings."""
    ids: list[str] = []
    seen = set()
    for entry in brayden_candidates:
        source = str(entry.get("source_id", "")).strip()
        if not source or source in seen:
            continue
        seen.add(source)
        ids.append(source)
    return ids


def pick_id_column(df: pd.DataFrame) -> str:
    """Find an ID column, preferring 'source_id' when present."""
    preferred = ["source_id", "asas_sn_id", "id"]
    for name in preferred:
        if name in df.columns:
            return name
    matches = [c for c in df.columns if "id" in c.lower()]
    if matches:
        return matches[0]
    raise ValueError(
        f"Could not determine an ID column. Available columns: {', '.join(df.columns)}"
    )


def locate_targets(
    df: pd.DataFrame,
    ids: Iterable[str],
    *,
    id_col: str,
) -> pd.DataFrame:
    ids_set = {str(i).strip() for i in ids}
    return df[df[id_col].astype(str).str.strip().isin(ids_set)].copy()


def print_matches(label: str, matches: pd.DataFrame, *, id_col: str) -> None:
    print(f"\n=== {label} ===")
    if matches.empty:
        print("No matches found.")
        return
    print(f"Found {len(matches)} matches:")
    for _, row in matches.iterrows():
        star_id = row[id_col]
        mag_bin = row.get("mag_bin", "N/A")
        lc_dir = row.get("lc_dir", "N/A")
        dat_path = row.get("dat_path", "N/A")
        print(f"- {star_id}")
        print(f"    mag_bin: {mag_bin}")
        print(f"    lc_dir : {lc_dir}")
        print(f"    dat    : {dat_path}")


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)
    id_col = pick_id_column(df)

    print(f"Loaded {len(df)} manifest rows from {MANIFEST_PATH} (ID column: '{id_col}')")

    all_ids = all_candidate_ids()
    primary_set = set(PRIMARY_TARGET_IDS)
    remaining_ids = [i for i in all_ids if i not in primary_set]

    primary_matches = locate_targets(df, PRIMARY_TARGET_IDS, id_col=id_col)
    remaining_matches = locate_targets(df, remaining_ids, id_col=id_col)

    print_matches("Primary 5 (pre-listed)", primary_matches, id_col=id_col)
    print_matches("Remaining brayden_candidates", remaining_matches, id_col=id_col)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}")
