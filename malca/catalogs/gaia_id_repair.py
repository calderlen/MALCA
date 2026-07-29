"""Repair stale Gaia DR2 IDs in review DBs and candidate exports."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from malca.enrichment.characterize import query_gaia_by_ids
from malca.config import GAIA_ID_MAPPING_CACHE, GAIA_LOCAL_CATALOG
from malca.products.feature_layers import (
    EXTERNAL_STATS_LAYER,
    _parse_layer_value,
    is_layer_first_frame,
    to_layer_first_frame,
    with_feature_columns,
)
from malca.catalogs.gaia_fetch import fetch_gaia_catalog
from malca.catalogs.gaia_ids import canonicalize_gaia_ids_in_frame, normalize_gaia_source_ids, parse_gaia_source_id
from malca.review.store import db_connect, ensure_review_db_schema, get_candidate_payload
from malca.io.table_io import read_feature_table, write_feature_table


GAIA_REPAIR_COLUMNS = (
    "source_id",
    "gaia_id",
    "gaia_dr2_id",
    "gaia_id_release",
    "gaia_id_mapping_status",
    "dr2_dr3_angular_distance_mas",
    "dr2_dr3_magnitude_difference",
    "ra",
    "dec",
    "parallax",
    "parallax_error",
    "pmra",
    "pmdec",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "bp_rp",
    "distance_gspphot",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "ruwe",
)


def _connect_review_db_for_repair(db_path: Path, *, write: bool) -> sqlite3.Connection:
    """Open review DBs read-only for dry-runs and migratable for writes."""
    if write:
        ensure_review_db_schema(db_path)
        return db_connect(db_path)

    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _missing_like(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip().lower() in {"", "nan", "none", "<na>", "null"}


def _merge_local_gaia_enrichment(
    frame: pd.DataFrame,
    *,
    gaia_cache_path: Path,
    fetch_gaia: bool,
) -> pd.DataFrame:
    if frame.empty or "source_id" not in frame.columns:
        return frame

    ids = normalize_gaia_source_ids(frame["source_id"].dropna().tolist())
    if not ids:
        return frame

    if fetch_gaia:
        fetch_gaia_catalog(ids, output_path=gaia_cache_path)

    try:
        gaia_df = query_gaia_by_ids(ids, cache_file=str(gaia_cache_path))
    except Exception as exc:
        print(f"Warning: could not merge local Gaia enrichment: {exc}")
        return frame

    if gaia_df.empty or "source_id" not in gaia_df.columns:
        return frame

    gaia_df = gaia_df.copy()
    gaia_df["source_id"] = gaia_df["source_id"].map(parse_gaia_source_id)
    gaia_df = gaia_df.dropna(subset=["source_id"]).drop_duplicates(subset=["source_id"], keep="last")
    lookup = gaia_df.set_index("source_id")

    out = frame.copy()
    source_ids = out["source_id"].map(parse_gaia_source_id)
    for column in lookup.columns:
        values = source_ids.map(lookup[column])
        if column in {"source_id"}:
            continue
        if column in out.columns:
            missing = out[column].map(_missing_like)
            out.loc[missing & values.notna(), column] = values.loc[missing & values.notna()]
        else:
            out[column] = values
    return out


def repair_gaia_ids_frame(
    df: pd.DataFrame,
    *,
    gaia_cache_path: Path = GAIA_LOCAL_CATALOG,
    mapping_cache_path: Path = GAIA_ID_MAPPING_CACHE,
    write_mapping_cache: bool = True,
    query_tap: bool = True,
    fetch_gaia: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Canonicalize Gaia IDs in a candidate frame and merge local Gaia columns."""
    if df.empty:
        return df, {"rows": 0, "translated": 0, "changed": 0}

    original_layer_first = is_layer_first_frame(df)
    work = with_feature_columns(df, ["source_id", "gaia_id", *GAIA_REPAIR_COLUMNS])
    tracked_cols = list(dict.fromkeys([col for col in GAIA_REPAIR_COLUMNS if col in work.columns]))
    before = work.reindex(columns=tracked_cols).copy()

    repaired = work.copy()
    gaia_ids = work["gaia_id"].map(parse_gaia_source_id) if "gaia_id" in work.columns else pd.Series(pd.NA, index=work.index)
    if "source_id" in work.columns:
        source_ids = work["source_id"].map(parse_gaia_source_id)
        repair_mask = gaia_ids.notna() & source_ids.isna()
    else:
        repair_mask = gaia_ids.notna()

    if repair_mask.any():
        repaired_subset = canonicalize_gaia_ids_in_frame(
            work.loc[repair_mask].copy(),
            gaia_cache_path=gaia_cache_path,
            mapping_cache_path=mapping_cache_path,
            write_mapping_cache=write_mapping_cache,
            query_tap=query_tap,
        )
        repaired_subset = _merge_local_gaia_enrichment(
            repaired_subset,
            gaia_cache_path=gaia_cache_path,
            fetch_gaia=fetch_gaia,
        )
        for column in repaired_subset.columns:
            if column not in repaired.columns:
                repaired[column] = pd.NA
            repaired.loc[repair_mask, column] = repaired_subset[column]

    translated = 0
    if "gaia_id_mapping_status" in repaired.columns:
        translated = int(repaired["gaia_id_mapping_status"].astype(str).eq("dr2_translated").sum())

    changed_mask = pd.Series(False, index=repaired.index)
    for column in tracked_cols:
        old = before[column].astype("string").fillna("")
        if column in repaired.columns:
            new = repaired[column].astype("string").fillna("")
        else:
            new = pd.Series("", index=repaired.index, dtype="string")
        changed_mask |= old.ne(new)
    changed = int(changed_mask.sum())

    if original_layer_first:
        repaired = to_layer_first_frame(repaired)

    return repaired, {"rows": int(len(df)), "translated": translated, "changed": changed}


def _read_export(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return read_feature_table(path)
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str)
    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    rows.append(json.loads(text))
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported candidate export format: {path}")


def _write_export(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix in {".parquet", ".pq"}:
        write_feature_table(df, path)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("w", encoding="utf-8") as handle:
            for record in df.to_dict("records"):
                handle.write(json.dumps(record, default=str) + "\n")
        return
    raise ValueError(f"Unsupported candidate export format: {path}")


def _update_payload_json(payload_json: str | None, updates: dict[str, Any]) -> str:
    try:
        payload = json.loads(payload_json) if payload_json else {}
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    external = _parse_layer_value(payload.get(EXTERNAL_STATS_LAYER))
    for key, value in updates.items():
        if _missing_like(value):
            continue
        if key in {"source_id", "gaia_id"}:
            payload[key] = value
        external[key] = value
    if external:
        payload[EXTERNAL_STATS_LAYER] = json.dumps(external, sort_keys=True, separators=(",", ":"), default=str)
    return json.dumps(payload, default=str)


def repair_review_db(
    db_path: Path,
    *,
    write: bool,
    gaia_cache_path: Path = GAIA_LOCAL_CATALOG,
    mapping_cache_path: Path = GAIA_ID_MAPPING_CACHE,
    write_mapping_cache: bool = True,
    query_tap: bool = True,
    fetch_gaia: bool = False,
) -> dict[str, int]:
    conn = _connect_review_db_for_repair(db_path, write=write)
    try:
        rows = conn.execute("SELECT candidate_id, payload_json FROM candidates").fetchall()
        if not rows:
            return {"rows": 0, "translated": 0, "changed": 0}

        records = []
        payload_json_by_id: dict[str, str | None] = {}
        for candidate_id, payload_json in rows:
            record = get_candidate_payload(conn, str(candidate_id))
            record["candidate_id"] = str(candidate_id)
            records.append(record)
            payload_json_by_id[str(candidate_id)] = payload_json

        repaired, stats = repair_gaia_ids_frame(
            pd.DataFrame(records),
            gaia_cache_path=gaia_cache_path,
            mapping_cache_path=mapping_cache_path,
            write_mapping_cache=write_mapping_cache,
            query_tap=query_tap,
            fetch_gaia=fetch_gaia,
        )
        if not write:
            return stats

        actual_cols = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(candidates)").fetchall()
        }
        sql_update_cols = [col for col in GAIA_REPAIR_COLUMNS if col in actual_cols and col in repaired.columns]
        payload_cols = [col for col in GAIA_REPAIR_COLUMNS if col in repaired.columns]

        with conn:
            for _, row in repaired.iterrows():
                candidate_id = str(row.get("candidate_id"))
                updates = {col: row.get(col) for col in payload_cols}
                payload_json = _update_payload_json(payload_json_by_id.get(candidate_id), updates)

                assignments = ["payload_json=?"]
                values: list[Any] = [payload_json]
                for col in sql_update_cols:
                    value = row.get(col)
                    values.append(None if _missing_like(value) else value)
                    assignments.append(f"{col}=?")
                values.append(candidate_id)
                conn.execute(
                    f"UPDATE candidates SET {', '.join(assignments)} WHERE candidate_id=?",
                    values,
                )
        return stats
    finally:
        conn.close()


def repair_export_file(
    path: Path,
    *,
    output_path: Path | None,
    write: bool,
    gaia_cache_path: Path = GAIA_LOCAL_CATALOG,
    mapping_cache_path: Path = GAIA_ID_MAPPING_CACHE,
    write_mapping_cache: bool = True,
    query_tap: bool = True,
    fetch_gaia: bool = False,
) -> dict[str, int]:
    df = _read_export(path)
    repaired, stats = repair_gaia_ids_frame(
        df,
        gaia_cache_path=gaia_cache_path,
        mapping_cache_path=mapping_cache_path,
        write_mapping_cache=write_mapping_cache,
        query_tap=query_tap,
        fetch_gaia=fetch_gaia,
    )
    if write or output_path is not None:
        target = output_path if output_path is not None else path
        _write_export(repaired, target)
    return stats


def _is_sqlite_review_db(path: Path) -> bool:
    if path.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
        return False
    try:
        conn = sqlite3.connect(str(path))
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='candidates'"
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except Exception:
        return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Repair Gaia DR2 IDs in MALCA review DBs/candidate exports.")
    parser.add_argument("paths", nargs="+", type=Path, help="Review DB or candidate export path(s).")
    parser.add_argument("--write", action="store_true", help="Write changes in place for DBs and exports.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Write repaired export copies here.")
    parser.add_argument("--gaia-cache", type=Path, default=GAIA_LOCAL_CATALOG, help="Local Gaia DR3 cache path.")
    parser.add_argument("--mapping-cache", type=Path, default=GAIA_ID_MAPPING_CACHE, help="DR2->DR3 mapping cache path.")
    parser.add_argument("--no-tap", action="store_true", help="Use only existing caches; do not query Gaia TAP.")
    parser.add_argument("--fetch-gaia", action="store_true", help="Fetch missing Gaia DR3 rows before merging enrichment.")
    args = parser.parse_args(argv)

    if args.fetch_gaia and not (args.write or args.output_dir is not None):
        parser.error("--fetch-gaia writes the Gaia cache; use it only with --write or --output-dir.")

    query_tap = not bool(args.no_tap)
    write_mapping_cache = bool(args.write or args.output_dir is not None)
    total = {"rows": 0, "translated": 0, "changed": 0}
    for path in args.paths:
        path = path.expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        if _is_sqlite_review_db(path):
            stats = repair_review_db(
                path,
                write=args.write,
                gaia_cache_path=args.gaia_cache.expanduser(),
                mapping_cache_path=args.mapping_cache.expanduser(),
                write_mapping_cache=write_mapping_cache,
                query_tap=query_tap,
                fetch_gaia=args.fetch_gaia,
            )
            action = "updated" if args.write else "would update"
            print(f"{path}: {action} {stats['changed']} row(s), translated {stats['translated']} DR2 ID(s)")
        else:
            output_path = None
            if args.output_dir is not None:
                output_path = args.output_dir.expanduser() / path.name
            stats = repair_export_file(
                path,
                output_path=output_path,
                write=args.write,
                gaia_cache_path=args.gaia_cache.expanduser(),
                mapping_cache_path=args.mapping_cache.expanduser(),
                write_mapping_cache=write_mapping_cache,
                query_tap=query_tap,
                fetch_gaia=args.fetch_gaia,
            )
            if args.write or output_path is not None:
                target = output_path if output_path is not None else path
                print(f"{path}: wrote {target} with {stats['changed']} changed row(s), translated {stats['translated']} DR2 ID(s)")
            else:
                print(f"{path}: would update {stats['changed']} row(s), translated {stats['translated']} DR2 ID(s)")

        for key in total:
            total[key] += stats[key]

    mode = "wrote" if args.write or args.output_dir is not None else "dry run"
    print(
        f"Gaia ID repair {mode}: scanned {total['rows']} row(s), "
        f"changed {total['changed']}, translated {total['translated']} DR2 ID(s)."
    )


if __name__ == "__main__":
    main()
