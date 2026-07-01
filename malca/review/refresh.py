from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from malca.products.feature_layers import FEATURE_LAYER_COLUMNS, feature_mapping_get, parse_layer_value
from malca.review.stats_merge import merge_stats_summary_into_payload
from malca.review.sync import auto_export_review_bundle
from malca.review.store import (
    _CANDIDATE_COLUMNS,
    _COL_NAMES,
    _as_bool,
    _flatten_review_payload,
    _is_payload_missing,
    _normalize_large_integer_like_id,
    _review_payload_extra,
    _to_float,
    _utc_now,
    db_connect,
    detect_run_directory_files,
    get_candidate_payload,
    load_candidates_file,
    replace_candidate_payload_fields,
)
from malca.core.stats import compute_stats


CORE_STATS_KEYS = {
    "baseline_mag",
    "cadence_median_days",
    "n_cameras",
    "n_points",
}


def _load_scope_candidate_ids(candidate_source: Path) -> list[str]:
    df = load_candidates_file(candidate_source)
    if df.empty:
        return []

    for col in ("candidate_id", "asas_sn_id"):
        if col not in df.columns:
            continue
        values = df[col].dropna().astype(str).str.strip()
        values = values[values != ""]
        if values.empty:
            continue
        return list(dict.fromkeys(values.tolist()))

    raise ValueError(
        f"Candidate scope file must contain candidate_id or asas_sn_id: {candidate_source}"
    )


def _resolve_scope_ids_to_candidate_ids(
    scope_ids: list[str],
    db_rows: list[tuple[object, object, object]],
) -> tuple[list[str], list[str]]:
    candidate_ids: set[str] = set()
    asas_to_candidate: dict[str, str] = {}
    ambiguous_asas_ids: set[str] = set()

    for raw_candidate_id, raw_asas_sn_id, _payload_json in db_rows:
        candidate_id = str(raw_candidate_id).strip()
        if candidate_id:
            candidate_ids.add(candidate_id)

        asas_sn_id = "" if raw_asas_sn_id is None else str(raw_asas_sn_id).strip()
        if not asas_sn_id:
            continue
        existing = asas_to_candidate.get(asas_sn_id)
        if existing is None:
            asas_to_candidate[asas_sn_id] = candidate_id
        elif existing != candidate_id:
            ambiguous_asas_ids.add(asas_sn_id)

    for asas_sn_id in ambiguous_asas_ids:
        asas_to_candidate.pop(asas_sn_id, None)

    matched_ids: list[str] = []
    missing_from_db: list[str] = []
    seen: set[str] = set()

    for scope_id in scope_ids:
        text = str(scope_id).strip()
        if not text:
            continue
        candidate_id = None
        if text in candidate_ids:
            candidate_id = text
        else:
            candidate_id = asas_to_candidate.get(text)

        if candidate_id is None:
            missing_from_db.append(text)
            continue
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        matched_ids.append(candidate_id)

    return matched_ids, missing_from_db


def _resolve_lightcurve_path(payload: dict, run_dir: Path) -> Path | None:
    bundle_dir = run_dir / "bundle_assets" / "lightcurves"
    candidate_names: list[str] = []

    for key in ("lc_path",):
        raw_path = feature_mapping_get(payload, key)
        if not raw_path:
            continue
        candidate = Path(str(raw_path)).expanduser()
        candidate_names.append(candidate.name)
        if candidate.suffix in (".dat", ".dat2", ".dat3"):
            candidate_names.append(candidate.with_suffix(".raw2").name)
        elif candidate.suffix == ".raw2":
            for ext in (".dat3", ".dat2", ".dat"):
                candidate_names.append(candidate.with_suffix(ext).name)
        if candidate.exists():
            return candidate

    for key in ("candidate_id", "asas_sn_id"):
        raw = feature_mapping_get(payload, key)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        candidate_names.extend([f"{text}.dat3", f"{text}.raw2", f"{text}.dat2", f"{text}.dat", f"{text}.csv"])

    if bundle_dir.exists():
        seen: set[str] = set()
        for name in candidate_names:
            if not name or name in seen:
                continue
            seen.add(name)
            candidate = bundle_dir / name
            if candidate.exists():
                return candidate

    return None


def _build_stats_updates(lc_path: Path, *, compute_ls: bool) -> dict[str, object]:
    file_ext = lc_path.suffix[1:] if lc_path.suffix.startswith(".") else None
    _df, summary = compute_stats(
        lc_path.stem,
        str(lc_path.parent),
        compute_ls=compute_ls,
        file_ext=file_ext,
    )
    updates: dict[str, object] = {"lc_path": str(lc_path)}
    merge_stats_summary_into_payload(updates, summary)
    return updates


def refresh_review_stats_from_run(
    run_dir: Path,
    db_path: Path,
    *,
    candidate_source: Path | None = None,
    compute_ls: bool = True,
    limit: int | None = None,
    verbose: bool = False,
) -> dict[str, object]:
    run_dir = Path(run_dir).expanduser().resolve()
    db_path = Path(db_path).expanduser().resolve()

    if candidate_source is None:
        detected = detect_run_directory_files(run_dir)
        detected_source = detected.get("candidates")
        if not isinstance(detected_source, Path):
            raise FileNotFoundError(
                f"Could not detect a candidate table under {run_dir}. Pass --candidate-source explicitly."
            )
        candidate_source = detected_source
    else:
        candidate_source = Path(candidate_source).expanduser().resolve()

    scope_ids = _load_scope_candidate_ids(candidate_source)
    if limit is not None:
        scope_ids = scope_ids[: max(int(limit), 0)]

    if not scope_ids:
        return {
            "candidate_source": str(candidate_source),
            "scoped_candidates": 0,
            "matched_db_rows": 0,
            "refreshed": 0,
            "unresolved": [],
            "failed": [],
            "missing_from_db": [],
        }

    with db_connect(db_path) as conn:
        rows = conn.execute("SELECT candidate_id, asas_sn_id, payload_json FROM candidates").fetchall()
        payload_by_id: dict[str, dict] = {}
        for candidate_id, asas_sn_id, _payload_json in rows:
            cid = str(candidate_id)
            payload = get_candidate_payload(conn, cid)
            payload["candidate_id"] = cid
            if asas_sn_id not in (None, "") and not payload.get("asas_sn_id"):
                payload["asas_sn_id"] = str(asas_sn_id)
            payload_by_id[cid] = payload

        table_cols = {
            str(info[1])
            for info in conn.execute("PRAGMA table_info(candidates)").fetchall()
        }
        clear_base = {col for col in table_cols if col.startswith("stats_")}
        clear_base.update(CORE_STATS_KEYS)

        matched_ids, missing_from_db = _resolve_scope_ids_to_candidate_ids(scope_ids, rows)

        refreshed = 0
        unresolved: list[str] = []
        failed: list[dict[str, str]] = []

        for idx, candidate_id in enumerate(matched_ids, start=1):
            payload = dict(payload_by_id[candidate_id])
            clear_keys = set(clear_base)
            for layer in FEATURE_LAYER_COLUMNS:
                clear_keys.update(
                    key
                    for key in parse_layer_value(payload.get(layer))
                    if key.startswith("stats_") or key in CORE_STATS_KEYS
                )

            lc_path = _resolve_lightcurve_path(payload, run_dir)
            if lc_path is None:
                unresolved.append(candidate_id)
                if verbose:
                    print(f"[{idx}/{len(matched_ids)}] unresolved light curve for {candidate_id}")
                continue

            try:
                updates = _build_stats_updates(lc_path, compute_ls=compute_ls)
                replace_candidate_payload_fields(
                    conn,
                    candidate_id,
                    updates,
                    clear_keys=clear_keys,
                    commit=False,
                )
                refreshed += 1
                if verbose and ((idx % 50 == 0) or (idx == len(matched_ids))):
                    print(f"[{idx}/{len(matched_ids)}] refreshed {refreshed} candidates")
            except Exception as exc:
                failed.append({"candidate_id": candidate_id, "error": str(exc)})
                if verbose:
                    print(f"[{idx}/{len(matched_ids)}] failed {candidate_id}: {exc}")

        conn.commit()

    return {
        "candidate_source": str(candidate_source),
        "scoped_candidates": len(scope_ids),
        "matched_db_rows": len(matched_ids),
        "refreshed": refreshed,
        "unresolved": unresolved,
        "failed": failed,
        "missing_from_db": missing_from_db,
    }


def _candidate_insert_rows_from_db_rows(
    rows: list[tuple[object, ...]],
    row_columns: list[str],
) -> list[tuple[object, ...]]:
    payload_rows: list[tuple[object, ...]] = []
    for row in rows:
        raw = dict(zip(row_columns, row))
        candidate_id = raw.get("candidate_id")
        source_path = raw.get("source_path")
        imported_at = raw.get("imported_at")
        try:
            payload = json.loads(raw.get("payload_json")) if raw.get("payload_json") else {}
        except Exception:
            payload = {}
        payload = _flatten_review_payload(payload if isinstance(payload, dict) else {})

        for col in _COL_NAMES:
            value = raw.get(col)
            if not _is_payload_missing(value):
                payload[col] = value

        payload["candidate_id"] = str(payload.get("candidate_id") or candidate_id)
        if "gaia_id" in payload:
            payload["gaia_id"] = _normalize_large_integer_like_id(payload.get("gaia_id"))
        if "source_id" in payload and payload.get("source_id") is not None:
            payload["source_id"] = _normalize_large_integer_like_id(payload.get("source_id"))

        vals: list[object] = [str(candidate_id), source_path]
        for col, _dtype, etype in _CANDIDATE_COLUMNS:
            raw = payload.get(col)
            if etype == "bool":
                vals.append(int(_as_bool(raw)) if raw is not None else None)
            elif etype == "float":
                vals.append(_to_float(raw))
            else:
                vals.append(str(raw) if raw is not None else None)
        vals.append(json.dumps(_review_payload_extra(payload), default=str))
        vals.append(imported_at or _utc_now())
        payload_rows.append(tuple(vals))
    return payload_rows


def rebuild_review_db(
    source_db: Path,
    target_db: Path,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    source_db = Path(source_db).expanduser().resolve()
    target_db = Path(target_db).expanduser().resolve()

    if source_db == target_db:
        raise ValueError("Target DB path must differ from source DB path.")
    if target_db.exists() and not overwrite:
        raise FileExistsError(f"Target DB already exists: {target_db}")
    if target_db.exists():
        target_db.unlink()

    with db_connect(source_db) as src_conn:
        source_candidate_cols = [
            col
            for col in ["candidate_id", "source_path", *_COL_NAMES, "payload_json", "imported_at"]
            if col in {str(row[1]) for row in src_conn.execute("PRAGMA table_info(candidates)").fetchall()}
        ]
        candidate_rows = src_conn.execute(
            f"SELECT {', '.join(source_candidate_cols)} FROM candidates"
        ).fetchall()
        reviews = pd.read_sql_query("SELECT * FROM reviews", src_conn)
        review_history = pd.read_sql_query("SELECT * FROM review_history", src_conn)
        app_state = pd.read_sql_query("SELECT * FROM app_state", src_conn)

    with db_connect(target_db) as dst_conn:
        all_col_names = ["candidate_id", "source_path"] + _COL_NAMES + ["payload_json", "imported_at"]
        placeholders = ", ".join(["?"] * len(all_col_names))
        insert_cols = ", ".join(all_col_names)
        candidate_payload_rows = _candidate_insert_rows_from_db_rows(candidate_rows, source_candidate_cols)
        if candidate_payload_rows:
            dst_conn.executemany(
                f"INSERT INTO candidates ({insert_cols}) VALUES ({placeholders})",
                candidate_payload_rows,
            )
        if not reviews.empty:
            reviews.to_sql("reviews", dst_conn, if_exists="append", index=False)
        if not review_history.empty:
            review_history.to_sql("review_history", dst_conn, if_exists="append", index=False)
        if not app_state.empty:
            dst_conn.executemany(
                """
                INSERT INTO app_state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=excluded.updated_at
                """,
                [
                    (str(row["key"]), str(row["value"]), str(row["updated_at"] or _utc_now()))
                    for _, row in app_state.iterrows()
                ],
            )
        dst_conn.commit()

    return {
        "candidates": len(candidate_rows),
        "reviews": len(reviews),
        "review_history": len(review_history),
        "app_state": len(app_state),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh review DB stats from local or bundled light curves."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory or imported bundle directory")
    parser.add_argument("--review-db", required=True, help="Review SQLite DB to refresh in place")
    parser.add_argument(
        "--candidate-source",
        default=None,
        help="Optional Parquet used to scope candidate IDs; defaults to best run results file",
    )
    parser.add_argument(
        "--no-compute-ls",
        action="store_true",
        help="Skip Lomb-Scargle recomputation while refreshing stats",
    )
    parser.add_argument("--limit", type=int, default=None, help="Refresh only the first N scoped candidates")
    parser.add_argument(
        "--output-review-db",
        default=None,
        help="Optional path for a rebuilt DB with only the current candidate schema",
    )
    parser.add_argument(
        "--overwrite-rebuild-db",
        action="store_true",
        help="Allow overwriting an existing --output-review-db target",
    )
    parser.add_argument(
        "--no-review-sync",
        dest="review_sync_enabled",
        action="store_false",
        help="Skip automatic reviews/*.jsonl export after refresh",
    )
    parser.add_argument(
        "--review-sync-dir",
        type=Path,
        default=Path("reviews"),
        help="Directory for automatic Git-trackable review export (default: reviews)",
    )
    parser.add_argument(
        "--review-sync-hash-assets",
        action="store_true",
        help="Include SHA-256 hashes for resolved assets in automatic review export",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-batch progress")
    parser.set_defaults(review_sync_enabled=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    db_path = Path(args.review_db).expanduser().resolve()
    candidate_source = None
    if args.candidate_source:
        candidate_source = Path(args.candidate_source).expanduser().resolve()

    print(f"Refreshing review stats in {db_path}")
    print(f"  run dir: {run_dir}")
    if candidate_source is not None:
        print(f"  scope file: {candidate_source}")

    result = refresh_review_stats_from_run(
        run_dir,
        db_path,
        candidate_source=candidate_source,
        compute_ls=not bool(args.no_compute_ls),
        limit=args.limit,
        verbose=bool(args.verbose),
    )
    print(
        "Refreshed {refreshed}/{matched_db_rows} scoped DB rows "
        "({scoped_candidates} candidate IDs in scope).".format(**result)
    )
    if result["missing_from_db"]:
        print(f"  Missing from DB: {len(result['missing_from_db'])}")
    if result["unresolved"]:
        print(f"  Unresolved light curves: {len(result['unresolved'])}")
    if result["failed"]:
        print(f"  Failed recomputes: {len(result['failed'])}")

    if args.review_sync_enabled:
        auto_export_review_bundle(
            db_path,
            args.review_sync_dir,
            hash_assets=bool(args.review_sync_hash_assets),
        )
    else:
        print("Review Git bundle auto-sync disabled by --no-review-sync")

    if args.output_review_db:
        rebuilt_path = Path(args.output_review_db).expanduser().resolve()
        rebuilt = rebuild_review_db(
            db_path,
            rebuilt_path,
            overwrite=bool(args.overwrite_rebuild_db),
        )
        print(f"Rebuilt DB written to {rebuilt_path}")
        print(
            "  copied {candidates} candidates, {reviews} reviews, "
            "{review_history} history rows, {app_state} app-state rows".format(**rebuilt)
        )


if __name__ == "__main__":
    main()
