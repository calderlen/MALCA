#!/usr/bin/env python
"""Backfill one median-offset, combined-light-curve Sokolovsky ``v`` for July 1.

For every candidate, usable V magnitudes are shifted by ``median(V)-median(g)``
before the g and V observations are concatenated.  ``v`` is calculated once
from that single light curve.  With ``--apply``, the canonical candidate
parquet and Review DB are updated after timestamped backups.
"""
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from malca.core.stats import compute_sokolovsky_peak_to_peak_summary
from malca.io.table_io import read_feature_table, write_feature_table, write_parquet_table
from malca.products.feature_layers import parse_layer_value
from malca.review.store import db_connect, ensure_review_db_schema, replace_candidate_payload_fields


DEFAULT_RUN_DIR = Path("output/runs/dat3-full-extended_2026-07-01-v4")
METRIC_KEY = "stats_variability_sokolovsky_v"
CLIPPED_MEAN_KEY = "stats_clipped_mean_mag_3sigma_about_median"
BAND_KEY = "stats_variability_sokolovsky_v_band"
OFFSET_KEY = "stats_variability_sokolovsky_v_v_minus_g_median_offset_mag"
N_POINTS_KEY = "stats_variability_sokolovsky_v_n_points"
STATUS_KEY = "stats_variability_sokolovsky_v_status"
LEGACY_KEYS = {
    "stats_variability_sokolovsky_v_g",
    "stats_variability_sokolovsky_v_vband",
    "stats_clipped_mean_mag_3sigma_about_median_g",
    "stats_clipped_mean_mag_3sigma_about_median_vband",
}
BACKFILL_KEYS = {
    METRIC_KEY,
    CLIPPED_MEAN_KEY,
    BAND_KEY,
    OFFSET_KEY,
    N_POINTS_KEY,
    STATUS_KEY,
    *LEGACY_KEYS,
}


def _resolve_lightcurve_path(
    candidate_id: str,
    asas_sn_id: str | None,
    raw_path: object,
    run_dir: Path,
) -> Path | None:
    if raw_path not in (None, ""):
        candidate = Path(str(raw_path)).expanduser()
        if candidate.exists():
            return candidate

    bundle_dir = run_dir / "bundle_assets" / "lightcurves"
    names = [candidate_id]
    if asas_sn_id not in (None, ""):
        names.append(str(asas_sn_id))
    if candidate_id.startswith("stv_"):
        names.append(candidate_id.removeprefix("stv_"))
    for name in dict.fromkeys(names):
        for suffix in (".dat3", ".raw2", ".dat2", ".dat", ".csv"):
            candidate = bundle_dir / f"{name}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _empty_record(candidate_id: str, *, status: str, lc_path: Path | None = None) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "lc_path_resolved": str(lc_path) if lc_path is not None else None,
        "sokolovsky_v": np.nan,
        "clipped_mean_mag_3sigma_about_median": np.nan,
        "sokolovsky_v_band": "none",
        "sokolovsky_v_v_minus_g_median_offset_mag": np.nan,
        "sokolovsky_v_n_points": 0,
        "sokolovsky_v_status": status,
    }


def _compute_one(item: tuple[str, str | None, str | None]) -> dict[str, object]:
    candidate_id, asas_sn_id, raw_path = item
    lc_path = _resolve_lightcurve_path(candidate_id, asas_sn_id, raw_path, Path(_WORKER_RUN_DIR))
    if lc_path is None:
        return _empty_record(candidate_id, status="lightcurve_unresolved")
    try:
        _frame, summary = compute_sokolovsky_peak_to_peak_summary(
            lc_path.stem,
            lc_path,
            file_ext=lc_path.suffix.lstrip("."),
        )
        return {
            "candidate_id": candidate_id,
            "lc_path_resolved": str(lc_path),
            "sokolovsky_v": summary["variability_sokolovsky_v"],
            "clipped_mean_mag_3sigma_about_median": summary[
                "clipped_mean_mag_3sigma_about_median"
            ],
            "sokolovsky_v_band": summary["sokolovsky_v_band"],
            "sokolovsky_v_v_minus_g_median_offset_mag": summary[
                "sokolovsky_v_v_minus_g_median_offset_mag"
            ],
            "sokolovsky_v_n_points": summary["sokolovsky_v_n_points"],
            "sokolovsky_v_status": summary["sokolovsky_v_status"],
        }
    except Exception as exc:
        record = _empty_record(candidate_id, status="compute_error", lc_path=lc_path)
        record["error"] = str(exc)
        return record


_WORKER_RUN_DIR = ""


def _worker_init(run_dir: str) -> None:
    global _WORKER_RUN_DIR
    _WORKER_RUN_DIR = run_dir


def _compute_backfill(source: pd.DataFrame, run_dir: Path, workers: int) -> pd.DataFrame:
    items = [
        (
            str(row.candidate_id),
            None if pd.isna(row.asas_sn_id) else str(row.asas_sn_id),
            None if pd.isna(row.lc_path) else str(row.lc_path),
        )
        for row in source[["candidate_id", "asas_sn_id", "lc_path"]].itertuples(index=False)
    ]
    if workers == 1:
        _worker_init(str(run_dir))
        records = [_compute_one(item) for item in items]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(str(run_dir),),
        ) as executor:
            records = list(executor.map(_compute_one, items, chunksize=32))
    return pd.DataFrame.from_records(records)


def _value(row: pd.Series, column: str) -> float | None:
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else None


def _patch_candidate_product(source: pd.DataFrame, backfill: pd.DataFrame) -> pd.DataFrame:
    updates = backfill.set_index("candidate_id", verify_integrity=True)
    out = source.copy()
    for index, candidate_id in out["candidate_id"].astype(str).items():
        result = updates.loc[candidate_id]
        lc_stats = dict(parse_layer_value(out.at[index, "lc_stats"]))
        for key, column in (
            (METRIC_KEY, "sokolovsky_v"),
            (CLIPPED_MEAN_KEY, "clipped_mean_mag_3sigma_about_median"),
            (OFFSET_KEY, "sokolovsky_v_v_minus_g_median_offset_mag"),
        ):
            value = _value(result, column)
            if value is None:
                lc_stats.pop(key, None)
            else:
                lc_stats[key] = value
        for key in LEGACY_KEYS:
            lc_stats.pop(key, None)
        lc_stats[BAND_KEY] = str(result["sokolovsky_v_band"])
        lc_stats[N_POINTS_KEY] = int(result["sokolovsky_v_n_points"])
        lc_stats[STATUS_KEY] = str(result["sokolovsky_v_status"])
        out.at[index, "lc_stats"] = json.dumps(
            lc_stats,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    return out


def _backup(path: Path, stamp: str) -> Path:
    backup = path.with_name(f"{path.name}.pre_sokolovsky_v_median_offset_{stamp}.bak")
    shutil.copy2(path, backup)
    return backup


def _update_review_db(review_db: Path, backfill: pd.DataFrame, commit_every: int) -> int:
    ensure_review_db_schema(review_db)
    updated = 0
    with db_connect(review_db) as conn:
        for index, row in backfill.iterrows():
            updates = {
                METRIC_KEY: _value(row, "sokolovsky_v"),
                CLIPPED_MEAN_KEY: _value(row, "clipped_mean_mag_3sigma_about_median"),
                OFFSET_KEY: _value(row, "sokolovsky_v_v_minus_g_median_offset_mag"),
                BAND_KEY: str(row["sokolovsky_v_band"]),
                N_POINTS_KEY: int(row["sokolovsky_v_n_points"]),
                STATUS_KEY: str(row["sokolovsky_v_status"]),
            }
            if replace_candidate_payload_fields(
                conn,
                str(row["candidate_id"]),
                updates,
                clear_keys=set(BACKFILL_KEYS),
                commit=False,
            ):
                updated += 1
            if commit_every > 0 and (index + 1) % commit_every == 0:
                conn.commit()
        conn.commit()
    return updated


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--review-db", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Combined-light-curve sidecar parquet")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--commit-every", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test candidate limit")
    parser.add_argument("--apply", action="store_true", help="Patch the candidate parquet and Review DB")
    parser.add_argument("--no-product-update", action="store_true")
    parser.add_argument("--no-db-update", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    candidates = (args.candidates or run_dir / "results" / "lc_events_vetted.parquet").expanduser().resolve()
    review_db = (args.review_db or run_dir / "review" / "review.db").expanduser().resolve()
    output = (
        args.output or run_dir / "results" / "sokolovsky_v_median_offset_backfill.parquet"
    ).expanduser().resolve()
    workers = max(int(args.workers), 1)

    if not candidates.exists():
        raise FileNotFoundError(f"Candidate product not found: {candidates}")
    if args.apply and not args.no_db_update and not review_db.exists():
        raise FileNotFoundError(f"Review DB not found: {review_db}")
    if output.exists() and not args.overwrite_output:
        raise FileExistsError(f"Sidecar already exists: {output}; pass --overwrite-output to replace it")
    if args.limit is not None and args.apply and not (args.no_product_update and args.no_db_update):
        raise ValueError("--limit may only be used with --no-product-update --no-db-update")

    source = read_feature_table(candidates)
    if "candidate_id" not in source.columns or "lc_path" not in source.columns:
        raise ValueError("Candidate product must contain candidate_id and lc_path")
    if "asas_sn_id" not in source.columns:
        source["asas_sn_id"] = pd.NA
    source["candidate_id"] = source["candidate_id"].astype(str)
    if args.limit is not None:
        source = source.head(max(int(args.limit), 0)).copy()
    if source.empty:
        raise ValueError("No candidates selected")

    backfill = _compute_backfill(source, run_dir, workers)
    if len(backfill) != len(source) or backfill["candidate_id"].nunique() != len(source):
        raise RuntimeError("Backfill rows do not match the selected candidate scope")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_table(backfill, output)

    counts = backfill["sokolovsky_v_status"].value_counts(dropna=False).to_dict()
    finite = int(np.isfinite(pd.to_numeric(backfill["sokolovsky_v"], errors="coerce")).sum())
    print(f"Wrote {len(backfill):,} median-offset combined rows ({finite:,} finite v values): {output}")
    print(f"Status counts: {counts}")
    if not args.apply:
        print("Dry run only: pass --apply to patch the canonical parquet and Review DB.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not args.no_product_update:
        if not args.no_backup:
            print(f"Candidate backup: {_backup(candidates, stamp)}")
        patched = _patch_candidate_product(source, backfill)
        write_feature_table(patched, candidates)
        print(f"Patched candidate parquet: {candidates} ({len(patched):,} rows)")
    if not args.no_db_update:
        if not args.no_backup:
            print(f"Review DB backup: {_backup(review_db, stamp)}")
        updated = _update_review_db(review_db, backfill, max(int(args.commit_every), 0))
        print(f"Patched Review DB: {review_db} ({updated:,} rows)")


if __name__ == "__main__":
    main()
