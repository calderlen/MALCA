from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd

from malca.catalogs.evidence import normalize_catalog_evidence, normalize_catalog_evidence_record
from malca.config import VSX_CROSSMATCH_PATH, VSX_MAX_SEP_ARCSEC, VSX_RAW_CATALOG_PATH
from malca.review.store import (
    db_connect,
    load_candidates_file,
    merge_candidate_results,
    merge_vetting_results,
    replace_candidate_payload_fields,
)
from malca.io.table_io import read_feature_table, read_parquet_table, write_feature_table
from malca.products.feature_layers import to_layer_first_frame, with_feature_columns
from malca.review.metadata import has_catalog_vetting_context, has_known_catalog_evidence
from malca.vsx.filter import colspecs as VSX_COLSPECS, vsx_columns as VSX_COLUMNS
from malca.vsx.metadata import normalize_vsx_match_columns, select_best_vsx_matches


def _vsx_backfill_columns(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    df = normalize_vsx_match_columns(df)
    keep: list[str] = []
    for col in [id_column, "asas_sn_id", "candidate_id", "vsx_class", "vsx_sep_arcsec", "vsx_period"]:
        if col in df.columns and col not in keep:
            keep.append(col)
    out = df[keep].copy()
    required = {id_column, "vsx_class", "vsx_sep_arcsec"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"VSX backfill input missing required columns: {missing}")
    return select_best_vsx_matches(out, id_column=id_column)


def _load_vsx_backfill_crossmatch(path: Path) -> pd.DataFrame:
    df = read_parquet_table(path)
    return _vsx_backfill_columns(df, "asas_sn_id")


def _candidate_coord_frame(conn) -> pd.DataFrame:
    table_cols = {
        str(info[1])
        for info in conn.execute("PRAGMA table_info(candidates)").fetchall()
    }
    cols = [c for c in ("candidate_id", "asas_sn_id", "ra", "dec", "ra_deg", "dec_deg", "payload_json") if c in table_cols]
    if "candidate_id" not in cols:
        return pd.DataFrame()
    df = pd.read_sql_query(f"SELECT {', '.join(cols)} FROM candidates", conn)
    if "payload_json" in df.columns:
        payload_coords: list[dict[str, object]] = []
        for raw in df["payload_json"].tolist():
            try:
                payload = json.loads(raw) if raw else {}
            except Exception:
                payload = {}
            payload_coords.append(
                {
                    "payload_ra": payload.get("ra", payload.get("ra_deg")),
                    "payload_dec": payload.get("dec", payload.get("dec_deg")),
                }
            )
        payload_df = pd.DataFrame(payload_coords, index=df.index)
        df = pd.concat([df, payload_df], axis=1)
    if "ra" not in df.columns and "ra_deg" in df.columns:
        df["ra"] = df["ra_deg"]
    elif "ra_deg" in df.columns:
        df["ra"] = pd.to_numeric(df["ra"], errors="coerce").combine_first(pd.to_numeric(df["ra_deg"], errors="coerce"))
    elif "payload_ra" in df.columns:
        df["ra"] = df["payload_ra"]
    if "dec" not in df.columns and "dec_deg" in df.columns:
        df["dec"] = df["dec_deg"]
    elif "dec_deg" in df.columns:
        df["dec"] = pd.to_numeric(df["dec"], errors="coerce").combine_first(pd.to_numeric(df["dec_deg"], errors="coerce"))
    elif "payload_dec" in df.columns:
        df["dec"] = df["payload_dec"]
    if "payload_ra" in df.columns:
        df["ra"] = pd.to_numeric(df["ra"], errors="coerce").combine_first(pd.to_numeric(df["payload_ra"], errors="coerce"))
    if "payload_dec" in df.columns:
        df["dec"] = pd.to_numeric(df["dec"], errors="coerce").combine_first(pd.to_numeric(df["payload_dec"], errors="coerce"))
    if "ra" not in df.columns or "dec" not in df.columns:
        return pd.DataFrame()
    df["ra"] = pd.to_numeric(df.get("ra"), errors="coerce")
    df["dec"] = pd.to_numeric(df.get("dec"), errors="coerce")
    return df.dropna(subset=["ra", "dec"]).reset_index(drop=True)


def _load_vsx_backfill_raw(
    conn,
    path: Path,
    *,
    radius_arcsec: float,
    chunksize: int,
) -> pd.DataFrame:
    candidates = _candidate_coord_frame(conn)
    if candidates.empty:
        return pd.DataFrame()

    candidate_coords = SkyCoord(
        ra=candidates["ra"].to_numpy(dtype=float) * u.deg,
        dec=candidates["dec"].to_numpy(dtype=float) * u.deg,
    )
    radius = float(radius_arcsec) * u.arcsec
    best: dict[str, dict[str, object]] = {}

    for chunk in pd.read_fwf(
        path,
        colspecs=VSX_COLSPECS,
        names=VSX_COLUMNS,
        dtype=str,
        chunksize=int(chunksize),
    ):
        chunk = chunk.copy()
        chunk["ra"] = pd.to_numeric(chunk["ra"], errors="coerce")
        chunk["dec"] = pd.to_numeric(chunk["dec"], errors="coerce")
        chunk["period"] = pd.to_numeric(chunk.get("period"), errors="coerce")
        chunk = chunk.dropna(subset=["ra", "dec"]).reset_index(drop=True)
        if chunk.empty:
            continue
        vsx_coords = SkyCoord(
            ra=chunk["ra"].to_numpy(dtype=float) * u.deg,
            dec=chunk["dec"].to_numpy(dtype=float) * u.deg,
        )
        idx_candidate, sep2d, _ = vsx_coords.match_to_catalog_sky(candidate_coords)
        matched = sep2d <= radius
        if not np.any(matched):
            continue
        matched_rows = chunk.loc[matched].copy()
        matched_rows["_candidate_idx"] = idx_candidate[matched]
        matched_rows["vsx_sep_arcsec"] = sep2d[matched].to(u.arcsec).value
        for _, row in matched_rows.iterrows():
            candidate = candidates.iloc[int(row["_candidate_idx"])]
            candidate_id = str(candidate["candidate_id"])
            sep = float(row["vsx_sep_arcsec"])
            previous = best.get(candidate_id)
            if previous is not None and sep >= float(previous["vsx_sep_arcsec"]):
                continue
            best[candidate_id] = {
                "candidate_id": candidate_id,
                "asas_sn_id": str(candidate.get("asas_sn_id") or "").strip(),
                "vsx_class": str(row.get("class") or "").strip(),
                "vsx_sep_arcsec": sep,
                "vsx_period": row.get("period"),
            }

    if not best:
        return pd.DataFrame()
    return _vsx_backfill_columns(pd.DataFrame(best.values()), "candidate_id")


def backfill_vsx_results(
    conn,
    *,
    crossmatch: Path,
    raw_vsx: Path | None,
    radius_arcsec: float,
    chunksize: int,
) -> int:
    updated = 0
    crossmatch_exists = crossmatch.exists()
    if crossmatch_exists:
        updated += merge_candidate_results(conn, _load_vsx_backfill_crossmatch(crossmatch), id_column="asas_sn_id")
    elif raw_vsx is None or not raw_vsx.exists():
        raise FileNotFoundError(f"VSX crossmatch file not found: {crossmatch}")

    if raw_vsx is not None and raw_vsx.exists() and not crossmatch_exists:
        raw_updates = _load_vsx_backfill_raw(
            conn,
            raw_vsx,
            radius_arcsec=radius_arcsec,
            chunksize=chunksize,
        )
        if not raw_updates.empty:
            updated += merge_candidate_results(conn, raw_updates, id_column="candidate_id")
    return updated


CATALOG_EVIDENCE_COLUMNS = (
    "vsx_sep_arcsec",
    "vsx_period",
    "vsx_class",
    "asassn_var_name",
    "asassn_var_period",
    "asassn_var_type",
)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null", "<na>"}
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _backup_file(path: Path, *, stamp: str) -> Path:
    backup = path.with_name(f"{path.name}.pre_catalog_evidence_{stamp}.bak")
    shutil.copy2(path, backup)
    return backup


def _catalog_evidence_update_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["candidate_id", *CATALOG_EVIDENCE_COLUMNS]
    view = with_feature_columns(df, columns)
    keep = [col for col in columns if col in view.columns]
    if "candidate_id" not in keep:
        return pd.DataFrame(columns=columns)
    return view[keep].copy()


def _fill_asassn_variables_blank_only(
    df: pd.DataFrame,
    *,
    local_csv: Path | None,
    radius_arcsec: float,
) -> pd.DataFrame:
    from malca.enrichment.vetting import crossmatch_asassn_variables

    out = with_feature_columns(df, ["ra", "dec", "asassn_var_name", "asassn_var_type", "asassn_var_period"])
    if "ra" not in out.columns or "dec" not in out.columns:
        return out
    matched = crossmatch_asassn_variables(
        out.copy(),
        method="local",
        radius_arcsec=radius_arcsec,
        local_csv=local_csv,
    )
    for column in ("asassn_var_name", "asassn_var_type", "asassn_var_period"):
        if column not in out.columns:
            out[column] = pd.NA
        if column not in matched.columns:
            continue
        fill = out[column].map(_is_missing).astype(bool) & ~matched[column].map(_is_missing).astype(bool)
        if fill.any():
            out.loc[fill, column] = matched.loc[fill, column]
    return out


def _sparse_merge_catalog_evidence(conn, update_df: pd.DataFrame) -> int:
    if update_df.empty or "candidate_id" not in update_df.columns:
        return 0

    known_ids = {
        str(row[0]).strip()
        for row in conn.execute("SELECT candidate_id FROM candidates").fetchall()
    }
    updated = 0
    for _, row in update_df.iterrows():
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id not in known_ids:
            continue
        updates = {
            col: row[col]
            for col in CATALOG_EVIDENCE_COLUMNS
            if col in update_df.columns and not _is_missing(row[col])
        }
        updates = normalize_catalog_evidence_record(updates)
        if not updates:
            continue
        if has_known_catalog_evidence(updates):
            updates["vetting_likely_known"] = True
        elif has_catalog_vetting_context(updates):
            updates["vetting_likely_known"] = False
        if replace_candidate_payload_fields(conn, candidate_id, updates, commit=False):
            updated += 1
    conn.commit()
    return updated


def repair_catalog_evidence_run(
    run_dir: Path,
    *,
    review_db: Path | None,
    neighbors_long_path: Path | None,
    vsx_max_sep_arcsec: float,
    include_asassn_var: bool,
    asassn_local_csv: Path | None,
    asassn_radius_arcsec: float,
    backup: bool,
) -> dict[str, int]:
    run_dir = run_dir.expanduser().resolve()
    results_dir = run_dir / "results"
    if neighbors_long_path is None:
        neighbors_long_path = results_dir / "neighbor_enrichment" / "neighbors_long.parquet"
    neighbors_long_path = neighbors_long_path.expanduser().resolve()
    if not neighbors_long_path.exists():
        raise FileNotFoundError(f"Neighbor long table not found: {neighbors_long_path}")

    neighbors_long = read_parquet_table(neighbors_long_path)
    candidate_paths = [
        results_dir / "lc_events_neighbors.parquet",
        results_dir / "lc_events_vetted.parquet",
        results_dir / "lc_events_external_lcs.parquet",
    ]
    existing_paths = [path for path in candidate_paths if path.exists()]
    if not existing_paths:
        raise FileNotFoundError(f"No repairable lc_events result parquets found under {results_dir}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stats = {"parquets_repaired": 0, "db_candidates_updated": 0}
    normalized_by_path: dict[Path, pd.DataFrame] = {}

    for path in existing_paths:
        df = read_feature_table(path)
        repaired = normalize_catalog_evidence(
            df,
            neighbors_long=neighbors_long,
            vsx_max_sep_arcsec=vsx_max_sep_arcsec,
        )
        if include_asassn_var:
            repaired = _fill_asassn_variables_blank_only(
                repaired,
                local_csv=asassn_local_csv,
                radius_arcsec=asassn_radius_arcsec,
            )
        if backup:
            _backup_file(path, stamp=stamp)
        write_feature_table(to_layer_first_frame(repaired), path)
        normalized_by_path[path] = repaired
        stats["parquets_repaired"] += 1

    if review_db is None:
        review_db = run_dir / "review" / "review.db"
    review_db = review_db.expanduser().resolve()
    if review_db.exists():
        if backup:
            _backup_file(review_db, stamp=stamp)
        source_path = next(
            (path for path in reversed(candidate_paths) if path in normalized_by_path),
            existing_paths[-1],
        )
        update_df = _catalog_evidence_update_frame(normalized_by_path[source_path])
        with closing(db_connect(review_db)) as conn:
            stats["db_candidates_updated"] = _sparse_merge_catalog_evidence(conn, update_df)
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malca review-maint",
        description="Review database maintenance commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_vetting = subparsers.add_parser(
        "merge-vetting",
        help="Merge vetting results into a review DB",
    )
    merge_vetting.add_argument("--review-db", required=True, type=Path, help="Review SQLite DB")
    merge_vetting.add_argument("--input", required=True, type=Path, help="Vetting parquet file")

    merge_candidates = subparsers.add_parser(
        "merge-candidates",
        help="Merge candidate columns into a review DB",
    )
    merge_candidates.add_argument("--review-db", required=True, type=Path, help="Review SQLite DB")
    merge_candidates.add_argument("--input", required=True, type=Path, help="Candidate Parquet file")

    backfill_vsx = subparsers.add_parser(
        "backfill-vsx",
        help="Backfill full VSX labels into a review DB",
    )
    backfill_vsx.add_argument("--review-db", required=True, type=Path, help="Review SQLite DB")
    backfill_vsx.add_argument("--crossmatch", type=Path, default=VSX_CROSSMATCH_PATH, help="Full ASAS-SN x VSX crossmatch parquet")
    backfill_vsx.add_argument("--raw-vsx", type=Path, default=VSX_RAW_CATALOG_PATH, help="Raw VSX catalog fallback")
    backfill_vsx.add_argument("--radius", type=float, default=VSX_MAX_SEP_ARCSEC, help="Raw VSX fallback match radius in arcsec")
    backfill_vsx.add_argument("--chunksize", type=int, default=200_000, help="Raw VSX fallback read chunk size")
    backfill_vsx.add_argument("--no-raw-fallback", action="store_true", help="Do not scan raw VSX when crossmatch is absent/incomplete")

    repair_catalog = subparsers.add_parser(
        "repair-catalog-evidence",
        help="Repair canonical catalog evidence in run products and review DB",
    )
    repair_catalog.add_argument("--run-dir", required=True, type=Path, help="MALCA run directory")
    repair_catalog.add_argument("--review-db", type=Path, default=None, help="Review SQLite DB (default: <run-dir>/review/review.db)")
    repair_catalog.add_argument("--neighbors-long", type=Path, default=None, help="neighbors_long parquet (default: <run-dir>/results/neighbor_enrichment/neighbors_long.parquet)")
    repair_catalog.add_argument("--vsx-max-sep", type=float, default=VSX_MAX_SEP_ARCSEC, help="Maximum VSX neighbor promotion separation in arcsec")
    repair_catalog.add_argument("--asassn-local-csv", type=Path, default=None, help="Local ASAS-SN variables CSV override")
    repair_catalog.add_argument("--asassn-radius", type=float, default=5.0, help="ASAS-SN variable match radius in arcsec")
    repair_catalog.add_argument("--skip-asassn-var", action="store_true", help="Only repair VSX evidence")
    repair_catalog.add_argument("--no-backup", action="store_true", help="Do not create .bak files before writing")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.command == "repair-catalog-evidence":
        stats = repair_catalog_evidence_run(
            args.run_dir,
            review_db=args.review_db,
            neighbors_long_path=args.neighbors_long,
            vsx_max_sep_arcsec=args.vsx_max_sep,
            include_asassn_var=not args.skip_asassn_var,
            asassn_local_csv=args.asassn_local_csv,
            asassn_radius_arcsec=args.asassn_radius,
            backup=not args.no_backup,
        )
        print(
            f"Repaired {stats['parquets_repaired']} result parquet(s); "
            f"updated {stats['db_candidates_updated']} DB candidate(s)."
        )
        return

    review_db = args.review_db.expanduser().resolve()
    with closing(db_connect(review_db)) as conn:
        if args.command == "merge-vetting":
            input_path = args.input.expanduser().resolve()
            if not input_path.exists():
                raise SystemExit(f"Input file not found: {input_path}")
            df = read_feature_table(input_path)
            updated = merge_vetting_results(conn, df)
        elif args.command == "merge-candidates":
            input_path = args.input.expanduser().resolve()
            if not input_path.exists():
                raise SystemExit(f"Input file not found: {input_path}")
            df = load_candidates_file(input_path)
            updated = merge_candidate_results(conn, df)
        elif args.command == "backfill-vsx":
            raw_vsx = None if args.no_raw_fallback else args.raw_vsx.expanduser().resolve()
            updated = backfill_vsx_results(
                conn,
                crossmatch=args.crossmatch.expanduser().resolve(),
                raw_vsx=raw_vsx,
                radius_arcsec=args.radius,
                chunksize=args.chunksize,
            )
        else:  # pragma: no cover - argparse enforces choices
            raise SystemExit(f"Unknown command: {args.command}")
    print(f"Updated {updated} candidates in {review_db}")


if __name__ == "__main__":
    main()
