from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path

from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd

from malca.config import VSX_CROSSMATCH_PATH, VSX_MAX_SEP_ARCSEC, VSX_RAW_CATALOG_PATH
from malca.review.store import (
    db_connect,
    load_candidates_file,
    merge_candidate_results,
    merge_vetting_results,
)
from malca.io.table_io import read_feature_table, read_parquet_table
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
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
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
