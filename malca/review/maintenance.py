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

from malca.catalogs.evidence import (
    CATALOG_NEIGHBOR_FILENAME,
    CATALOG_NEIGHBOR_OUTPUT_SUBDIR,
    DEFAULT_CATALOG_NEIGHBOR_QUERY_RADIUS_ARCSEC,
    normalize_catalog_evidence,
    normalize_catalog_evidence_record,
)
from malca.config import VSX_CROSSMATCH_PATH, VSX_MAX_SEP_ARCSEC, VSX_RAW_CATALOG_PATH
from malca.review.store import (
    db_connect,
    ensure_review_db_schema,
    get_candidate_payload,
    load_candidates_file,
    merge_candidate_results,
    merge_vetting_results,
    replace_candidate_payload_fields,
    upsert_catalog_neighbor_rows,
    validate_review_db_integrity,
)
from malca.io.table_io import read_feature_table, read_parquet_table, write_feature_table, write_parquet_table
from malca.products.feature_layers import to_layer_first_frame, with_feature_columns
from malca.review.metadata import has_catalog_vetting_context, has_known_catalog_evidence
from malca.vsx.filter import colspecs as VSX_COLSPECS, vsx_columns as VSX_COLUMNS
from malca.vsx.metadata import normalize_vsx_match_columns, select_best_vsx_matches
from malca.vsx.nearby import VsxNeighbor, find_nearby_vsx


VSX_LIVE_BACKFILL_FILENAME = "vsx_live_backfill.parquet"
VSX_LIVE_PRODUCT_COLUMNS = ("vsx_class", "vsx_sep_arcsec", "vsx_period")
VSX_LIVE_BACKFILL_COLUMNS = (
    "candidate_id",
    "asas_sn_id",
    "gaia_id",
    "ra",
    "dec",
    "vsx_live_status",
    "vsx_live_error",
    "vsx_source",
    "vsx_queried_at",
    "vsx_query_radius_arcsec",
    "vsx_query_limit",
    "vsx_query_timeout_sec",
    "vsx_class",
    "vsx_sep_arcsec",
    "vsx_period",
    "vsx_oid",
    "vsx_name",
    "vsx_ra_deg",
    "vsx_dec_deg",
    "vsx_type_label",
    "vsx_url",
    "vsx_n_neighbors",
    "vsx_neighbor_oids",
    "vsx_neighbor_classes",
    "vsx_neighbor_sep_arcsec",
)
VSX_LIVE_CANDIDATE_FILENAMES = (
    "lc_events_neighbors.parquet",
    "lc_events_vetted.parquet",
    "lc_events_external_lcs.parquet",
)


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


def _clean_text(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _append_payload_coords(df: pd.DataFrame) -> pd.DataFrame:
    if "payload_json" not in df.columns:
        return df
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
    return pd.concat([df, payload_df], axis=1)


def _coalesce_numeric(df: pd.DataFrame, *columns: str) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for column in columns:
        if column not in df.columns:
            continue
        out = out.combine_first(pd.to_numeric(df[column], errors="coerce"))
    return out


def _normalize_candidate_coord_frame(df: pd.DataFrame, *, drop_missing: bool) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["candidate_id", "asas_sn_id", "gaia_id", "ra", "dec", *VSX_LIVE_PRODUCT_COLUMNS])

    requested = [
        "candidate_id",
        "asas_sn_id",
        "gaia_id",
        "ra",
        "dec",
        "ra_deg",
        "dec_deg",
        "payload_json",
        *VSX_LIVE_PRODUCT_COLUMNS,
    ]
    view = with_feature_columns(df, requested)
    if "candidate_id" not in view.columns:
        return pd.DataFrame(columns=["candidate_id", "asas_sn_id", "gaia_id", "ra", "dec", *VSX_LIVE_PRODUCT_COLUMNS])
    view = _append_payload_coords(view)
    for col in ("asas_sn_id", "gaia_id", *VSX_LIVE_PRODUCT_COLUMNS):
        if col not in view.columns:
            view[col] = pd.NA
    view["ra"] = _coalesce_numeric(view, "ra", "ra_deg", "payload_ra")
    view["dec"] = _coalesce_numeric(view, "dec", "dec_deg", "payload_dec")
    keep = ["candidate_id", "asas_sn_id", "gaia_id", "ra", "dec", *VSX_LIVE_PRODUCT_COLUMNS]
    out = view[keep].copy()
    if drop_missing:
        out = out.dropna(subset=["ra", "dec"])
    return out.reset_index(drop=True)


def _candidate_coord_frame(conn, *, drop_missing: bool = True) -> pd.DataFrame:
    table_cols = {
        str(info[1])
        for info in conn.execute("PRAGMA table_info(candidates)").fetchall()
    }
    cols = [
        c
        for c in (
            "candidate_id",
            "asas_sn_id",
            "gaia_id",
            "ra",
            "dec",
            "ra_deg",
            "dec_deg",
            "vsx_class",
            "vsx_sep_arcsec",
            "vsx_period",
            "payload_json",
        )
        if c in table_cols
    ]
    if "candidate_id" not in cols:
        return pd.DataFrame()
    df = pd.read_sql_query(f"SELECT {', '.join(cols)} FROM candidates", conn)
    return _normalize_candidate_coord_frame(df, drop_missing=drop_missing)


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


def _usable_vsx_neighbor(neighbors: list[VsxNeighbor]) -> VsxNeighbor | None:
    for neighbor in neighbors:
        if str(neighbor.vsx_type or "").strip():
            return neighbor
    return None


def _format_neighbor_sep(value: float) -> str:
    try:
        return f"{float(value):.6g}"
    except Exception:
        return ""


def _neighbor_summary(neighbors: list[VsxNeighbor]) -> dict[str, object]:
    return {
        "vsx_n_neighbors": int(len(neighbors)),
        "vsx_neighbor_oids": "|".join(str(neighbor.oid or "").strip() for neighbor in neighbors),
        "vsx_neighbor_classes": "|".join(str(neighbor.vsx_type or "").strip() for neighbor in neighbors),
        "vsx_neighbor_sep_arcsec": "|".join(_format_neighbor_sep(neighbor.sep_arcsec) for neighbor in neighbors),
    }


def _empty_vsx_live_backfill_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(VSX_LIVE_BACKFILL_COLUMNS))


def _base_vsx_live_record(
    row: pd.Series,
    *,
    status: str,
    error: str | None,
    queried_at: str,
    radius_arcsec: float,
    timeout_sec: float,
    limit: int,
) -> dict[str, object]:
    return {
        "candidate_id": _clean_text(row.get("candidate_id")),
        "asas_sn_id": _clean_text(row.get("asas_sn_id")),
        "gaia_id": _clean_text(row.get("gaia_id")),
        "ra": row.get("ra"),
        "dec": row.get("dec"),
        "vsx_live_status": status,
        "vsx_live_error": error,
        "vsx_source": "vizier_live",
        "vsx_queried_at": queried_at,
        "vsx_query_radius_arcsec": float(radius_arcsec),
        "vsx_query_limit": int(limit),
        "vsx_query_timeout_sec": float(timeout_sec),
        "vsx_class": None,
        "vsx_sep_arcsec": None,
        "vsx_period": None,
        "vsx_oid": None,
        "vsx_name": None,
        "vsx_ra_deg": None,
        "vsx_dec_deg": None,
        "vsx_type_label": None,
        "vsx_url": None,
        "vsx_n_neighbors": 0,
        "vsx_neighbor_oids": "",
        "vsx_neighbor_classes": "",
        "vsx_neighbor_sep_arcsec": "",
    }


def _valid_coord_pair(ra: object, dec: object) -> tuple[bool, str | None]:
    try:
        ra_value = float(ra)
        dec_value = float(dec)
    except (TypeError, ValueError):
        return False, "missing coordinates"
    if not np.isfinite(ra_value) or not np.isfinite(dec_value):
        return False, "missing coordinates"
    if not (0.0 <= ra_value <= 360.0 and -90.0 <= dec_value <= 90.0):
        return False, "coordinates out of bounds"
    return True, None


def _build_vsx_live_backfill_frame(
    candidates: pd.DataFrame,
    *,
    radius_arcsec: float,
    timeout_sec: float,
    limit: int,
    only_missing: bool = True,
    max_candidates: int | None = None,
    progress_every: int = 25,
) -> pd.DataFrame:
    candidates = _normalize_candidate_coord_frame(candidates, drop_missing=False)
    if candidates.empty:
        return _empty_vsx_live_backfill_frame()

    if only_missing and "vsx_class" in candidates.columns:
        existing = candidates["vsx_class"].map(_is_missing).astype(bool)
        candidates = candidates.loc[existing].reset_index(drop=True)

    if max_candidates is not None:
        candidates = candidates.head(int(max_candidates)).reset_index(drop=True)

    records: list[dict[str, object]] = []
    total = len(candidates)
    queried_at = datetime.now(timezone.utc).isoformat()
    progress_interval = max(int(progress_every), 0)
    matched_count = 0
    no_match_count = 0
    missing_coords_count = 0
    query_failed_count = 0

    def _print_progress(idx: int) -> None:
        scanned = idx + 1
        if progress_interval <= 0:
            return
        if scanned != total and scanned % progress_interval != 0:
            return
        print(
            "Live VSX lookup: "
            f"scanned {scanned}/{total}, matched {matched_count}, "
            f"no match {no_match_count}, missing coords {missing_coords_count}, "
            f"failed {query_failed_count}",
            flush=True,
        )

    if progress_interval > 0:
        print(
            "Live VSX lookup: "
            f"scanning {total} candidate(s) "
            f"(radius={float(radius_arcsec):g} arcsec, timeout={float(timeout_sec):g}s, limit={int(limit)})",
            flush=True,
        )

    for idx, row in candidates.iterrows():
        valid_coord, coord_error = _valid_coord_pair(row.get("ra"), row.get("dec"))
        if not valid_coord:
            records.append(
                _base_vsx_live_record(
                    row,
                    status="missing_coords",
                    error=coord_error,
                    queried_at=queried_at,
                    radius_arcsec=radius_arcsec,
                    timeout_sec=timeout_sec,
                    limit=limit,
                )
            )
            missing_coords_count += 1
            _print_progress(idx)
            continue

        try:
            neighbors = find_nearby_vsx(
                row.get("ra"),
                row.get("dec"),
                limit=int(limit),
                radius_arcsec=float(radius_arcsec),
                timeout_sec=float(timeout_sec),
            )
        except Exception as exc:
            records.append(
                _base_vsx_live_record(
                    row,
                    status="query_failed",
                    error=str(exc),
                    queried_at=queried_at,
                    radius_arcsec=radius_arcsec,
                    timeout_sec=timeout_sec,
                    limit=limit,
                )
            )
            query_failed_count += 1
            _print_progress(idx)
            continue

        record = _base_vsx_live_record(
            row,
            status="no_match",
            error=None,
            queried_at=queried_at,
            radius_arcsec=radius_arcsec,
            timeout_sec=timeout_sec,
            limit=limit,
        )
        record.update(_neighbor_summary(neighbors))
        neighbor = _usable_vsx_neighbor(neighbors)
        if neighbor is None:
            records.append(record)
            no_match_count += 1
            _print_progress(idx)
            continue

        record.update(
            {
                "vsx_live_status": "matched",
                "vsx_class": str(neighbor.vsx_type).strip(),
                "vsx_sep_arcsec": float(neighbor.sep_arcsec),
                "vsx_period": None if neighbor.period_days is None else float(neighbor.period_days),
                "vsx_oid": _clean_text(neighbor.oid),
                "vsx_name": _clean_text(neighbor.name),
                "vsx_ra_deg": neighbor.ra_deg,
                "vsx_dec_deg": neighbor.dec_deg,
                "vsx_type_label": _clean_text(neighbor.type_label),
                "vsx_url": _clean_text(neighbor.url),
            }
        )
        records.append(record)
        matched_count += 1
        _print_progress(idx)

    if not records:
        return _empty_vsx_live_backfill_frame()
    out = pd.DataFrame(records)
    for column in VSX_LIVE_BACKFILL_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out[list(VSX_LIVE_BACKFILL_COLUMNS)].copy()


def _live_updates_from_backfill_frame(backfill_df: pd.DataFrame) -> pd.DataFrame:
    if backfill_df.empty:
        return pd.DataFrame()
    if "vsx_live_status" in backfill_df.columns:
        matched = backfill_df.loc[backfill_df["vsx_live_status"].astype(str).eq("matched")].copy()
    else:
        matched = backfill_df.copy()
    if matched.empty:
        return pd.DataFrame()
    keep = [col for col in ("candidate_id", "asas_sn_id", *VSX_LIVE_PRODUCT_COLUMNS) if col in matched.columns]
    updates = matched[keep].copy()
    if "vsx_class" in updates.columns:
        updates = updates.loc[~updates["vsx_class"].map(_is_missing)].copy()
    if updates.empty:
        return pd.DataFrame()
    return _vsx_backfill_columns(updates, "candidate_id")


def _load_vsx_backfill_live(
    conn,
    *,
    radius_arcsec: float,
    timeout_sec: float,
    limit: int,
    only_missing: bool = True,
    max_candidates: int | None = None,
    progress_every: int = 25,
) -> pd.DataFrame:
    candidates = _candidate_coord_frame(conn, drop_missing=False)
    backfill_df = _build_vsx_live_backfill_frame(
        candidates,
        radius_arcsec=radius_arcsec,
        timeout_sec=timeout_sec,
        limit=limit,
        only_missing=only_missing,
        max_candidates=max_candidates,
        progress_every=progress_every,
    )
    return _live_updates_from_backfill_frame(backfill_df)


def _merge_vsx_live_updates(
    conn,
    live_updates: pd.DataFrame,
    *,
    only_missing: bool,
) -> int:
    if live_updates.empty or "candidate_id" not in live_updates.columns:
        return 0

    table_cols = {
        str(info[1])
        for info in conn.execute("PRAGMA table_info(candidates)").fetchall()
    }
    select_cols = ["candidate_id", *[col for col in VSX_LIVE_PRODUCT_COLUMNS if col in table_cols]]
    current = {
        str(row[0]).strip(): dict(zip(select_cols[1:], row[1:]))
        for row in conn.execute(f"SELECT {', '.join(select_cols)} FROM candidates").fetchall()
    }
    updated = 0
    for _, row in live_updates.iterrows():
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id not in current:
            continue

        updates = {}
        existing = current[candidate_id]
        for col in VSX_LIVE_PRODUCT_COLUMNS:
            if col not in live_updates.columns or _is_missing(row.get(col)):
                continue
            if only_missing and not _is_missing(existing.get(col)):
                continue
            updates[col] = row[col]
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


def backfill_vsx_live_results(
    conn,
    *,
    radius_arcsec: float,
    timeout_sec: float,
    limit: int,
    only_missing: bool = True,
    max_candidates: int | None = None,
    progress_every: int = 25,
    dry_run: bool = False,
) -> int:
    live_updates = _load_vsx_backfill_live(
        conn,
        radius_arcsec=radius_arcsec,
        timeout_sec=timeout_sec,
        limit=limit,
        only_missing=only_missing,
        max_candidates=max_candidates,
        progress_every=progress_every,
    )
    if live_updates.empty:
        print("Live VSX lookup found no candidate updates")
        return 0
    if dry_run:
        print(live_updates.to_string(index=False))
        print(f"Dry run: would update {len(live_updates)} candidates from live VSX")
        return 0
    return _merge_vsx_live_updates(conn, live_updates, only_missing=only_missing)


CATALOG_EVIDENCE_COLUMNS = (
    "vsx_sep_arcsec",
    "vsx_period",
    "vsx_class",
    "nearby_vsx_dipper_contaminant",
    "nearby_vsx_dipper_class",
    "nearby_vsx_dipper_sep_arcsec",
    "nearby_vsx_dipper_period",
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


def _backup_file(path: Path, *, stamp: str, label: str = "catalog_evidence") -> Path:
    backup = path.with_name(f"{path.name}.pre_{label}_{stamp}.bak")
    shutil.copy2(path, backup)
    return backup


def _candidate_result_paths(results_dir: Path) -> list[Path]:
    return [results_dir / name for name in VSX_LIVE_CANDIDATE_FILENAMES]


def _primary_vsx_live_candidate_path(results_dir: Path) -> Path:
    preferred = results_dir / "lc_events_vetted.parquet"
    if preferred.exists():
        return preferred
    existing = [path for path in _candidate_result_paths(results_dir) if path.exists()]
    if not existing:
        raise FileNotFoundError(f"No lc_events result parquets found under {results_dir}")
    return existing[0]


def _values_equal(left: object, right: object) -> bool:
    if _is_missing(left) and _is_missing(right):
        return True
    if _is_missing(left) or _is_missing(right):
        return False
    try:
        return bool(float(left) == float(right))
    except (TypeError, ValueError):
        return str(left) == str(right)


def _merge_vsx_live_into_candidate_frame(
    df: pd.DataFrame,
    live_updates: pd.DataFrame,
    *,
    only_missing: bool,
) -> tuple[pd.DataFrame, int]:
    if df.empty or live_updates.empty or "candidate_id" not in live_updates.columns:
        return df.copy(), 0
    out = with_feature_columns(df, ["candidate_id", *VSX_LIVE_PRODUCT_COLUMNS])
    if "candidate_id" not in out.columns:
        return out, 0
    for col in VSX_LIVE_PRODUCT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    updates = live_updates.copy()
    updates["candidate_id"] = updates["candidate_id"].astype(str).str.strip()
    update_lookup = {
        str(row["candidate_id"]).strip(): row
        for _, row in updates.iterrows()
        if str(row.get("candidate_id") or "").strip()
    }

    changed_rows: set[object] = set()
    for idx, row in out.iterrows():
        candidate_id = str(row.get("candidate_id") or "").strip()
        update = update_lookup.get(candidate_id)
        if update is None:
            continue
        for col in VSX_LIVE_PRODUCT_COLUMNS:
            if col not in update or _is_missing(update.get(col)):
                continue
            if only_missing and not _is_missing(out.at[idx, col]):
                continue
            if _values_equal(out.at[idx, col], update[col]):
                continue
            out.at[idx, col] = update[col]
            changed_rows.add(idx)
    return out, len(changed_rows)


def _vsx_live_status_stats(backfill_df: pd.DataFrame) -> dict[str, int]:
    stats = {
        "sidecar_rows": int(len(backfill_df)),
        "matched": 0,
        "no_match": 0,
        "missing_coords": 0,
        "query_failed": 0,
    }
    if backfill_df.empty or "vsx_live_status" not in backfill_df.columns:
        return stats
    counts = backfill_df["vsx_live_status"].astype(str).value_counts(dropna=False)
    for key in ("matched", "no_match", "missing_coords", "query_failed"):
        stats[key] = int(counts.get(key, 0))
    return stats


def backfill_vsx_live_run(
    run_dir: Path,
    *,
    review_db: Path | None,
    output_path: Path | None,
    radius_arcsec: float,
    timeout_sec: float,
    limit: int,
    only_missing: bool = True,
    max_candidates: int | None = None,
    progress_every: int = 25,
    dry_run: bool = False,
    update_products: bool = True,
    update_db: bool = True,
    backup: bool = True,
) -> dict[str, object]:
    run_dir = run_dir.expanduser().resolve()
    results_dir = run_dir / "results"
    source_path = _primary_vsx_live_candidate_path(results_dir)
    source_df = read_feature_table(source_path)
    candidates = _normalize_candidate_coord_frame(source_df, drop_missing=False)
    backfill_df = _build_vsx_live_backfill_frame(
        candidates,
        radius_arcsec=radius_arcsec,
        timeout_sec=timeout_sec,
        limit=limit,
        only_missing=only_missing,
        max_candidates=max_candidates,
        progress_every=progress_every,
    )
    live_updates = _live_updates_from_backfill_frame(backfill_df)

    if output_path is None:
        output_path = results_dir / VSX_LIVE_BACKFILL_FILENAME
    output_path = output_path.expanduser().resolve()

    stats: dict[str, object] = {
        **_vsx_live_status_stats(backfill_df),
        "sidecar_path": str(output_path),
        "parquets_updated": 0,
        "parquet_rows_updated": 0,
        "db_candidates_updated": 0,
    }
    if dry_run:
        if backfill_df.empty:
            print("Live VSX lookup produced no sidecar rows")
        else:
            print(backfill_df.to_string(index=False))
        print(
            "Dry run: would write "
            f"{len(backfill_df)} live VSX row(s) to {output_path}; "
            f"{len(live_updates)} matched candidate update(s)"
        )
        return stats

    write_parquet_table(backfill_df, output_path)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if update_products and not live_updates.empty:
        for path in [path for path in _candidate_result_paths(results_dir) if path.exists()]:
            df = read_feature_table(path)
            patched, changed = _merge_vsx_live_into_candidate_frame(
                df,
                live_updates,
                only_missing=only_missing,
            )
            if changed <= 0:
                continue
            if backup:
                _backup_file(path, stamp=stamp, label="vsx_live_backfill")
            write_feature_table(to_layer_first_frame(patched), path)
            stats["parquets_updated"] = int(stats["parquets_updated"]) + 1
            stats["parquet_rows_updated"] = int(stats["parquet_rows_updated"]) + int(changed)

    if update_db and not live_updates.empty:
        if review_db is None:
            review_db = run_dir / "review" / "review.db"
        review_db = review_db.expanduser().resolve()
        if review_db.exists():
            if backup:
                _backup_file(review_db, stamp=stamp, label="vsx_live_backfill")
            with closing(db_connect(review_db)) as conn:
                stats["db_candidates_updated"] = _merge_vsx_live_updates(
                    conn,
                    live_updates,
                    only_missing=only_missing,
                )
    return stats


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


CATALOG_NEIGHBOR_BACKFILL_CANDIDATE_FILENAMES = (
    "lc_events_external_lcs.parquet",
    "lc_events_vetted.parquet",
    "lc_events_spectra.parquet",
    "lc_events_neighbors.parquet",
    "lc_events_characterized.parquet",
    "lc_events_classified.parquet",
    "lc_events.parquet",
)


def _catalog_neighbor_default_output(run_dir: Path | None) -> Path:
    if run_dir is None:
        return Path(CATALOG_NEIGHBOR_OUTPUT_SUBDIR) / CATALOG_NEIGHBOR_FILENAME
    return run_dir / "results" / CATALOG_NEIGHBOR_OUTPUT_SUBDIR / CATALOG_NEIGHBOR_FILENAME


def _catalog_neighbor_candidate_path(run_dir: Path) -> Path | None:
    results_dir = run_dir / "results"
    for filename in CATALOG_NEIGHBOR_BACKFILL_CANDIDATE_FILENAMES:
        path = results_dir / filename
        if path.exists():
            return path
    return None


def _load_catalog_neighbor_candidate_file(path: Path) -> pd.DataFrame:
    try:
        return load_candidates_file(path)
    except ValueError:
        return read_parquet_table(path)


def _load_catalog_neighbor_review_candidates(review_db: Path) -> pd.DataFrame:
    with closing(db_connect(review_db)) as conn:
        ids = [
            str(row[0])
            for row in conn.execute("SELECT candidate_id FROM candidates ORDER BY candidate_id").fetchall()
        ]
        return pd.DataFrame([get_candidate_payload(conn, candidate_id) for candidate_id in ids])


def _load_catalog_neighbor_backfill_candidates(
    *,
    input_path: Path | None,
    run_dir: Path | None,
    review_db: Path | None,
) -> tuple[pd.DataFrame, Path | None]:
    if input_path is not None:
        path = input_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Candidate input not found: {path}")
        return _load_catalog_neighbor_candidate_file(path), path

    if run_dir is not None:
        path = _catalog_neighbor_candidate_path(run_dir)
        if path is not None:
            return _load_catalog_neighbor_candidate_file(path), path

    if review_db is not None:
        return _load_catalog_neighbor_review_candidates(review_db.expanduser().resolve()), None

    raise FileNotFoundError("Provide --input, --run-dir with result parquets, or --review-db")


def backfill_catalog_neighbors_run(
    *,
    run_dir: Path | None,
    review_db: Path | None,
    input_path: Path | None,
    output_path: Path | None,
    radius_arcsec: float,
    method: str,
    catalogs: list[str] | tuple[str, ...] | None,
    chunk_size: int,
    xmatch_timeout_sec: float | None,
    update_db: bool,
) -> dict[str, object]:
    from malca.enrichment.vetting import (
        CATALOG_NEIGHBOR_DEFAULT_CHUNK_SIZE,
        CATALOG_NEIGHBOR_DEFAULT_XMATCH_TIMEOUT_SEC,
        collect_catalog_neighbors,
    )

    resolved_run_dir = run_dir.expanduser().resolve() if run_dir is not None else None
    resolved_review_db = review_db.expanduser().resolve() if review_db is not None else None
    if resolved_review_db is None and resolved_run_dir is not None:
        default_db = resolved_run_dir / "review" / "review.db"
        if default_db.exists():
            resolved_review_db = default_db
    if update_db and resolved_review_db is None:
        raise FileNotFoundError("--review-db is required for DB update when --run-dir has no review/review.db")
    if output_path is None:
        output_path = _catalog_neighbor_default_output(resolved_run_dir)
    output_path = output_path.expanduser().resolve()

    candidates, source_path = _load_catalog_neighbor_backfill_candidates(
        input_path=input_path,
        run_dir=resolved_run_dir,
        review_db=resolved_review_db,
    )
    selected_catalogs = tuple(catalogs) if catalogs else None
    effective_chunk_size = int(chunk_size or CATALOG_NEIGHBOR_DEFAULT_CHUNK_SIZE)
    effective_timeout = (
        CATALOG_NEIGHBOR_DEFAULT_XMATCH_TIMEOUT_SEC
        if xmatch_timeout_sec is None
        else float(xmatch_timeout_sec)
    )
    print(
        "Catalog-neighbor backfill: "
        f"source={source_path if source_path is not None else resolved_review_db}; "
        f"candidates={len(candidates)}; radius={float(radius_arcsec):g}\"; "
        f"method={method}; chunk_size={effective_chunk_size}; "
        f"xmatch_timeout={effective_timeout:g}s",
        flush=True,
    )
    neighbors = collect_catalog_neighbors(
        candidates,
        radius_arcsec=radius_arcsec,
        method="xmatch" if method == "xmatch" else "tap",
        catalogs=selected_catalogs,
        chunk_size=effective_chunk_size,
        xmatch_timeout_sec=effective_timeout,
        show_progress=True,
    )
    print(f"Catalog-neighbor backfill: writing sidecar {output_path}", flush=True)
    write_parquet_table(neighbors, output_path)

    db_rows = 0
    if update_db:
        print(f"Catalog-neighbor backfill: importing into {resolved_review_db}", flush=True)
        with closing(db_connect(resolved_review_db)) as conn:
            db_rows = upsert_catalog_neighbor_rows(conn, neighbors)

    return {
        "candidate_rows": int(len(candidates)),
        "neighbor_rows": int(len(neighbors)),
        "output": str(output_path),
        "source": str(source_path) if source_path is not None else str(resolved_review_db),
        "review_db": str(resolved_review_db) if resolved_review_db is not None else None,
        "db_rows": int(db_rows),
        "radius_arcsec": float(radius_arcsec),
        "chunk_size": effective_chunk_size,
        "xmatch_timeout_sec": effective_timeout,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malca review-maint",
        description="Review database maintenance commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_db = subparsers.add_parser(
        "validate-db",
        help="Run explicit SQLite and SED integrity checks on a review DB",
    )
    validate_db.add_argument("--review-db", required=True, type=Path, help="Review SQLite DB")

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

    backfill_vsx_live = subparsers.add_parser(
        "backfill-vsx-live",
        help="Backfill VSX labels using live VizieR cone searches",
    )
    backfill_vsx_live.add_argument("--run-dir", type=Path, default=None, help="MALCA run directory; writes <run-dir>/results/vsx_live_backfill.parquet and patches run products")
    backfill_vsx_live.add_argument("--review-db", type=Path, default=None, help="Review SQLite DB (required without --run-dir; default with --run-dir: <run-dir>/review/review.db)")
    backfill_vsx_live.add_argument("--output", type=Path, default=None, help="Live VSX sidecar parquet (default: <run-dir>/results/vsx_live_backfill.parquet)")
    backfill_vsx_live.add_argument("--radius", type=float, default=VSX_MAX_SEP_ARCSEC, help="Live VSX match radius in arcsec")
    backfill_vsx_live.add_argument("--timeout", type=float, default=5.0, help="Per-candidate VizieR timeout in seconds")
    backfill_vsx_live.add_argument("--limit", type=int, default=3, help="Maximum nearby VSX rows to inspect per candidate")
    backfill_vsx_live.add_argument("--overwrite-existing", action="store_true", help="Also refresh candidates that already have vsx_class")
    backfill_vsx_live.add_argument("--max-candidates", type=int, default=None, help="Limit candidates scanned, useful for smoke tests")
    backfill_vsx_live.add_argument("--progress-every", type=int, default=25, help="Print live lookup progress every N scanned candidates; use 0 to disable")
    backfill_vsx_live.add_argument("--dry-run", action="store_true", help="Print live VSX rows without writing outputs")
    backfill_vsx_live.add_argument("--no-product-update", action="store_true", help="Write the sidecar but do not patch lc_events result parquets")
    backfill_vsx_live.add_argument("--no-db-update", action="store_true", help="Write the sidecar/products but do not update a review DB")
    backfill_vsx_live.add_argument("--no-backup", action="store_true", help="Do not create .bak files before patching products or DB")

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

    backfill_neighbors = subparsers.add_parser(
        "backfill-catalog-neighbors",
        help="Collect long-form catalog neighbors and optionally import them into a review DB",
    )
    backfill_neighbors.add_argument("--run-dir", type=Path, default=None, help="MALCA run directory; default output is <run-dir>/results/vetting_catalog_neighbors/catalog_neighbors.parquet")
    backfill_neighbors.add_argument("--review-db", type=Path, default=None, help="Review SQLite DB (default with --run-dir: <run-dir>/review/review.db if present)")
    backfill_neighbors.add_argument("--input", type=Path, default=None, help="Candidate parquet/CSV input override")
    backfill_neighbors.add_argument("--output", type=Path, default=None, help="Output sidecar parquet")
    backfill_neighbors.add_argument("--radius", type=float, default=DEFAULT_CATALOG_NEIGHBOR_QUERY_RADIUS_ARCSEC, help="Catalog-neighbor collection radius in arcsec")
    backfill_neighbors.add_argument("--method", choices=["xmatch", "tap"], default="xmatch", help="Remote crossmatch method for catalogs that support both")
    backfill_neighbors.add_argument("--catalogs", type=str, default=None, help="Comma-separated v1 catalogs to collect (default: all)")
    backfill_neighbors.add_argument("--chunk-size", type=int, default=250, help="Rows per remote crossmatch chunk")
    backfill_neighbors.add_argument("--xmatch-timeout", type=float, default=120.0, help="Timeout in seconds for each remote XMatch/TAP chunk")
    backfill_neighbors.add_argument("--no-db-update", action="store_true", help="Write the sidecar but do not import it into the review DB")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.command == "validate-db":
        review_db = args.review_db.expanduser().resolve()
        ensure_review_db_schema(review_db)
        result = validate_review_db_integrity(review_db)
        print(
            f"Review DB validation passed: {result['path']} "
            f"(schema={result['review_schema_version']}, quick_check=ok, foreign_keys=ok)"
        )
        return

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

    if args.command == "backfill-catalog-neighbors":
        stats = backfill_catalog_neighbors_run(
            run_dir=args.run_dir,
            review_db=args.review_db,
            input_path=args.input,
            output_path=args.output,
            radius_arcsec=args.radius,
            method=args.method,
            catalogs=(
                [value.strip() for value in str(args.catalogs).split(",") if value.strip()]
                if args.catalogs
                else None
            ),
            chunk_size=args.chunk_size,
            xmatch_timeout_sec=args.xmatch_timeout,
            update_db=not args.no_db_update,
        )
        db_note = (
            f"; imported {stats['db_rows']} row(s) into {stats['review_db']}"
            if stats.get("review_db") and not args.no_db_update
            else ""
        )
        print(
            f"Collected {stats['neighbor_rows']} catalog-neighbor row(s) "
            f"for {stats['candidate_rows']} candidate(s) at radius "
            f"{stats['radius_arcsec']:g} arcsec. Sidecar: {stats['output']}{db_note}."
        )
        return

    if args.command == "backfill-vsx-live" and args.run_dir is not None:
        stats = backfill_vsx_live_run(
            args.run_dir,
            review_db=args.review_db,
            output_path=args.output,
            radius_arcsec=args.radius,
            timeout_sec=args.timeout,
            limit=args.limit,
            only_missing=not args.overwrite_existing,
            max_candidates=args.max_candidates,
            progress_every=args.progress_every,
            dry_run=args.dry_run,
            update_products=not args.no_product_update,
            update_db=not args.no_db_update,
            backup=not args.no_backup,
        )
        print(
            f"Live VSX scanned {stats['sidecar_rows']} candidate(s): "
            f"{stats['matched']} matched, {stats['no_match']} no match, "
            f"{stats['missing_coords']} missing coords, {stats['query_failed']} failed. "
            f"Sidecar: {stats['sidecar_path']}. "
            f"Updated {stats['parquet_rows_updated']} row(s) across "
            f"{stats['parquets_updated']} result parquet(s); "
            f"updated {stats['db_candidates_updated']} DB candidate(s)."
        )
        return

    if args.command == "backfill-vsx-live" and args.review_db is None:
        raise SystemExit("--review-db is required when --run-dir is not provided")

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
        elif args.command == "backfill-vsx-live":
            updated = backfill_vsx_live_results(
                conn,
                radius_arcsec=args.radius,
                timeout_sec=args.timeout,
                limit=args.limit,
                only_missing=not args.overwrite_existing,
                max_candidates=args.max_candidates,
                progress_every=args.progress_every,
                dry_run=args.dry_run,
            )
        else:  # pragma: no cover - argparse enforces choices
            raise SystemExit(f"Unknown command: {args.command}")
    print(f"Updated {updated} candidates in {review_db}")


if __name__ == "__main__":
    main()
