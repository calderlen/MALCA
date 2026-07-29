"""Fit stored SED photometry without re-querying any catalog."""

from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from malca.enrichment.sed_model import (
    SED_MODEL_FIT_VERSION,
    fit_sed_models,
    sed_fit_input_state,
    sed_fit_recipe_hash,
    sed_measurement_set_hash,
    upsert_sed_model_results,
)
from malca.enrichment.sed_photometry import _ensure_candidate_id, _expand_sqlite_candidate_payloads
from malca.review.store import db_connect


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malca sed-fit",
        description=(
            "Backfill the current extinction-aware, bandpass-integrated Kurucz fit "
            "from photometry already stored in a review database. Catalogs are not queried."
        ),
    )
    parser.add_argument("review_db", type=Path, help="Review SQLite DB containing candidates and sed_photometry")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Refit every candidate with stored photometry; default is missing/stale fits only.",
    )
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help="Restrict to a candidate ID; repeat for multiple candidates.",
    )
    parser.add_argument(
        "--event-class",
        default=None,
        help=(
            "Restrict to candidates with this reviews.event_class label, for example "
            "'dipper'. May be combined with --all to force-refit that cohort only."
        ),
    )
    parser.add_argument(
        "--control-for-event-class",
        default=None,
        help=(
            "Select an ordinary-star control cohort matched to this reviewed event class "
            "in Teff, W1, Galactic latitude, fit quality, and extinction."
        ),
    )
    parser.add_argument(
        "--control-sample-size",
        type=int,
        default=None,
        help="Number of matched controls to backfill; requires --control-for-event-class.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many candidates")
    parser.add_argument("--batch-size", type=int, default=25, help="Commit this many candidates per batch (default: 25)")
    parser.add_argument("--fit-workers", type=int, default=1, help="Candidate fitting threads (default: 1)")
    parser.add_argument("--curve-points", type=int, default=400, help="Samples in each stored model curve (default: 400)")
    parser.add_argument(
        "--no-bandpass-download",
        action="store_true",
        help="Use only response curves already present in the local SVO cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the selected backfill population without fitting or changing fit rows.",
    )
    return parser


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone() is not None


def _review_event_class_candidate_ids(conn: sqlite3.Connection, event_class: str) -> list[str]:
    label = str(event_class or "").strip()
    if not label:
        return []
    if not _table_exists(conn, "reviews"):
        raise ValueError("Review DB does not contain a reviews table required by --event-class")
    review_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(reviews)").fetchall()}
    if "candidate_id" not in review_columns or "event_class" not in review_columns:
        raise ValueError("Review DB reviews table does not expose candidate_id and event_class")
    return [
        str(row[0])
        for row in conn.execute(
            "SELECT DISTINCT candidate_id FROM reviews "
            "WHERE lower(trim(COALESCE(event_class, ''))) = lower(trim(?)) "
            "ORDER BY candidate_id",
            (label,),
        ).fetchall()
    ]


def _matched_control_candidate_ids(
    conn: sqlite3.Connection,
    event_class: str,
    sample_size: int,
) -> list[str]:
    """Select deterministic ordinary-star controls near an event cohort."""

    sample_size = max(int(sample_size), 0)
    if sample_size == 0:
        return []
    required_tables = {"candidates", "reviews", "sed_model_fits"}
    missing = sorted(table for table in required_tables if not _table_exists(conn, table))
    if missing:
        raise ValueError(f"Review DB lacks matched-control table(s): {', '.join(missing)}")
    candidate_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(candidates)")}
    required_columns = {"candidate_id", "w1", "gal_b", "A_v_3d", "yso_class"}
    missing_columns = sorted(required_columns - candidate_columns)
    if missing_columns:
        raise ValueError(f"Candidates table lacks matched-control field(s): {', '.join(missing_columns)}")

    feature_select = (
        "f.teff_k AS teff_k, c.w1 AS w1, ABS(c.gal_b) AS abs_gal_b, "
        "f.reduced_chi2 AS reduced_chi2, c.A_v_3d AS av"
    )
    targets = pd.read_sql_query(
        f"SELECT DISTINCT c.candidate_id, {feature_select} "
        "FROM candidates c JOIN reviews r ON r.candidate_id = c.candidate_id "
        "JOIN sed_model_fits f ON f.candidate_id = c.candidate_id "
        "WHERE lower(trim(COALESCE(r.event_class, ''))) = lower(trim(?)) "
        "AND f.status = 'ok' AND f.fit_version = ?",
        conn,
        params=(str(event_class), SED_MODEL_FIT_VERSION),
    )
    if targets.empty:
        raise ValueError(
            f"No current {SED_MODEL_FIT_VERSION} fits exist for event class {event_class!r}"
        )
    pool = pd.read_sql_query(
        f"SELECT DISTINCT c.candidate_id, {feature_select} "
        "FROM candidates c JOIN sed_model_fits f ON f.candidate_id = c.candidate_id "
        "WHERE f.status = 'ok' AND COALESCE(f.fit_version, '') != ? "
        "AND lower(trim(COALESCE(c.yso_class, ''))) = 'main sequence' "
        "AND c.w1 IS NOT NULL AND c.gal_b IS NOT NULL AND f.teff_k IS NOT NULL "
        "AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.candidate_id = c.candidate_id)",
        conn,
        params=(SED_MODEL_FIT_VERSION,),
    )
    if pool.empty:
        raise ValueError("No unreviewed main-sequence candidates are available as controls")

    feature_names = ["teff_k", "w1", "abs_gal_b", "reduced_chi2", "av"]
    target_values = targets[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    pool_values = pool[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    # Work in log fit-quality so a handful of poor legacy fits cannot dominate
    # Euclidean matching.
    target_values[:, 3] = np.log10(np.clip(target_values[:, 3], 1.0e-3, None))
    pool_values[:, 3] = np.log10(np.clip(pool_values[:, 3], 1.0e-3, None))
    centers = np.nanmedian(target_values, axis=0)
    target_scale = np.nanpercentile(target_values, 75, axis=0) - np.nanpercentile(target_values, 25, axis=0)
    pool_scale = np.nanpercentile(pool_values, 75, axis=0) - np.nanpercentile(pool_values, 25, axis=0)
    scales = np.where(np.isfinite(target_scale) & (target_scale > 1.0e-8), target_scale, pool_scale)
    scales = np.where(np.isfinite(scales) & (scales > 1.0e-8), scales, 1.0)
    target_values = np.where(np.isfinite(target_values), target_values, centers)
    pool_values = np.where(np.isfinite(pool_values), pool_values, centers)
    target_scaled = (target_values - centers) / scales
    pool_scaled = (pool_values - centers) / scales
    distances = np.sum((target_scaled[:, None, :] - pool_scaled[None, :, :]) ** 2, axis=2)
    rankings = np.argsort(distances, axis=1)

    selected_positions: list[int] = []
    selected_set: set[int] = set()
    cursors = np.zeros(len(targets), dtype=int)
    target_index = 0
    wanted = min(sample_size, len(pool))
    while len(selected_positions) < wanted:
        row = target_index % len(targets)
        while cursors[row] < rankings.shape[1] and int(rankings[row, cursors[row]]) in selected_set:
            cursors[row] += 1
        if cursors[row] < rankings.shape[1]:
            position = int(rankings[row, cursors[row]])
            selected_set.add(position)
            selected_positions.append(position)
            cursors[row] += 1
        target_index += 1
    return pool.iloc[selected_positions]["candidate_id"].astype(str).tolist()


def _selected_candidate_ids(
    conn: sqlite3.Connection,
    *,
    refit_all: bool,
    candidate_ids: list[str],
    limit: int | None,
    current_state: pd.DataFrame | None = None,
) -> list[str]:
    has_legacy_photometry = _table_exists(conn, "sed_photometry")
    has_v3_photometry = _table_exists(conn, "sed_measurements")
    if not has_legacy_photometry and not has_v3_photometry:
        raise ValueError("Review DB does not contain legacy or v3 SED photometry")
    if has_legacy_photometry and has_v3_photometry:
        candidate_source_sql = (
            "(SELECT candidate_id FROM sed_photometry "
            "UNION SELECT candidate_id FROM sed_measurements)"
        )
    elif has_v3_photometry:
        candidate_source_sql = "(SELECT candidate_id FROM sed_measurements)"
    else:
        candidate_source_sql = "(SELECT candidate_id FROM sed_photometry)"

    where: list[str] = []
    params: list[object] = []
    fit_columns = (
        {str(row[1]) for row in conn.execute("PRAGMA table_info(sed_model_fits)").fetchall()}
        if _table_exists(conn, "sed_model_fits")
        else set()
    )
    can_check_version = "fit_version" in fit_columns
    can_check_hashes = {"measurement_set_hash", "fit_recipe_hash"}.issubset(fit_columns)
    can_check_status = "status" in fit_columns
    clean_ids = sorted({str(value).strip() for value in candidate_ids if str(value).strip()})

    if not refit_all and can_check_version and can_check_hashes:
        provenance_columns = (
            "candidate_context_hash",
            "response_manifest_hash",
            "calibration_manifest_hash",
            "input_policy_manifest_hash",
            "model_grid_hash",
        )
        provenance_select = [
            f"f.{column}" if column in fit_columns else f"NULL AS {column}"
            for column in provenance_columns
        ]
        candidate_sql = (
            "SELECT DISTINCT p.candidate_id, f.fit_version, f.measurement_set_hash, f.fit_recipe_hash, "
            + ("f.status" if can_check_status else "NULL AS status")
            + ", "
            + ", ".join(provenance_select)
            + " "
            f"FROM {candidate_source_sql} p "
            "LEFT JOIN sed_model_fits f ON f.candidate_id = p.candidate_id "
        )
        candidate_params: list[object] = []
        if clean_ids:
            candidate_sql += f"WHERE p.candidate_id IN ({', '.join(['?'] * len(clean_ids))}) "
            candidate_params.extend(clean_ids)
        candidate_sql += "ORDER BY p.candidate_id"
        fit_state = conn.execute(candidate_sql, candidate_params).fetchall()
        current_recipe_hash = sed_fit_recipe_hash()
        state_by_candidate: dict[str, dict[str, object]] = {}
        if current_state is not None and not current_state.empty:
            state_by_candidate = {
                str(row.get("candidate_id")): dict(row)
                for _, row in current_state.iterrows()
            }
        current_candidates = {
            str(row[0])
            for row in fit_state
            if str(row[1] or "") == SED_MODEL_FIT_VERSION
            and str(row[3] or "")
            and str(row[2] or "")
        }
        current_hashes: dict[str, str] = {}
        if current_candidates:
            stored = _stored_photometry(conn, sorted(current_candidates))
            if not stored.empty and "candidate_id" in stored:
                stored["candidate_id"] = stored["candidate_id"].astype(str)
                stored = stored[stored["candidate_id"].isin(current_candidates)]
                current_hashes = {
                    str(candidate_id): sed_measurement_set_hash(group)
                    for candidate_id, group in stored.groupby("candidate_id", sort=False)
                }
        selected: list[str] = []
        for row in fit_state:
            candidate_id = str(row[0])
            expected = state_by_candidate.get(candidate_id)
            expected_measurement = (
                str(expected.get("measurement_set_hash") or "")
                if expected is not None
                else current_hashes.get(candidate_id, "")
            )
            expected_recipe = (
                str(expected.get("fit_recipe_hash") or "")
                if expected is not None
                else current_recipe_hash
            )
            stale = (
                str(row[1] or "") != SED_MODEL_FIT_VERSION
                or str(row[2] or "") != expected_measurement
                or str(row[3] or "") != expected_recipe
            )
            if can_check_status and str(row[4] or "").strip().lower() not in {
                "ok",
                "insufficient_data",
            }:
                stale = True
            if expected is not None:
                for offset, column in enumerate(provenance_columns, start=5):
                    if str(row[offset] or "") != str(expected.get(column) or ""):
                        stale = True
                        break
            if stale:
                selected.append(candidate_id)
        if limit is not None:
            if int(limit) < 1:
                return []
            selected = selected[: int(limit)]
        return selected

    if not refit_all and can_check_version:
        where.append("(f.candidate_id IS NULL OR COALESCE(f.fit_version, '') != ?)")
        params.append(SED_MODEL_FIT_VERSION)
    if clean_ids:
        where.append(f"p.candidate_id IN ({', '.join(['?'] * len(clean_ids))})")
        params.extend(clean_ids)

    sql = (
        "SELECT DISTINCT p.candidate_id "
        f"FROM {candidate_source_sql} p "
    )
    if can_check_version:
        sql += "LEFT JOIN sed_model_fits f ON f.candidate_id = p.candidate_id "
    if where:
        sql += "WHERE " + " AND ".join(where) + " "
    sql += "ORDER BY p.candidate_id"
    if limit is not None:
        if int(limit) < 1:
            return []
        sql += " LIMIT ?"
        params.append(int(limit))
    return [str(row[0]) for row in conn.execute(sql, params).fetchall()]


def _stored_photometry(conn: sqlite3.Connection, candidate_ids: list[str]) -> pd.DataFrame:
    """Load fit inputs from canonical v3 storage, with legacy fallback.

    Native measurements are one-to-many in epoch/exposure/instrument.  Reading
    the legacy object-mean table here discarded that identity (most visibly for
    exact NSC measurement rows).  Query the immutable measurement and selected
    base-normalization tables in bounded batches.  A candidate falls back to
    ``sed_photometry`` only when it has no canonical v3 rows at all.
    """
    clean_ids = list(dict.fromkeys(str(value).strip() for value in candidate_ids if str(value).strip()))
    if not clean_ids:
        return pd.DataFrame()

    canonical = _stored_v3_photometry(conn, clean_ids)
    canonical_ids = (
        set(canonical["candidate_id"].astype(str))
        if not canonical.empty and "candidate_id" in canonical.columns
        else set()
    )
    legacy_ids = [candidate_id for candidate_id in clean_ids if candidate_id not in canonical_ids]
    legacy = _stored_legacy_photometry(conn, legacy_ids)
    if canonical.empty:
        return legacy.reset_index(drop=True)
    if legacy.empty:
        return canonical.reset_index(drop=True)
    return pd.concat([canonical, legacy], ignore_index=True, sort=False)


def _id_batches(candidate_ids: list[str], size: int = 500) -> list[list[str]]:
    return [candidate_ids[start : start + size] for start in range(0, len(candidate_ids), size)]


def _stored_legacy_photometry(conn: sqlite3.Connection, candidate_ids: list[str]) -> pd.DataFrame:
    if not candidate_ids or not _table_exists(conn, "sed_photometry"):
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for batch in _id_batches(candidate_ids):
        placeholders = ", ".join(["?"] * len(batch))
        frames.append(
            pd.read_sql_query(
                f"SELECT * FROM sed_photometry WHERE candidate_id IN ({placeholders}) "
                "ORDER BY candidate_id, source, band",
                conn,
                params=batch,
            )
        )
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _stored_v3_photometry(conn: sqlite3.Connection, candidate_ids: list[str]) -> pd.DataFrame:
    if (
        not candidate_ids
        or not _table_exists(conn, "sed_measurements")
        or not _table_exists(conn, "sed_measurement_normalizations")
    ):
        return pd.DataFrame()

    select_columns = """
        m.measurement_id,
        m.candidate_id,
        m.source,
        m.catalog,
        m.release,
        m.catalog_object_id,
        m.catalog_measurement_id,
        m.exposure_id,
        m.instrument,
        m.band,
        m.epoch_mjd,
        m.native_value,
        m.native_error,
        m.native_unit,
        m.observable_kind,
        m.is_upper_limit,
        m.is_synthetic,
        m.quality_flags,
        m.quality_status,
        m.match_sep_arcsec,
        m.match_probability,
        m.response_id,
        m.calibration_id AS measurement_calibration_id,
        m.passband_fidelity,
        m.fit_policy,
        m.correlation_group,
        m.raw_measurement_json,
        m.provenance_json AS measurement_provenance_json,
        m.ingestion_version,
        n.normalization_version,
        n.flux_nu_jy,
        n.flux_nu_jy_err,
        n.flux_lambda,
        n.flux_lambda_err,
        n.lambda_l_lambda,
        n.lambda_l_lambda_err,
        n.lambda_pivot_angstrom,
        n.lambda_mean_angstrom,
        n.lambda_nominal_angstrom,
        n.lambda_reference_angstrom,
        n.lambda_isophotal_angstrom,
        n.lambda_effective_angstrom,
        n.plot_lambda_angstrom,
        n.plot_lambda_kind,
        n.response_hash,
        n.calibration_hash,
        n.normalization_hash,
        n.normalization_method,
        n.provenance_json AS normalization_provenance_json
    """
    frames: list[pd.DataFrame] = []
    for batch in _id_batches(candidate_ids):
        placeholders = ", ".join(["?"] * len(batch))
        frames.append(
            pd.read_sql_query(
                f"SELECT {select_columns} "
                "FROM sed_measurements m "
                "JOIN sed_measurement_normalizations n ON n.measurement_id = m.measurement_id "
                f"WHERE n.normalization_version = ? AND m.candidate_id IN ({placeholders}) "
                "ORDER BY m.candidate_id, m.epoch_mjd, m.source, m.band, m.measurement_id",
                conn,
                params=["sed-measurement-v3", *batch],
            )
        )
    joined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if joined.empty:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for _, item in joined.iterrows():
        observable = str(item.get("observable_kind") or "").strip().lower()
        native_unit = str(item.get("native_unit") or "").strip()
        is_quoted_fnu = observable == "quoted_fnu" or native_unit.casefold() == "jy"
        is_magnitude = observable in {"ab_mag", "vega_mag", "magnitude"} or native_unit.casefold() in {
            "mag",
            "magnitude",
        }
        if is_quoted_fnu:
            mag = None
            mag_err = None
            mag_system = "Jy"
        elif is_magnitude:
            mag = item.get("native_value")
            mag_err = item.get("native_error")
            mag_system = "AB" if observable == "ab_mag" else "Vega" if observable == "vega_mag" else ""
        else:
            mag = None
            mag_err = None
            mag_system = ""
        effective = item.get("lambda_effective_angstrom")
        if pd.isna(effective):
            effective = item.get("plot_lambda_angstrom")
        records.append(
            {
                "measurement_id": item.get("measurement_id"),
                "candidate_id": str(item.get("candidate_id") or ""),
                "source": item.get("source"),
                "catalog": item.get("catalog"),
                "catalog_release": item.get("release"),
                "release": item.get("release"),
                "catalog_object_id": item.get("catalog_object_id"),
                "source_object_id": item.get("catalog_object_id"),
                "catalog_measurement_id": item.get("catalog_measurement_id"),
                "exposure_id": item.get("exposure_id"),
                "instrument": item.get("instrument"),
                "band": item.get("band"),
                "epoch_mjd": item.get("epoch_mjd"),
                "native_value": item.get("native_value"),
                "native_error": item.get("native_error"),
                "native_unit": native_unit,
                "native_flux_unit": native_unit,
                "observable_kind": observable,
                "mag": mag,
                "mag_err": mag_err,
                "mag_system": mag_system,
                "flux_nu_jy": item.get("flux_nu_jy"),
                "flux_nu_jy_err": item.get("flux_nu_jy_err"),
                "observed_flux_nu_jy": item.get("flux_nu_jy"),
                "observed_flux_nu_jy_err": item.get("flux_nu_jy_err"),
                "flux_lambda": item.get("flux_lambda"),
                "flux_lambda_err": item.get("flux_lambda_err"),
                "lambda_l_lambda": item.get("lambda_l_lambda"),
                "lambda_l_lambda_err": item.get("lambda_l_lambda_err"),
                "lambda_eff_angstrom": effective,
                "lambda_pivot_angstrom": item.get("lambda_pivot_angstrom"),
                "lambda_mean_angstrom": item.get("lambda_mean_angstrom"),
                "lambda_nominal_angstrom": item.get("lambda_nominal_angstrom"),
                "lambda_reference_angstrom": item.get("lambda_reference_angstrom"),
                "lambda_isophotal_angstrom": item.get("lambda_isophotal_angstrom"),
                "plot_lambda_angstrom": item.get("plot_lambda_angstrom"),
                "plot_lambda_kind": item.get("plot_lambda_kind"),
                "sep_arcsec": item.get("match_sep_arcsec"),
                "match_sep_arcsec": item.get("match_sep_arcsec"),
                "match_probability": item.get("match_probability"),
                "is_synthetic": item.get("is_synthetic"),
                "is_upper_limit": item.get("is_upper_limit"),
                "quality_flags": item.get("quality_flags"),
                "quality_status": item.get("quality_status"),
                "svo_filter_id": item.get("response_id"),
                "response_id": item.get("response_id"),
                "calibration_id": item.get("measurement_calibration_id"),
                "passband_fidelity": item.get("passband_fidelity"),
                "fit_policy": item.get("fit_policy"),
                "correlation_group": item.get("correlation_group"),
                "normalization_version": item.get("normalization_version"),
                "response_hash": item.get("response_hash"),
                "calibration_hash": item.get("calibration_hash"),
                "normalization_hash": item.get("normalization_hash"),
                "normalization_method": item.get("normalization_method"),
                "raw_measurement_json": item.get("raw_measurement_json"),
                "measurement_provenance_json": item.get("measurement_provenance_json"),
                "normalization_provenance_json": item.get("normalization_provenance_json"),
                "ingestion_version": item.get("ingestion_version"),
            }
        )
    return pd.DataFrame(records)


def run(args: argparse.Namespace) -> dict[str, int]:
    review_db = Path(args.review_db).expanduser()
    if not review_db.exists():
        raise FileNotFoundError(f"Review DB does not exist: {review_db}")

    connection = sqlite3.connect(review_db) if bool(getattr(args, "dry_run", False)) else db_connect(review_db)
    with closing(connection) as conn:
        explicit_ids = {
            str(value).strip()
            for value in list(getattr(args, "candidate_id", []) or [])
            if str(value).strip()
        }
        event_class = str(getattr(args, "event_class", "") or "").strip()
        control_event_class = str(getattr(args, "control_for_event_class", "") or "").strip()
        control_sample_size = getattr(args, "control_sample_size", None)
        if control_event_class or control_sample_size is not None:
            if not control_event_class or control_sample_size is None:
                raise ValueError(
                    "--control-for-event-class and --control-sample-size must be supplied together"
                )
            if event_class or explicit_ids:
                raise ValueError(
                    "Matched-control selection cannot be combined with --event-class or --candidate-id"
                )
            requested_ids = set(
                _matched_control_candidate_ids(conn, control_event_class, int(control_sample_size))
            )
            print(
                f"Matched {len(requested_ids)} unreviewed main-sequence control(s) "
                f"to event class {control_event_class!r}"
            )
        elif event_class:
            class_ids = set(_review_event_class_candidate_ids(conn, event_class))
            requested_ids = class_ids & explicit_ids if explicit_ids else class_ids
            if not requested_ids:
                print(f"Selected 0 candidate(s): no stored reviews match event class {event_class!r}")
                return {"selected": 0, "fit_rows": 0, "curve_rows": 0, "point_rows": 0, "ok": 0}
        else:
            requested_ids = explicit_ids
        if requested_ids:
            requested_list = sorted(requested_ids)
            placeholders = ", ".join(["?"] * len(requested_list))
            candidates = pd.read_sql_query(
                f"SELECT * FROM candidates WHERE candidate_id IN ({placeholders})",
                conn,
                params=requested_list,
            )
        else:
            candidates = pd.read_sql_query("SELECT * FROM candidates", conn)
        candidates = _ensure_candidate_id(_expand_sqlite_candidate_payloads(candidates))
        if "candidate_id" not in candidates.columns:
            raise ValueError("Review DB candidates do not expose candidate_id")
        candidates["candidate_id"] = candidates["candidate_id"].astype(str)
        candidates = candidates.drop_duplicates("candidate_id", keep="last").set_index("candidate_id", drop=False)
        current_state = None
        if not bool(getattr(args, "all", False)):
            current_version_ids: set[str] = set()
            if _table_exists(conn, "sed_model_fits"):
                fit_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(sed_model_fits)").fetchall()}
                if "fit_version" in fit_columns:
                    sql = "SELECT candidate_id FROM sed_model_fits WHERE fit_version = ?"
                    fit_params: list[object] = [SED_MODEL_FIT_VERSION]
                    if requested_ids:
                        requested_list = sorted(requested_ids)
                        placeholders = ", ".join(["?"] * len(requested_list))
                        sql += f" AND candidate_id IN ({placeholders})"
                        fit_params.extend(requested_list)
                    current_version_ids = {
                        str(row[0]) for row in conn.execute(sql, fit_params).fetchall()
                    }
            if current_version_ids:
                current_list = sorted(current_version_ids)
                selection_photometry = _stored_photometry(conn, current_list)
                selection_candidates = candidates[candidates["candidate_id"].isin(current_version_ids)]
                current_state = sed_fit_input_state(
                    selection_candidates.reset_index(drop=True),
                    selection_photometry,
                )
        selected = _selected_candidate_ids(
            conn,
            refit_all=bool(getattr(args, "all", False)),
            candidate_ids=sorted(requested_ids),
            limit=getattr(args, "limit", None),
            current_state=current_state,
        )
        print(
            f"Selected {len(selected)} candidate(s) for {SED_MODEL_FIT_VERSION} "
            f"from stored SED photometry in {review_db}"
        )
        if bool(getattr(args, "dry_run", False)) or not selected:
            return {"selected": len(selected), "fit_rows": 0, "curve_rows": 0, "point_rows": 0, "ok": 0}

        batch_size = max(int(getattr(args, "batch_size", 25)), 1)
        totals = {"selected": len(selected), "fit_rows": 0, "curve_rows": 0, "point_rows": 0, "ok": 0}
        for start in range(0, len(selected), batch_size):
            batch_ids = selected[start : start + batch_size]
            missing_candidates = [candidate_id for candidate_id in batch_ids if candidate_id not in candidates.index]
            if missing_candidates:
                raise ValueError(f"Candidate payload(s) missing for: {', '.join(missing_candidates[:5])}")
            candidate_batch = candidates.loc[batch_ids].reset_index(drop=True)
            photometry_batch = _stored_photometry(conn, batch_ids)
            fits, curves, points = fit_sed_models(
                candidate_batch,
                photometry_batch,
                curve_points=max(int(getattr(args, "curve_points", 400)), 32),
                progress_callback=lambda message: print(message, flush=True),
                allow_bandpass_download=not bool(getattr(args, "no_bandpass_download", False)),
                return_points=True,
                workers=max(int(getattr(args, "fit_workers", 1)), 1),
            )
            n_fits, n_curves = upsert_sed_model_results(
                conn,
                fits,
                curves,
                points,
                replace_candidate_ids=batch_ids,
                measurement_rows=photometry_batch,
            )
            totals["fit_rows"] += int(n_fits)
            totals["curve_rows"] += int(n_curves)
            totals["point_rows"] += int(len(points))
            totals["ok"] += int((fits["status"].astype(str) == "ok").sum())
            finished = min(start + len(batch_ids), len(selected))
            print(
                f"Committed {finished}/{len(selected)} candidate(s): "
                f"{totals['ok']} ok, {totals['curve_rows']} curve rows, "
                f"{totals['point_rows']} point diagnostics",
                flush=True,
            )
    return totals


def main(argv: list[str] | None = None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
