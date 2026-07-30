"""Reusable diagnostics for spatial and temporal clustering of triggered dips.

The primary analysis unit is one detector-kept dip run. Historical review
databases retain only run counts, so the companion replay script reconstructs
run intervals and their actual triggering field/camera labels before these
diagnostics are applied. Candidate-level best-dip helpers remain available for
older audits but are not an acceptable substitute for run-level systematics.

All functions are read-only unless a caller explicitly writes their returned
tables.  Randomized analyses use ``numpy.random.Generator`` and are
deterministic for a fixed ``random_state``.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

try:  # Optional acceleration for the full 335k-run schedule-block null.
    from numba import njit
except ImportError:  # pragma: no cover - the pure-Python fallback is equivalent.
    njit = None

from malca.io.lightcurve_io import (
    ASASSN_REDUCED_JD_OFFSET,
    MJD_OFFSET,
    load_lightcurve_df,
)


DEFAULT_RANDOM_STATE = 20260721
DEFAULT_CANDIDATE_COLUMNS = (
    "candidate_id",
    "source_path",
    "timescale",
    "asas_sn_id",
    "lc_path",
    "source_id",
    "gaia_id",
    "ra",
    "dec",
    "asassn_field_key",
    "camera_name_key",
    "dip_significant",
    "dip_best_morph",
    "dip_best_log_bf",
    "dip_best_delta_bic",
    "dip_best_width_param",
    "dip_best_amp",
    "dip_best_t0",
    "dip_trigger_max",
    "dip_trigger_threshold",
    "dip_max_event_prob",
    "dip_run_count",
    "stats_jd_start",
    "stats_jd_end",
    "jd_first",
    "jd_last",
    "distance_gspphot",
    "bj_r_med_photogeo",
    "bj_r_med_geo",
    "dust_distance_pc",
    "distance_pc",
    "dist_pc",
    "parallax",
    "payload_json",
)
DISTANCE_FIELDS = (
    "distance_gspphot",
    "bj_r_med_photogeo",
    "bj_r_med_geo",
    "dust_distance_pc",
    "distance_pc",
    "dist_pc",
)


def _quoted_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _readonly_connection(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Review database does not exist: {path}")
    uri = f"file:{quote(path.as_posix(), safe='/')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    rows = connection.execute(f"PRAGMA table_info({_quoted_identifier(table)})").fetchall()
    return [str(row[1]) for row in rows]


def load_review_candidates(
    db_path: str | Path,
    *,
    candidate_columns: Sequence[str] | str | None = None,
    include_reviews: bool = True,
    review_columns: Sequence[str] | str | None = None,
) -> pd.DataFrame:
    """Load candidates (and optionally reviews) from SQLite in read-only mode.

    Parameters
    ----------
    db_path
        Path to a MALCA review SQLite database.
    candidate_columns
        Candidate columns to select.  ``None`` selects the available subset of
        :data:`DEFAULT_CANDIDATE_COLUMNS`; ``"*"`` selects the full table.
    include_reviews
        Left join the ``reviews`` table when it exists.
    review_columns
        Review columns to select.  ``None`` selects all available columns and
        ``"*"`` is equivalent.  Review names that collide with candidate
        names are prefixed with ``review_``.

    Returns
    -------
    pandas.DataFrame
        One row per candidate.  The loader never creates or updates SQLite
        state, including journals.
    """

    connection = _readonly_connection(db_path)
    try:
        available = _table_columns(connection, "candidates")
        if not available:
            raise ValueError(f"No candidates table in review database: {db_path}")
        if candidate_columns is None:
            selected = [column for column in DEFAULT_CANDIDATE_COLUMNS if column in available]
        elif isinstance(candidate_columns, str) and candidate_columns == "*":
            selected = available
        else:
            requested_candidates = [candidate_columns] if isinstance(candidate_columns, str) else candidate_columns
            selected = list(dict.fromkeys(str(column) for column in requested_candidates))
            missing = sorted(set(selected) - set(available))
            if missing:
                raise KeyError(f"Candidate columns not present in database: {missing}")
        if "candidate_id" not in selected:
            selected.insert(0, "candidate_id")
        projection = ", ".join(_quoted_identifier(column) for column in selected)
        candidates = pd.read_sql_query(
            f"SELECT {projection} FROM candidates ORDER BY candidate_id",
            connection,
        )

        review_available = _table_columns(connection, "reviews") if include_reviews else []
        if review_available:
            if review_columns is None or (isinstance(review_columns, str) and review_columns == "*"):
                selected_reviews = review_available
            else:
                requested_reviews = [review_columns] if isinstance(review_columns, str) else review_columns
                selected_reviews = list(dict.fromkeys(str(column) for column in requested_reviews))
                missing = sorted(set(selected_reviews) - set(review_available))
                if missing:
                    raise KeyError(f"Review columns not present in database: {missing}")
            if "candidate_id" not in selected_reviews:
                selected_reviews.insert(0, "candidate_id")
            review_projection = ", ".join(_quoted_identifier(column) for column in selected_reviews)
            reviews = pd.read_sql_query(
                f"SELECT {review_projection} FROM reviews ORDER BY candidate_id",
                connection,
            )
            collisions = (set(candidates.columns) & set(reviews.columns)) - {"candidate_id"}
            reviews = reviews.rename(columns={name: f"review_{name}" for name in collisions})
            candidates = candidates.merge(reviews, on="candidate_id", how="left", validate="one_to_one")
        return candidates
    finally:
        connection.close()


def _finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _payload_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
    elif isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        payload = dict(decoded) if isinstance(decoded, Mapping) else {}
    else:
        return {}
    nested = payload.get("extra_json")
    if isinstance(nested, str) and nested.strip():
        try:
            nested = json.loads(nested)
        except (TypeError, ValueError, json.JSONDecodeError):
            nested = None
    if isinstance(nested, Mapping):
        for key, nested_value in nested.items():
            payload.setdefault(str(key), nested_value)
    return payload


def select_canonical_distances(candidates: pd.DataFrame) -> pd.DataFrame:
    """Select one positive physical distance per row with provenance.

    The direct scalar ``distance_gspphot`` column is authoritative when it is
    populated.  Fallbacks are extracted from ``payload_json`` (including a
    JSON-encoded ``extra_json`` member) in this order: Bailer-Jones
    photogeometric, Bailer-Jones geometric, GSP-Phot, dust input distance, and
    generic stored distance.  A naive inverse-parallax distance is not created.

    Returns a frame indexed like ``candidates`` with ``distance_pc``,
    ``distance_source``, and ``distance_from_payload``.
    """

    distances = pd.Series(np.nan, index=candidates.index, dtype=float)
    sources = pd.Series(pd.NA, index=candidates.index, dtype="string")
    from_payload = pd.Series(False, index=candidates.index, dtype=bool)

    direct_priority = (
        ("distance_gspphot", "distance_gspphot"),
        ("bj_r_med_photogeo", "bj_r_med_photogeo"),
        ("bj_r_med_geo", "bj_r_med_geo"),
        ("dust_distance_pc", "dust_distance_pc"),
        ("distance_pc", "distance_pc"),
        ("dist_pc", "dist_pc"),
    )
    for column, label in direct_priority:
        if column not in candidates.columns:
            continue
        values = pd.to_numeric(candidates[column], errors="coerce")
        use = distances.isna() & np.isfinite(values) & values.gt(0)
        distances.loc[use] = values.loc[use].astype(float)
        sources.loc[use] = label

    if "payload_json" in candidates.columns:
        fallback_priority = (
            ("bj_r_med_photogeo", "payload:bj_r_med_photogeo"),
            ("bj_r_med_geo", "payload:bj_r_med_geo"),
            ("distance_gspphot", "payload:distance_gspphot"),
            ("dust_distance_pc", "payload:dust_distance_pc"),
            ("distance_pc", "payload:distance_pc"),
            ("dist_pc", "payload:dist_pc"),
        )
        for index in candidates.index[distances.isna()]:
            payload = _payload_mapping(candidates.at[index, "payload_json"])
            for key, label in fallback_priority:
                value = _finite_positive(payload.get(key))
                if value is not None:
                    distances.at[index] = value
                    sources.at[index] = label
                    from_payload.at[index] = True
                    break
    return pd.DataFrame(
        {
            "distance_pc": distances,
            "distance_source": sources,
            "distance_from_payload": from_payload,
        },
        index=candidates.index,
    )


def _bool_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    truthy = values.astype("string").str.strip().str.lower().isin({"true", "t", "yes", "y"})
    return numeric.fillna(0).ne(0) | truthy


def _clean_identifier(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    return text.mask(text.str.lower().isin({"", "nan", "none", "null", "<na>"}))


def prepare_best_dip_events(
    candidates: pd.DataFrame,
    *,
    source_col: str | None = None,
    require_significant: bool = True,
    reduced_jd_offset: float = ASASSN_REDUCED_JD_OFFSET,
    mjd_offset: float = MJD_OFFSET,
    enforce_observed_span: bool = True,
    observation_start_col: str | None = None,
    observation_end_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare one valid best-dip event per source and a row-level QC audit.

    ``dip_best_t0`` and the observation bounds are treated as ASAS-SN reduced
    Julian dates.  ``jd_first/jd_last`` are preferred because they share the
    event detector's provenance; ``stats_jd_start/end`` are the fallback.
    Valid events receive full JD, MJD, integer MJD night, and ISO
    calendar-night columns.  Missing coordinates or distance are QC warnings,
    not event-fatal, because time-only diagnostics remain valid.

    Returns
    -------
    events, qc
        ``events`` contains only selected valid rows and has a unique
        ``source_key``.  ``qc`` has one row per input with boolean checks,
        ``qc_reasons`` (fatal), ``qc_warnings`` (analysis-specific missing
        information), and the same converted time columns.
    """

    required = {"dip_best_t0"}
    if require_significant:
        required.add("dip_significant")
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise KeyError(f"Required event columns are missing: {missing}")
    if source_col is not None and source_col not in candidates.columns:
        raise KeyError(f"source_col is missing: {source_col}")

    work = candidates.copy().reset_index(drop=False).rename(columns={"index": "input_index"})
    if source_col is None:
        for candidate in ("candidate_id", "asas_sn_id", "source_id", "lc_path", "source_path"):
            if candidate in work.columns:
                source_col = candidate
                break
    if source_col is None:
        source_values = pd.Series(pd.NA, index=work.index, dtype="string")
    else:
        source_values = _clean_identifier(work[source_col])
        if source_col in {"lc_path", "source_path"}:
            source_values = source_values.map(lambda value: Path(str(value)).stem if pd.notna(value) else pd.NA)
    work["source_key"] = source_values

    t0 = pd.to_numeric(work["dip_best_t0"], errors="coerce")
    work["dip_t0_reduced_jd"] = t0
    work["dip_jd"] = t0 + float(reduced_jd_offset)
    work["dip_mjd"] = work["dip_jd"] - float(mjd_offset)
    finite_mjd = pd.to_numeric(work["dip_mjd"], errors="coerce")
    night_mjd = np.floor(finite_mjd)
    work["dip_night_mjd"] = night_mjd.astype("Int64")
    # Passing nullable/invalid values through pandas' unit conversion can
    # raise FloatingPointError under the stricter NumPy error state used by
    # some Jupyter kernels.  Convert only finite dates in pandas' practical
    # nanosecond range; invalid event times remain auditable as missing ISO
    # labels and are rejected by the QC checks below.
    calendar_mask = np.isfinite(night_mjd) & night_mjd.between(-100_000, 100_000)
    calendar_dates = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns]")
    if bool(calendar_mask.any()):
        calendar_dates.loc[calendar_mask] = pd.to_datetime(
            night_mjd.loc[calendar_mask].astype("int64"),
            unit="D",
            origin="1858-11-17",
            errors="coerce",
        ).to_numpy()
    work["dip_night"] = calendar_dates.dt.strftime("%Y-%m-%d")

    distance = select_canonical_distances(work)
    for column in distance.columns:
        work[column] = distance[column]

    qc = pd.DataFrame(index=work.index)
    qc["input_index"] = work["input_index"]
    qc["source_key"] = work["source_key"]
    qc["candidate_id"] = work.get("candidate_id", pd.Series(pd.NA, index=work.index))
    qc["qc_has_source"] = work["source_key"].notna()
    qc["qc_triggered"] = _bool_series(work["dip_significant"]) if "dip_significant" in work else True
    qc["qc_finite_t0"] = np.isfinite(t0)
    qc["qc_reduced_jd_scale"] = qc["qc_finite_t0"] & t0.between(0.0, 1_000_000.0)

    if (observation_start_col is None) != (observation_end_col is None):
        raise ValueError("observation_start_col and observation_end_col must be supplied together")
    if observation_start_col is None:
        for start_candidate, end_candidate in (
            ("jd_first", "jd_last"),
            ("stats_jd_start", "stats_jd_end"),
        ):
            if start_candidate in work.columns and end_candidate in work.columns:
                observation_start_col, observation_end_col = start_candidate, end_candidate
                break
    if observation_start_col is not None:
        missing_bounds = {observation_start_col, str(observation_end_col)} - set(work.columns)
        if missing_bounds:
            raise KeyError(f"Observation-bound columns are missing: {sorted(missing_bounds)}")
        start = pd.to_numeric(work[observation_start_col], errors="coerce")
        end = pd.to_numeric(work[str(observation_end_col)], errors="coerce")
        observed_span_source = f"{observation_start_col}/{observation_end_col}"
    else:
        start = pd.Series(np.nan, index=work.index, dtype=float)
        end = pd.Series(np.nan, index=work.index, dtype=float)
        observed_span_source = "unavailable"
    if not isinstance(start, pd.Series):
        start = pd.Series(start, index=work.index, dtype=float)
    if not isinstance(end, pd.Series):
        end = pd.Series(end, index=work.index, dtype=float)
    qc["qc_observed_span_available"] = np.isfinite(start) & np.isfinite(end) & end.ge(start)
    qc["qc_in_observed_span"] = (~qc["qc_observed_span_available"]) | (t0.ge(start) & t0.le(end))
    qc["qc_observed_span_source"] = observed_span_source
    qc["qc_valid_time"] = qc["qc_finite_t0"] & qc["qc_reduced_jd_scale"]
    if enforce_observed_span:
        qc["qc_valid_time"] &= qc["qc_in_observed_span"]
    ra = pd.to_numeric(work.get("ra", np.nan), errors="coerce")
    dec = pd.to_numeric(work.get("dec", np.nan), errors="coerce")
    if not isinstance(ra, pd.Series):
        ra = pd.Series(ra, index=work.index, dtype=float)
    if not isinstance(dec, pd.Series):
        dec = pd.Series(dec, index=work.index, dtype=float)
    qc["qc_coordinates_valid"] = np.isfinite(ra) & np.isfinite(dec) & dec.between(-90.0, 90.0)
    qc["qc_distance_available"] = np.isfinite(work["distance_pc"]) & work["distance_pc"].gt(0)

    eligible = qc["qc_has_source"] & qc["qc_valid_time"]
    if require_significant:
        eligible &= qc["qc_triggered"]
    trigger_rank = (
        pd.to_numeric(work["dip_trigger_max"], errors="coerce")
        if "dip_trigger_max" in work
        else pd.Series(np.nan, index=work.index, dtype=float)
    )
    log_bf_rank = (
        pd.to_numeric(work["dip_best_log_bf"], errors="coerce")
        if "dip_best_log_bf" in work
        else pd.Series(np.nan, index=work.index, dtype=float)
    )
    ranking = pd.DataFrame(
        {
            "row": work.index,
            "source_key": work["source_key"],
            "eligible": eligible.astype(int),
            "trigger": trigger_rank.fillna(-np.inf),
            "log_bf": log_bf_rank.fillna(-np.inf),
        }
    )
    ranking = ranking.sort_values(
        ["source_key", "eligible", "trigger", "log_bf", "row"],
        ascending=[True, False, False, False, True],
        na_position="last",
        kind="mergesort",
    )
    selected_rows = set(
        ranking.loc[ranking["source_key"].notna()].drop_duplicates("source_key", keep="first")["row"].tolist()
    )
    qc["qc_selected_for_source"] = work.index.to_series().isin(selected_rows)
    qc["qc_valid"] = eligible & qc["qc_selected_for_source"]

    fatal_checks = (
        ("missing_source", ~qc["qc_has_source"]),
        ("not_triggered", require_significant & ~qc["qc_triggered"]),
        ("missing_t0", ~qc["qc_finite_t0"]),
        ("invalid_reduced_jd", qc["qc_finite_t0"] & ~qc["qc_reduced_jd_scale"]),
        (
            "outside_observed_span",
            enforce_observed_span
            & qc["qc_finite_t0"]
            & qc["qc_reduced_jd_scale"]
            & qc["qc_observed_span_available"]
            & ~qc["qc_in_observed_span"],
        ),
        ("duplicate_source", eligible & ~qc["qc_selected_for_source"]),
    )
    warning_checks = (
        ("observed_span_unavailable", ~qc["qc_observed_span_available"]),
        ("missing_coordinates", ~qc["qc_coordinates_valid"]),
        ("missing_distance", ~qc["qc_distance_available"]),
    )
    qc["qc_reasons"] = [
        ";".join(label for label, mask in fatal_checks if bool(mask.iloc[row])) for row in work.index
    ]
    qc["qc_warnings"] = [
        ";".join(label for label, mask in warning_checks if bool(mask.iloc[row])) for row in work.index
    ]
    for column in (
        "dip_t0_reduced_jd",
        "dip_jd",
        "dip_mjd",
        "dip_night_mjd",
        "dip_night",
        "distance_pc",
        "distance_source",
        "distance_from_payload",
    ):
        qc[column] = work[column]

    events = work.loc[qc["qc_valid"]].copy()
    events = events.sort_values(["dip_mjd", "source_key"], kind="mergesort").reset_index(drop=True)
    if events["source_key"].duplicated().any():
        raise RuntimeError("Internal error: prepared event source keys are not unique")
    qc_counts = {
        "n_input": int(len(qc)),
        "n_triggered": int(qc["qc_triggered"].sum()),
        "n_finite_t0": int(qc["qc_finite_t0"].sum()),
        "n_valid_time": int(qc["qc_valid_time"].sum()),
        "n_valid_events": int(qc["qc_valid"].sum()),
        "n_with_coordinates": int(qc["qc_coordinates_valid"].sum()),
        "n_with_distance": int(qc["qc_distance_available"].sum()),
        "n_valid_events_with_coordinates": int((qc["qc_valid"] & qc["qc_coordinates_valid"]).sum()),
        "n_valid_events_with_distance": int((qc["qc_valid"] & qc["qc_distance_available"]).sum()),
        "observation_span_source": observed_span_source,
    }
    qc.attrs["summary"] = qc_counts
    events.attrs["qc_summary"] = qc_counts
    return events, qc.reset_index(drop=True)


def prepare_triggered_dip_runs(
    run_table: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join replayed detector runs to source metadata without collapsing epochs.

    Every valid output row is one detector-kept dip run with a unique
    ``event_id``. Candidate metadata supplies sky position and distance;
    field/camera labels remain those of the triggered observations in that run.
    A replay count difference is a QC warning, not a reason to collapse or
    discard otherwise valid run intervals.
    """

    required_runs = {
        "event_id", "candidate_id", "source_key", "run_number",
        "run_start_jd", "run_end_jd", "dip_jd", "dip_mjd",
        "asassn_field_key", "camera_name_key",
    }
    missing_runs = sorted(required_runs - set(run_table.columns))
    if missing_runs:
        raise KeyError(f"Triggered-run columns are missing: {missing_runs}")
    if "candidate_id" not in candidates:
        raise KeyError("candidates must contain candidate_id")

    runs = run_table.copy().reset_index(drop=True)
    for column in ("event_id", "candidate_id", "source_key"):
        runs[column] = _clean_identifier(runs[column])
    if runs["event_id"].isna().any() or runs["event_id"].duplicated().any():
        raise ValueError("Triggered-run event_id values must be non-null and unique")
    if runs["candidate_id"].isna().any() or runs["source_key"].isna().any():
        raise ValueError("Triggered-run candidate/source identifiers must be non-null")
    if not runs["candidate_id"].eq(runs["source_key"]).all():
        raise ValueError("Triggered-run source_key must equal candidate_id")

    source_metadata = candidates.copy()
    source_metadata["candidate_id"] = _clean_identifier(source_metadata["candidate_id"])
    if source_metadata["candidate_id"].isna().any() or source_metadata["candidate_id"].duplicated().any():
        raise ValueError("Candidate metadata must have one non-null row per candidate_id")
    protected = set(runs.columns) - {"candidate_id"}
    metadata_columns = [
        column for column in source_metadata.columns
        if column == "candidate_id" or column not in protected
    ]
    events = runs.merge(
        source_metadata[metadata_columns],
        on="candidate_id",
        how="left",
        validate="many_to_one",
        indicator="_candidate_merge",
    )

    if {"trigger_fields_json", "trigger_cameras_json"} <= set(events.columns):
        def joint_modal_trigger_group(field_value: Any, camera_value: Any) -> tuple[Any, Any]:
            try:
                fields = json.loads(field_value) if isinstance(field_value, str) else []
                cameras = json.loads(camera_value) if isinstance(camera_value, str) else []
            except (TypeError, json.JSONDecodeError):
                return pd.NA, pd.NA
            counts: dict[tuple[str, str], int] = {}
            for field, camera in zip(fields, cameras):
                if field is None or camera is None:
                    continue
                pair = (str(field), str(camera))
                counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                return pd.NA, pd.NA
            return min(counts, key=lambda pair: (-counts[pair], pair))

        modal_groups = [
            joint_modal_trigger_group(field_value, camera_value)
            for field_value, camera_value in zip(
                events["trigger_fields_json"], events["trigger_cameras_json"]
            )
        ]
        has_joint_group = np.asarray([pd.notna(pair[0]) and pd.notna(pair[1]) for pair in modal_groups])
        events.loc[has_joint_group, "asassn_field_key"] = [
            pair[0] for pair, keep in zip(modal_groups, has_joint_group) if keep
        ]
        events.loc[has_joint_group, "camera_name_key"] = [
            pair[1] for pair, keep in zip(modal_groups, has_joint_group) if keep
        ]
        events["run_group_attribution"] = np.where(
            has_joint_group, "joint_modal_trigger_pair", "stored_scalar_fallback"
        )

    start_jd = pd.to_numeric(events["run_start_jd"], errors="coerce")
    end_jd = pd.to_numeric(events["run_end_jd"], errors="coerce")
    peak_jd = pd.to_numeric(events["dip_jd"], errors="coerce")
    peak_mjd = pd.to_numeric(events["dip_mjd"], errors="coerce")
    events["run_start_mjd"] = start_jd - MJD_OFFSET
    events["run_end_mjd"] = end_jd - MJD_OFFSET
    events["dip_night_mjd"] = np.floor(peak_mjd).astype("Int64")
    events["dip_night"] = pd.to_datetime(
        events["dip_night_mjd"].astype(float), unit="D", origin="1858-11-17", errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    distance = select_canonical_distances(events)
    for column in distance.columns:
        events[column] = distance[column]

    observed_counts = events.groupby("candidate_id", observed=True)["event_id"].size()
    if "dip_run_count" in source_metadata:
        expected_counts = pd.to_numeric(
            source_metadata.set_index("candidate_id")["dip_run_count"], errors="coerce"
        )
        count_matches = observed_counts.eq(expected_counts.reindex(observed_counts.index))
    else:
        count_matches = pd.Series(True, index=observed_counts.index)
    count_match_by_source = count_matches.to_dict()
    qc = pd.DataFrame(
        {
            "event_id": events["event_id"],
            "candidate_id": events["candidate_id"],
            "source_key": events["source_key"],
            "qc_candidate_joined": events["_candidate_merge"].eq("both"),
            "qc_finite_interval": np.isfinite(start_jd) & np.isfinite(end_jd) & end_jd.ge(start_jd),
            "qc_peak_in_interval": np.isfinite(peak_jd) & peak_jd.ge(start_jd) & peak_jd.le(end_jd),
            "qc_source_run_count_matches": events["candidate_id"].map(count_match_by_source).fillna(False),
        }
    )
    ra = pd.to_numeric(events.get("ra", np.nan), errors="coerce")
    dec = pd.to_numeric(events.get("dec", np.nan), errors="coerce")
    if not isinstance(ra, pd.Series):
        ra = pd.Series(ra, index=events.index, dtype=float)
    if not isinstance(dec, pd.Series):
        dec = pd.Series(dec, index=events.index, dtype=float)
    qc["qc_coordinates_valid"] = np.isfinite(ra) & np.isfinite(dec) & dec.between(-90.0, 90.0)
    qc["qc_distance_available"] = np.isfinite(events["distance_pc"]) & events["distance_pc"].gt(0)
    qc["qc_valid"] = (
        qc["qc_candidate_joined"]
        & qc["qc_finite_interval"]
        & qc["qc_peak_in_interval"]
    )
    qc["qc_reasons"] = [
        ";".join(
            label for label, condition in (
                ("candidate_not_joined", not bool(qc.loc[index, "qc_candidate_joined"])),
                ("invalid_interval", not bool(qc.loc[index, "qc_finite_interval"])),
                ("peak_outside_interval", not bool(qc.loc[index, "qc_peak_in_interval"])),
            ) if condition
        )
        for index in qc.index
    ]
    qc["qc_warnings"] = [
        ";".join(
            label for label, condition in (
                ("missing_coordinates", not bool(qc.loc[index, "qc_coordinates_valid"])),
                ("missing_distance", not bool(qc.loc[index, "qc_distance_available"])),
                ("source_run_count_mismatch", not bool(qc.loc[index, "qc_source_run_count_matches"])),
            ) if condition
        )
        for index in qc.index
    ]
    events = events.loc[qc["qc_valid"]].drop(columns="_candidate_merge").copy()
    events = events.sort_values(
        ["dip_mjd", "candidate_id", "run_number"], kind="mergesort"
    ).reset_index(drop=True)
    qc.attrs["summary"] = {
        "n_input_runs": int(len(qc)),
        "n_valid_runs": int(qc["qc_valid"].sum()),
        "n_sources": int(events["source_key"].nunique()),
        "n_source_count_mismatches": int((~count_matches).sum()),
    }
    events.attrs["qc_summary"] = dict(qc.attrs["summary"])
    return events, qc.reset_index(drop=True)


def _wilson_interval(successes: pd.Series, totals: pd.Series, z: float = 1.959963984540054) -> tuple[pd.Series, pd.Series]:
    n = totals.astype(float)
    p = successes.astype(float) / n.replace(0, np.nan)
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    lower = (center - half).clip(0.0, 1.0).mask(successes.eq(0), 0.0)
    upper = (center + half).clip(0.0, 1.0).mask(successes.eq(totals), 1.0)
    return lower, upper


def candidate_group_trigger_rates(
    candidates: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("asassn_field_key", "camera_name_key"),
    trigger_col: str = "dip_significant",
    drop_missing: bool = True,
) -> dict[str, pd.DataFrame]:
    """Return candidate-level dip trigger rates by field, camera, and their joint group."""

    if isinstance(group_cols, str):
        group_cols = (group_cols,)
    missing = [column for column in (*group_cols, trigger_col) if column not in candidates.columns]
    if missing:
        raise KeyError(f"Columns required for trigger rates are missing: {missing}")
    work = candidates[list(group_cols)].copy()
    work["_triggered"] = _bool_series(candidates[trigger_col]).astype(int)
    for column in group_cols:
        work[column] = _clean_identifier(work[column])

    def summarize(columns: list[str]) -> pd.DataFrame:
        selected = work.dropna(subset=columns) if drop_missing else work.fillna({column: "<missing>" for column in columns})
        if selected.empty:
            return pd.DataFrame(
                columns=[
                    *columns,
                    "n_candidates",
                    "n_triggered",
                    "trigger_rate",
                    "rate_ci_low",
                    "rate_ci_high",
                    "overall_trigger_rate",
                    "odds_ratio_vs_rest",
                    "pvalue_overrepresented",
                    "qvalue_overrepresented",
                ]
            )
        result = selected.groupby(columns, observed=True, sort=True)["_triggered"].agg(
            n_candidates="size", n_triggered="sum"
        ).reset_index()
        result["trigger_rate"] = result["n_triggered"] / result["n_candidates"]
        result["rate_ci_low"], result["rate_ci_high"] = _wilson_interval(
            result["n_triggered"], result["n_candidates"]
        )
        total_candidates = int(len(selected))
        total_triggered = int(selected["_triggered"].sum())
        result["overall_trigger_rate"] = total_triggered / total_candidates if total_candidates else np.nan
        odds_ratios: list[float] = []
        pvalues: list[float] = []
        for row in result.itertuples(index=False):
            group_n = int(row.n_candidates)
            group_triggered = int(row.n_triggered)
            rest_n = total_candidates - group_n
            rest_triggered = total_triggered - group_triggered
            if group_n <= 0 or rest_n <= 0:
                odds_ratios.append(np.nan)
                pvalues.append(np.nan)
                continue
            table = np.asarray(
                [
                    [group_triggered, group_n - group_triggered],
                    [rest_triggered, rest_n - rest_triggered],
                ],
                dtype=np.int64,
            )
            fisher = stats.fisher_exact(table, alternative="greater")
            odds_ratios.append(float(fisher.statistic))
            pvalues.append(float(fisher.pvalue))
        result["odds_ratio_vs_rest"] = odds_ratios
        result["pvalue_overrepresented"] = pvalues
        result["qvalue_overrepresented"] = benjamini_hochberg(result["pvalue_overrepresented"])
        return result.sort_values(["trigger_rate", "n_candidates"], ascending=[False, False], kind="mergesort").reset_index(drop=True)

    output: dict[str, pd.DataFrame] = {}
    if len(group_cols) >= 1:
        output["field"] = summarize([group_cols[0]])
    if len(group_cols) >= 2:
        output["camera"] = summarize([group_cols[1]])
        output["camera_field"] = summarize(list(group_cols[:2]))
    for column in group_cols[2:]:
        output[column] = summarize([column])
    return output


def triggered_run_group_rates(
    runs: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("asassn_field_key", "camera_name_key"),
) -> dict[str, pd.DataFrame]:
    """Summarize actual run incidence per detector observation by group."""

    if isinstance(group_cols, str):
        group_cols = (group_cols,)
    required_runs = {"source_key", "n_trigger_points", *group_cols}
    required_exposures = {"source_key", "n_observations", "n_observed_nights", *group_cols}
    missing_runs = sorted(required_runs - set(runs.columns))
    missing_exposures = sorted(required_exposures - set(exposures.columns))
    if missing_runs or missing_exposures:
        raise KeyError(
            f"Run-rate columns are missing: runs={missing_runs}, exposures={missing_exposures}"
        )

    def summarize(columns: list[str]) -> pd.DataFrame:
        run_work = runs.dropna(subset=columns).copy()
        exposure_work = exposures.dropna(subset=columns).copy()
        run_summary = run_work.groupby(columns, observed=True, sort=True).agg(
            n_runs=("event_id", "size"),
            n_trigger_points=("n_trigger_points", "sum"),
            n_triggered_sources=("source_key", "nunique"),
        ).reset_index()
        exposure_summary = exposure_work.groupby(columns, observed=True, sort=True).agg(
            n_observations=("n_observations", "sum"),
            n_observed_nights=("n_observed_nights", "sum"),
            n_available_sources=("source_key", "nunique"),
        ).reset_index()
        result = exposure_summary.merge(run_summary, on=columns, how="outer", validate="one_to_one")
        for column in (
            "n_observations", "n_observed_nights", "n_available_sources",
            "n_runs", "n_trigger_points", "n_triggered_sources",
        ):
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype("int64")
        result["runs_per_1000_observations"] = 1000.0 * result["n_runs"] / result["n_observations"].replace(0, np.nan)
        result["trigger_points_per_1000_observations"] = (
            1000.0 * result["n_trigger_points"] / result["n_observations"].replace(0, np.nan)
        )
        result["runs_per_available_source"] = result["n_runs"] / result["n_available_sources"].replace(0, np.nan)
        total_runs = int(result["n_runs"].sum())
        total_observations = int(result["n_observations"].sum())
        result["expected_runs_from_observations"] = (
            total_runs * result["n_observations"] / total_observations
            if total_observations > 0 else np.nan
        )
        result["pvalue_run_overdensity"] = stats.poisson.sf(
            result["n_runs"] - 1, result["expected_runs_from_observations"]
        )
        result["qvalue_run_overdensity"] = benjamini_hochberg(result["pvalue_run_overdensity"])
        result["rate_error_per_1000_observations"] = (
            1000.0 * np.sqrt(result["n_runs"]) / result["n_observations"].replace(0, np.nan)
        )
        return result.sort_values(
            ["runs_per_1000_observations", "n_runs"], ascending=[False, False], kind="mergesort"
        ).reset_index(drop=True)

    output: dict[str, pd.DataFrame] = {}
    if len(group_cols) >= 1:
        output["field"] = summarize([group_cols[0]])
    if len(group_cols) >= 2:
        output["camera"] = summarize([group_cols[1]])
        output["camera_field"] = summarize(list(group_cols[:2]))
    for column in group_cols[2:]:
        output[column] = summarize([column])
    return output


def benjamini_hochberg(pvalues: pd.Series | Sequence[float]) -> pd.Series:
    """Return Benjamini-Hochberg FDR q-values, preserving invalid entries."""

    series = pvalues if isinstance(pvalues, pd.Series) else pd.Series(pvalues, dtype=float)
    numeric = pd.to_numeric(series, errors="coerce")
    valid = np.isfinite(numeric) & numeric.between(0.0, 1.0)
    adjusted = pd.Series(np.nan, index=series.index, dtype=float)
    if not bool(valid.any()):
        return adjusted
    ordered = numeric.loc[valid].sort_values(kind="mergesort")
    n = len(ordered)
    raw = ordered.to_numpy(dtype=float) * n / np.arange(1, n + 1, dtype=float)
    adjusted.loc[ordered.index] = np.clip(np.minimum.accumulate(raw[::-1])[::-1], 0.0, 1.0)
    return adjusted


def summarize_nights(
    events: pd.DataFrame,
    *,
    all_candidates: pd.DataFrame | None = None,
    exposure_table: pd.DataFrame | None = None,
    night_col: str = "dip_night_mjd",
    availability_start_col: str | None = None,
    availability_end_col: str | None = None,
    exposure_good_only: bool = False,
    reduced_jd_offset: float = ASASSN_REDUCED_JD_OFFSET,
    mjd_offset: float = MJD_OFFSET,
) -> pd.DataFrame:
    """Summarize event counts, approximate exposure, rates, and night FDR.

    Raw ``exposure_table`` values (``night_mjd`` and ``n_sources``) take
    precedence.  Otherwise candidate baseline intervals provide only a coarse
    availability denominator; they do not encode seasonal gaps.  Expected
    counts are proportional to exposure, and one-sided Poisson tail p-values
    receive Benjamini-Hochberg correction.
    """

    if night_col not in events.columns:
        raise KeyError(f"Event night column is missing: {night_col}")
    nights = pd.to_numeric(events[night_col], errors="coerce").dropna().astype(int)
    if nights.empty:
        return pd.DataFrame(columns=["night_mjd", "night", "n_events", "n_exposed_sources", "event_rate", "expected_events", "pvalue", "qvalue"])
    minimum, maximum = int(nights.min()), int(nights.max())
    result = pd.DataFrame({"night_mjd": np.arange(minimum, maximum + 1, dtype=int)})
    counts = nights.value_counts().sort_index()
    result["n_events"] = result["night_mjd"].map(counts).fillna(0).astype(int)
    result["n_exposed_sources"] = np.nan
    result["exposure_source"] = "none"

    if exposure_table is not None and not exposure_table.empty:
        exposure_night_col = "night_mjd" if "night_mjd" in exposure_table else night_col
        if exposure_night_col not in exposure_table:
            raise KeyError("exposure_table must contain night_mjd (or night_col)")
        identifier = next(
            (column for column in ("source_key", "candidate_id", "source_id") if column in exposure_table),
            None,
        )
        if identifier is not None:
            raw_frame = exposure_table[[exposure_night_col, identifier] + (["n_good_observations"] if "n_good_observations" in exposure_table else [])].copy()
            raw_frame[exposure_night_col] = pd.to_numeric(raw_frame[exposure_night_col], errors="coerce")
            raw_frame[identifier] = _clean_identifier(raw_frame[identifier])
            if exposure_good_only and "n_good_observations" in raw_frame:
                raw_frame = raw_frame.loc[pd.to_numeric(raw_frame["n_good_observations"], errors="coerce").gt(0)]
            raw = raw_frame.dropna(subset=[exposure_night_col, identifier]).groupby(
                exposure_night_col, observed=True
            )[identifier].nunique()
        elif exposure_good_only and "n_good_sources" in exposure_table:
            raw_frame = exposure_table[[exposure_night_col, "n_good_sources"]].copy()
            raw_frame[exposure_night_col] = pd.to_numeric(raw_frame[exposure_night_col], errors="coerce")
            raw_frame["n_good_sources"] = pd.to_numeric(raw_frame["n_good_sources"], errors="coerce")
            raw = raw_frame.groupby(exposure_night_col, observed=True)["n_good_sources"].sum(min_count=1)
        elif "n_sources" in exposure_table:
            raw_frame = exposure_table[[exposure_night_col, "n_sources"]].copy()
            raw_frame[exposure_night_col] = pd.to_numeric(raw_frame[exposure_night_col], errors="coerce")
            raw_frame["n_sources"] = pd.to_numeric(raw_frame["n_sources"], errors="coerce")
            raw = raw_frame.groupby(exposure_night_col, observed=True)["n_sources"].sum(min_count=1)
        else:
            raise KeyError("exposure_table must contain a source identifier or n_sources")
        result["n_exposed_sources"] = result["night_mjd"].map(raw)
        result["exposure_source"] = "raw_lightcurve_scan"
    elif all_candidates is not None:
        if (availability_start_col is None) != (availability_end_col is None):
            raise ValueError("availability_start_col and availability_end_col must be supplied together")
        if availability_start_col is None:
            for start_candidate, end_candidate in (
                ("jd_first", "jd_last"),
                ("stats_jd_start", "stats_jd_end"),
            ):
                if start_candidate in all_candidates and end_candidate in all_candidates:
                    availability_start_col, availability_end_col = start_candidate, end_candidate
                    break
        if availability_start_col is None:
            raise KeyError("No candidate availability-bound columns were found")
        missing = {availability_start_col, availability_end_col} - set(all_candidates.columns)
        if missing:
            raise KeyError(f"Candidate availability columns are missing: {sorted(missing)}")
        starts = pd.to_numeric(all_candidates[availability_start_col], errors="coerce")
        ends = pd.to_numeric(all_candidates[availability_end_col], errors="coerce")
        shift = float(reduced_jd_offset) - float(mjd_offset)
        starts = np.floor(starts + shift)
        ends = np.floor(ends + shift)
        valid = (
            np.isfinite(starts)
            & np.isfinite(ends)
            & ends.ge(starts)
            & ends.ge(minimum)
            & starts.le(maximum)
        )
        start_values = starts.loc[valid].clip(lower=minimum, upper=maximum).astype(int)
        end_values = ends.loc[valid].clip(lower=minimum, upper=maximum).astype(int)
        overlaps = end_values.ge(start_values)
        delta = np.zeros(len(result) + 1, dtype=np.int64)
        np.add.at(delta, start_values.loc[overlaps].to_numpy() - minimum, 1)
        np.add.at(delta, end_values.loc[overlaps].to_numpy() - minimum + 1, -1)
        result["n_exposed_sources"] = np.cumsum(delta[:-1])
        result["exposure_source"] = "candidate_baseline_approximation"

    exposure = pd.to_numeric(result["n_exposed_sources"], errors="coerce")
    if bool((exposure > 0).any()):
        result["event_rate"] = result["n_events"] / exposure.replace(0, np.nan)
        weights = exposure.fillna(0).clip(lower=0)
        result["expected_events"] = result["n_events"].sum() * weights / weights.sum()
    else:
        result["event_rate"] = np.nan
        result["expected_events"] = result["n_events"].sum() / len(result)
    result["pvalue"] = stats.poisson.sf(result["n_events"] - 1, result["expected_events"])
    result["qvalue"] = benjamini_hochberg(result["pvalue"])
    result["night"] = pd.to_datetime(
        result["night_mjd"], unit="D", origin="1858-11-17", errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    return result.sort_values("night_mjd").reset_index(drop=True)


def _event_ids(events: pd.DataFrame) -> pd.Series:
    for column in ("event_id", "source_key", "candidate_id", "asas_sn_id"):
        if column in events.columns:
            identifiers = _clean_identifier(events[column])
            if identifiers.notna().all() and not identifiers.duplicated().any():
                return identifiers.astype(str)
    return pd.Series([f"event_{index}" for index in range(len(events))], index=events.index, dtype="string")


def build_nearby_pairs(
    events: pd.DataFrame,
    *,
    max_sep_deg: float = 5.0,
    ra_col: str = "ra",
    dec_col: str = "dec",
    time_col: str = "dip_mjd",
) -> pd.DataFrame:
    """Build nearby sky pairs with a 3-D unit-vector ``cKDTree`` query.

    The search cost depends on the number of returned neighbors and does not
    allocate an all-pairs distance matrix.
    """

    missing = [column for column in (ra_col, dec_col, time_col) if column not in events.columns]
    if missing:
        raise KeyError(f"Pair-search columns are missing: {missing}")
    if not (0 < float(max_sep_deg) <= 180):
        raise ValueError("max_sep_deg must be in (0, 180]")
    work = events.reset_index(drop=True).copy()
    work["event_id"] = _event_ids(work).to_numpy()
    ra = pd.to_numeric(work[ra_col], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(work[dec_col], errors="coerce").to_numpy(dtype=float)
    time = pd.to_numeric(work[time_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(ra) & np.isfinite(dec) & (dec >= -90) & (dec <= 90)
    positions = np.flatnonzero(valid)
    columns = [
        "event_id_i",
        "event_id_j",
        "event_row_i",
        "event_row_j",
        "angular_sep_deg",
        "time_lag_days",
        "field_i",
        "field_j",
        "same_field",
        "camera_i",
        "camera_j",
        "same_camera",
    ]
    if len(positions) < 2:
        return pd.DataFrame(columns=columns)
    ra_rad = np.deg2rad(np.mod(ra[positions], 360.0))
    dec_rad = np.deg2rad(dec[positions])
    xyz = np.column_stack((np.cos(dec_rad) * np.cos(ra_rad), np.cos(dec_rad) * np.sin(ra_rad), np.sin(dec_rad)))
    chord = 2.0 * np.sin(np.deg2rad(float(max_sep_deg)) / 2.0)
    local_pairs = cKDTree(xyz).query_pairs(chord, output_type="ndarray")
    if local_pairs.size == 0:
        return pd.DataFrame(columns=columns)
    pair_i = positions[local_pairs[:, 0]]
    pair_j = positions[local_pairs[:, 1]]
    dots = np.einsum("ij,ij->i", xyz[local_pairs[:, 0]], xyz[local_pairs[:, 1]])
    separation = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))
    result = pd.DataFrame(
        {
            "event_id_i": work.loc[pair_i, "event_id"].to_numpy(),
            "event_id_j": work.loc[pair_j, "event_id"].to_numpy(),
            "event_row_i": pair_i,
            "event_row_j": pair_j,
            "angular_sep_deg": separation,
            "time_lag_days": np.abs(time[pair_i] - time[pair_j]),
        }
    )
    for column, output in (("asassn_field_key", "field"), ("camera_name_key", "camera")):
        values = _clean_identifier(work[column]) if column in work else pd.Series(pd.NA, index=work.index)
        result[f"{output}_i"] = values.iloc[pair_i].to_numpy()
        result[f"{output}_j"] = values.iloc[pair_j].to_numpy()
        result[f"same_{output}"] = result[f"{output}_i"].notna() & result[f"{output}_i"].eq(result[f"{output}_j"])
    return result.sort_values(["angular_sep_deg", "time_lag_days", "event_id_i", "event_id_j"], kind="mergesort").reset_index(drop=True)


def build_nearby_run_pairs(
    runs: pd.DataFrame,
    *,
    max_sep_deg: float = 5.0,
    max_interval_gap_days: float = 7.0,
    ra_col: str = "ra",
    dec_col: str = "dec",
    peak_time_col: str = "dip_mjd",
    start_time_col: str = "run_start_mjd",
    end_time_col: str = "run_end_mjd",
    source_col: str = "source_key",
) -> pd.DataFrame:
    """Match runs from distinct nearby sources using their full intervals.

    Sky neighbors are found once per source. For each neighboring source pair,
    all run intervals whose edge-to-edge gap is at most
    ``max_interval_gap_days`` are returned. This avoids the combinatorial and
    scientifically incorrect same-source duplication produced by applying an
    event-level sky tree to recurring runs.
    """

    required = {
        "event_id", source_col, ra_col, dec_col, peak_time_col,
        start_time_col, end_time_col,
    }
    missing = sorted(required - set(runs.columns))
    if missing:
        raise KeyError(f"Run-pair columns are missing: {missing}")
    if not (0 < float(max_sep_deg) <= 180):
        raise ValueError("max_sep_deg must be in (0, 180]")
    if not np.isfinite(max_interval_gap_days) or max_interval_gap_days < 0:
        raise ValueError("max_interval_gap_days must be finite and non-negative")

    work = runs.reset_index(drop=True).copy()
    event_ids = _clean_identifier(work["event_id"])
    sources = _clean_identifier(work[source_col])
    if event_ids.isna().any() or event_ids.duplicated().any() or sources.isna().any():
        raise ValueError("Run event IDs must be unique and source IDs must be non-null")
    work["event_id"] = event_ids.astype(str)
    work[source_col] = sources.astype(str)
    source_coordinates = work[[source_col, ra_col, dec_col]].copy()
    source_coordinates[ra_col] = pd.to_numeric(source_coordinates[ra_col], errors="coerce")
    source_coordinates[dec_col] = pd.to_numeric(source_coordinates[dec_col], errors="coerce")
    coordinate_spread = source_coordinates.groupby(source_col, observed=True).agg(
        ra_min=(ra_col, "min"), ra_max=(ra_col, "max"),
        dec_min=(dec_col, "min"), dec_max=(dec_col, "max"),
    )
    inconsistent = (
        coordinate_spread["ra_max"].sub(coordinate_spread["ra_min"]).abs().gt(1e-10)
        | coordinate_spread["dec_max"].sub(coordinate_spread["dec_min"]).abs().gt(1e-10)
    )
    if bool(inconsistent.any()):
        raise ValueError("Sky coordinates vary between runs of the same source")
    source_table = source_coordinates.drop_duplicates(source_col, keep="first").reset_index(drop=True)
    source_ra = pd.to_numeric(source_table[ra_col], errors="coerce").to_numpy(float)
    source_dec = pd.to_numeric(source_table[dec_col], errors="coerce").to_numpy(float)
    valid_sources = np.isfinite(source_ra) & np.isfinite(source_dec) & (source_dec >= -90) & (source_dec <= 90)
    source_positions = np.flatnonzero(valid_sources)
    columns = [
        "event_id_i", "event_id_j", "event_row_i", "event_row_j",
        "source_i", "source_j", "angular_sep_deg", "time_lag_days",
        "peak_time_lag_days", "intervals_overlap", "field_i", "field_j",
        "same_field", "camera_i", "camera_j", "same_camera",
    ]
    if len(source_positions) < 2:
        return pd.DataFrame(columns=columns)
    ra_rad = np.deg2rad(np.mod(source_ra[source_positions], 360.0))
    dec_rad = np.deg2rad(source_dec[source_positions])
    xyz = np.column_stack(
        (np.cos(dec_rad) * np.cos(ra_rad), np.cos(dec_rad) * np.sin(ra_rad), np.sin(dec_rad))
    )
    chord = 2.0 * np.sin(np.deg2rad(float(max_sep_deg)) / 2.0)
    local_source_pairs = cKDTree(xyz).query_pairs(chord, output_type="ndarray")
    if local_source_pairs.size == 0:
        return pd.DataFrame(columns=columns)
    source_pair_i = source_positions[local_source_pairs[:, 0]]
    source_pair_j = source_positions[local_source_pairs[:, 1]]
    dots = np.einsum("ij,ij->i", xyz[local_source_pairs[:, 0]], xyz[local_source_pairs[:, 1]])
    source_separations = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))

    peak = pd.to_numeric(work[peak_time_col], errors="coerce").to_numpy(float)
    start = pd.to_numeric(work[start_time_col], errors="coerce").to_numpy(float)
    end = pd.to_numeric(work[end_time_col], errors="coerce").to_numpy(float)
    run_indices_by_source = {
        str(source): np.asarray(indices, dtype=int)[
            np.argsort(start[np.asarray(indices, dtype=int)], kind="mergesort")
        ]
        for source, indices in work.groupby(source_col, observed=True, sort=False).indices.items()
    }
    event_id_values = work["event_id"].astype(str).to_numpy()
    field = (
        _clean_identifier(work["asassn_field_key"]).to_numpy(dtype=object)
        if "asassn_field_key" in work
        else np.full(len(work), None, dtype=object)
    )
    camera = (
        _clean_identifier(work["camera_name_key"]).to_numpy(dtype=object)
        if "camera_name_key" in work
        else np.full(len(work), None, dtype=object)
    )
    output: dict[str, list[np.ndarray]] = {column: [] for column in columns}
    source_names = source_table[source_col].astype(str).to_numpy()
    for source_row_i, source_row_j, separation in zip(source_pair_i, source_pair_j, source_separations):
        source_i = source_names[source_row_i]
        source_j = source_names[source_row_j]
        indices_i = run_indices_by_source[source_i]
        indices_j = run_indices_by_source[source_j]
        lower = np.searchsorted(
            end[indices_j], start[indices_i] - float(max_interval_gap_days), side="left"
        )
        upper = np.searchsorted(
            start[indices_j], end[indices_i] + float(max_interval_gap_days), side="right"
        )
        match_counts = np.maximum(upper - lower, 0)
        match_count = int(match_counts.sum())
        if match_count == 0:
            continue
        local_i = np.repeat(np.arange(len(indices_i), dtype=int), match_counts)
        local_j = np.empty(match_count, dtype=int)
        cursor = 0
        for low, high in zip(lower, upper):
            width = max(int(high - low), 0)
            if width:
                local_j[cursor : cursor + width] = np.arange(low, high, dtype=int)
                cursor += width
        pair_i = indices_i[local_i]
        pair_j = indices_j[local_j]
        interval_gap = np.maximum(
            0.0,
            np.maximum(start[pair_i] - end[pair_j], start[pair_j] - end[pair_i]),
        )
        valid_gap = np.isfinite(interval_gap) & (interval_gap <= float(max_interval_gap_days))
        if not valid_gap.all():
            pair_i, pair_j, interval_gap = pair_i[valid_gap], pair_j[valid_gap], interval_gap[valid_gap]
        count = len(pair_i)
        if count == 0:
            continue
        field_i, field_j = field[pair_i], field[pair_j]
        camera_i, camera_j = camera[pair_i], camera[pair_j]
        output["event_id_i"].append(event_id_values[pair_i])
        output["event_id_j"].append(event_id_values[pair_j])
        output["event_row_i"].append(pair_i)
        output["event_row_j"].append(pair_j)
        output["source_i"].append(np.full(count, source_i, dtype=object))
        output["source_j"].append(np.full(count, source_j, dtype=object))
        output["angular_sep_deg"].append(np.full(count, float(separation)))
        output["time_lag_days"].append(interval_gap)
        output["peak_time_lag_days"].append(np.abs(peak[pair_i] - peak[pair_j]))
        output["intervals_overlap"].append(interval_gap == 0.0)
        output["field_i"].append(field_i)
        output["field_j"].append(field_j)
        output["same_field"].append(pd.notna(field_i) & pd.notna(field_j) & (field_i == field_j))
        output["camera_i"].append(camera_i)
        output["camera_j"].append(camera_j)
        output["same_camera"].append(pd.notna(camera_i) & pd.notna(camera_j) & (camera_i == camera_j))
    result = pd.DataFrame(
        {column: np.concatenate(chunks) for column, chunks in output.items()},
        columns=columns,
    ) if output["event_id_i"] else pd.DataFrame(columns=columns)
    if result.empty:
        return result
    return result.sort_values(
        ["angular_sep_deg", "time_lag_days", "peak_time_lag_days", "event_id_i", "event_id_j"],
        kind="mergesort",
    ).reset_index(drop=True)


def _permutation_groups(frame: pd.DataFrame, strata: Sequence[str]) -> list[np.ndarray]:
    if isinstance(strata, str):
        strata = (strata,)
    if not strata:
        return [np.arange(len(frame), dtype=int)]
    missing = sorted(set(strata) - set(frame.columns))
    if missing:
        raise KeyError(f"Permutation strata are missing: {missing}")
    keys = frame[list(strata)].copy()
    for column in strata:
        keys[column] = _clean_identifier(keys[column]).fillna("<missing>")
    grouped = keys.groupby(list(strata), sort=True, dropna=False).indices
    ordered_keys = sorted(
        grouped,
        key=lambda key: tuple(str(value) for value in (key if isinstance(key, tuple) else (key,))),
    )
    return [np.asarray(grouped[key], dtype=int) for key in ordered_keys]


def _cumulative_pair_counts(
    separations: np.ndarray,
    lags: np.ndarray,
    angular_thresholds: np.ndarray,
    lag_thresholds: np.ndarray,
) -> np.ndarray:
    angular_index = np.searchsorted(angular_thresholds, separations, side="left")
    lag_index = np.searchsorted(lag_thresholds, lags, side="left")
    valid = (
        np.isfinite(separations)
        & np.isfinite(lags)
        & (angular_index < len(angular_thresholds))
        & (lag_index < len(lag_thresholds))
    )
    flat = angular_index[valid] * len(lag_thresholds) + lag_index[valid]
    exact = np.bincount(flat, minlength=len(angular_thresholds) * len(lag_thresholds)).reshape(
        len(angular_thresholds), len(lag_thresholds)
    )
    return exact.cumsum(axis=0).cumsum(axis=1)


def _schedule_pair_counts_by_bins_python(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    angular_index: np.ndarray,
    schedule_assignment: np.ndarray,
    schedule_offsets: np.ndarray,
    schedule_starts: np.ndarray,
    schedule_ends: np.ndarray,
    lag_thresholds: np.ndarray,
    n_angular_bins: int,
) -> np.ndarray:
    """Count expanded-interval overlaps for a spatial source-pair graph."""

    exact = np.zeros((n_angular_bins, len(lag_thresholds)), dtype=np.int64)
    for pair_position in range(len(pair_i)):
        left_schedule = int(schedule_assignment[int(pair_i[pair_position])])
        right_schedule = int(schedule_assignment[int(pair_j[pair_position])])
        left_begin = int(schedule_offsets[left_schedule])
        left_stop = int(schedule_offsets[left_schedule + 1])
        right_begin = int(schedule_offsets[right_schedule])
        right_stop = int(schedule_offsets[right_schedule + 1])
        bin_index = int(angular_index[pair_position])
        for lag_index in range(len(lag_thresholds)):
            lag = float(lag_thresholds[lag_index])
            lower = right_begin
            upper = right_begin
            count = 0
            for left_position in range(left_begin, left_stop):
                left_start = schedule_starts[left_position]
                left_end = schedule_ends[left_position]
                while lower < right_stop and schedule_ends[lower] < left_start - lag:
                    lower += 1
                if upper < lower:
                    upper = lower
                while upper < right_stop and schedule_starts[upper] <= left_end + lag:
                    upper += 1
                if upper > lower:
                    count += upper - lower
            exact[bin_index, lag_index] += count
    for angular_position in range(1, n_angular_bins):
        exact[angular_position] += exact[angular_position - 1]
    return exact


_schedule_pair_counts_by_bins = (
    njit(nogil=True)(_schedule_pair_counts_by_bins_python)
    if njit is not None
    else _schedule_pair_counts_by_bins_python
)


def summarize_pair_excess(
    pairs: pd.DataFrame,
    events: pd.DataFrame,
    *,
    angular_bins_deg: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    lag_thresholds_days: Sequence[float] = (0.5, 1.0, 7.0),
    strata: Sequence[str] = ("asassn_field_key", "camera_name_key"),
    n_permutations: int = 1000,
    random_state: int | np.random.Generator = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Measure cumulative angular/time pair excess against a stratified null.

    Dip times are independently shuffled within the joint ``strata`` groups;
    coordinates and the nearby-pair graph remain fixed.  Reported empirical
    p-values are one-sided for an excess and are BH-adjusted across the grid.
    """

    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")
    required_pair = {"event_id_i", "event_id_j", "angular_sep_deg"}
    missing = sorted(required_pair - set(pairs.columns))
    if missing:
        raise KeyError(f"Pair summary columns are missing: {missing}")
    if "dip_mjd" not in events:
        raise KeyError("events must contain dip_mjd")
    if isinstance(strata, str):
        strata = (strata,)
    angular = np.asarray(sorted(set(float(value) for value in angular_bins_deg)), dtype=float)
    lags_max = np.asarray(sorted(set(float(value) for value in lag_thresholds_days)), dtype=float)
    if (
        angular.size == 0
        or lags_max.size == 0
        or not np.isfinite(angular).all()
        or not np.isfinite(lags_max).all()
        or np.any(angular <= 0)
        or np.any(lags_max < 0)
    ):
        raise ValueError("Angular thresholds must be positive and lag thresholds non-negative")

    work = events.reset_index(drop=True).copy()
    work["event_id"] = _event_ids(work).to_numpy()
    if work["event_id"].duplicated().any():
        raise ValueError("Event identifiers must be unique")
    id_to_row = pd.Series(work.index.to_numpy(), index=work["event_id"].astype(str))
    pair_i = pairs["event_id_i"].astype(str).map(id_to_row)
    pair_j = pairs["event_id_j"].astype(str).map(id_to_row)
    valid_pair = pair_i.notna() & pair_j.notna()
    pair_i_array = pair_i.loc[valid_pair].astype(int).to_numpy()
    pair_j_array = pair_j.loc[valid_pair].astype(int).to_numpy()
    separations = pd.to_numeric(pairs.loc[valid_pair, "angular_sep_deg"], errors="coerce").to_numpy(dtype=float)
    times = pd.to_numeric(work["dip_mjd"], errors="coerce").to_numpy(dtype=float)
    observed_lags = np.abs(times[pair_i_array] - times[pair_j_array])
    observed = _cumulative_pair_counts(separations, observed_lags, angular, lags_max)

    groups = _permutation_groups(work, strata)
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    null = np.zeros((n_permutations, len(angular), len(lags_max)), dtype=np.int64)
    for permutation in range(n_permutations):
        shuffled = times.copy()
        for indices in groups:
            finite_indices = indices[np.isfinite(times[indices])]
            if len(finite_indices) > 1:
                shuffled[finite_indices] = times[finite_indices][rng.permutation(len(finite_indices))]
        shuffled_lags = np.abs(shuffled[pair_i_array] - shuffled[pair_j_array])
        null[permutation] = _cumulative_pair_counts(separations, shuffled_lags, angular, lags_max)

    rows: list[dict[str, Any]] = []
    for angular_index, angular_max in enumerate(angular):
        for lag_index, lag_max in enumerate(lags_max):
            samples = null[:, angular_index, lag_index].astype(float)
            observed_count = int(observed[angular_index, lag_index])
            null_mean = float(samples.mean())
            null_std = float(samples.std(ddof=1)) if len(samples) > 1 else np.nan
            rows.append(
                {
                    "angular_max_deg": angular_max,
                    "lag_max_days": lag_max,
                    "observed_pairs": observed_count,
                    "null_mean_pairs": null_mean,
                    "null_std_pairs": null_std,
                    "excess_pairs": observed_count - null_mean,
                    "excess_ratio": observed_count / null_mean if null_mean > 0 else np.nan,
                    "z_score": (observed_count - null_mean) / null_std if null_std > 0 else np.nan,
                    "pvalue": (1.0 + float(np.count_nonzero(samples >= observed_count))) / (n_permutations + 1.0),
                }
            )
    result = pd.DataFrame(rows)
    result["qvalue"] = benjamini_hochberg(result["pvalue"])
    result["n_permutations"] = int(n_permutations)
    result["strata"] = "+".join(strata) if strata else "global"
    result["n_strata"] = len(groups)
    result["n_permutable_events"] = int(sum(len(group) for group in groups if len(group) > 1))
    result["n_input_pairs"] = int(len(pairs))
    result["n_matched_pairs"] = int(valid_pair.sum())
    return result


def summarize_run_pair_excess(
    runs: pd.DataFrame,
    *,
    angular_bins_deg: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    lag_thresholds_days: Sequence[float] = (0.5, 1.0, 7.0),
    strata: Sequence[str] = ("asassn_field_key", "camera_name_key"),
    source_col: str = "source_key",
    ra_col: str = "ra",
    dec_col: str = "dec",
    start_time_col: str = "run_start_mjd",
    end_time_col: str = "run_end_mjd",
    n_permutations: int = 100,
    random_state: int | np.random.Generator = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Count close run intervals against a source-schedule block null.

    The null exchanges complete run schedules between sources within modal
    field/camera strata. It therefore preserves recurrence and duration within
    each source and avoids treating hundreds of runs from one source as
    independent timestamps.
    """

    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")
    if isinstance(strata, str):
        strata = (strata,)
    required = {source_col, ra_col, dec_col, start_time_col, end_time_col, *strata}
    missing = sorted(required - set(runs.columns))
    if missing:
        raise KeyError(f"Run-pair summary columns are missing: {missing}")
    angular = np.asarray(sorted(set(float(value) for value in angular_bins_deg)), dtype=float)
    lags = np.asarray(sorted(set(float(value) for value in lag_thresholds_days)), dtype=float)
    if angular.size == 0 or lags.size == 0 or np.any(angular <= 0) or np.any(lags < 0):
        raise ValueError("Angular thresholds must be positive and lag thresholds non-negative")

    work = runs.copy().reset_index(drop=True)
    work[source_col] = _clean_identifier(work[source_col])
    work[start_time_col] = pd.to_numeric(work[start_time_col], errors="coerce")
    work[end_time_col] = pd.to_numeric(work[end_time_col], errors="coerce")
    valid = (
        work[source_col].notna()
        & np.isfinite(work[start_time_col])
        & np.isfinite(work[end_time_col])
        & work[end_time_col].ge(work[start_time_col])
    )
    work = work.loc[valid].copy()

    def modal_value(series: pd.Series) -> str:
        cleaned = _clean_identifier(series).dropna().astype(str)
        if cleaned.empty:
            return "<missing>"
        counts = cleaned.value_counts()
        return str(sorted(counts[counts.eq(counts.max())].index)[0])

    aggregations: dict[str, tuple[str, Any]] = {
        "ra": (ra_col, "first"),
        "dec": (dec_col, "first"),
    }
    for column in strata:
        aggregations[column] = (column, modal_value)
    source_table = work.groupby(source_col, observed=True, sort=True).agg(**aggregations).reset_index()
    source_ra = pd.to_numeric(source_table["ra"], errors="coerce").to_numpy(float)
    source_dec = pd.to_numeric(source_table["dec"], errors="coerce").to_numpy(float)
    valid_source = np.isfinite(source_ra) & np.isfinite(source_dec) & (source_dec >= -90) & (source_dec <= 90)
    positions = np.flatnonzero(valid_source)
    if len(positions) < 2:
        return pd.DataFrame()
    ra_rad = np.deg2rad(np.mod(source_ra[positions], 360.0))
    dec_rad = np.deg2rad(source_dec[positions])
    xyz = np.column_stack(
        (np.cos(dec_rad) * np.cos(ra_rad), np.cos(dec_rad) * np.sin(ra_rad), np.sin(dec_rad))
    )
    chord = 2.0 * np.sin(np.deg2rad(float(angular.max())) / 2.0)
    local_pairs = cKDTree(xyz).query_pairs(chord, output_type="ndarray")
    if local_pairs.size == 0:
        return pd.DataFrame()
    pair_i = positions[local_pairs[:, 0]]
    pair_j = positions[local_pairs[:, 1]]
    dots = np.einsum("ij,ij->i", xyz[local_pairs[:, 0]], xyz[local_pairs[:, 1]])
    separations = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))
    angular_index = np.searchsorted(angular, separations, side="left")
    keep_pair = angular_index < len(angular)
    pair_i, pair_j = pair_i[keep_pair], pair_j[keep_pair]
    angular_index = angular_index[keep_pair]

    source_names = source_table[source_col].astype(str).to_numpy()
    grouped = work.groupby(source_col, observed=True, sort=False).indices
    schedule_starts: list[np.ndarray] = []
    schedule_ends: list[np.ndarray] = []
    for source_name in source_names:
        indices = np.asarray(grouped.get(source_name, []), dtype=int)
        starts = pd.to_numeric(work.iloc[indices][start_time_col], errors="coerce").to_numpy(float)
        ends = pd.to_numeric(work.iloc[indices][end_time_col], errors="coerce").to_numpy(float)
        order = np.argsort(starts, kind="mergesort")
        starts, ends = starts[order], ends[order]
        if np.any(np.diff(ends) < 0):
            raise ValueError(f"Run ends are not monotonic within source {source_name}")
        schedule_starts.append(starts)
        schedule_ends.append(ends)

    schedule_offsets = np.zeros(len(source_names) + 1, dtype=np.int64)
    schedule_offsets[1:] = np.cumsum([len(values) for values in schedule_starts])
    flat_starts = np.concatenate(schedule_starts).astype(float, copy=False)
    flat_ends = np.concatenate(schedule_ends).astype(float, copy=False)

    def cumulative_counts(schedule_assignment: np.ndarray) -> np.ndarray:
        return _schedule_pair_counts_by_bins(
            pair_i.astype(np.int64, copy=False),
            pair_j.astype(np.int64, copy=False),
            angular_index.astype(np.int64, copy=False),
            schedule_assignment.astype(np.int64, copy=False),
            schedule_offsets,
            flat_starts,
            flat_ends,
            lags,
            len(angular),
        )

    identity = np.arange(len(source_table), dtype=int)
    observed = cumulative_counts(identity)
    groups = _permutation_groups(source_table, strata)
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    null = np.zeros((n_permutations, len(angular), len(lags)), dtype=np.int64)
    for permutation in range(n_permutations):
        assignment = identity.copy()
        for indices in groups:
            if len(indices) > 1:
                assignment[indices] = indices[rng.permutation(len(indices))]
        null[permutation] = cumulative_counts(assignment)

    rows: list[dict[str, Any]] = []
    for angular_position, angular_max in enumerate(angular):
        for lag_position, lag_max in enumerate(lags):
            samples = null[:, angular_position, lag_position].astype(float)
            observed_count = int(observed[angular_position, lag_position])
            null_mean = float(samples.mean())
            null_std = float(samples.std(ddof=1)) if len(samples) > 1 else np.nan
            rows.append(
                {
                    "angular_max_deg": angular_max,
                    "lag_max_days": lag_max,
                    "observed_pairs": observed_count,
                    "null_mean_pairs": null_mean,
                    "null_std_pairs": null_std,
                    "excess_pairs": observed_count - null_mean,
                    "excess_ratio": observed_count / null_mean if null_mean > 0 else np.nan,
                    "z_score": (observed_count - null_mean) / null_std if null_std > 0 else np.nan,
                    "pvalue": (1.0 + float(np.count_nonzero(samples >= observed_count))) / (n_permutations + 1.0),
                }
            )
    result = pd.DataFrame(rows)
    result["qvalue"] = benjamini_hochberg(result["pvalue"])
    result["n_permutations"] = n_permutations
    result["strata"] = "+".join(strata) if strata else "global"
    result["n_strata"] = len(groups)
    result["n_sources"] = len(source_table)
    result["n_runs"] = len(work)
    result["n_spatial_source_pairs"] = len(pair_i)
    result["permutation_unit"] = "source_run_schedule_block"
    return result


def _pearson_from_ranks(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = math.sqrt(float(np.dot(first_centered, first_centered) * np.dot(second_centered, second_centered)))
    return float(np.dot(first_centered, second_centered) / denominator) if denominator > 0 else np.nan


def _event_spearman_from_source_values(
    source_values: np.ndarray,
    source_event_counts: np.ndarray,
    source_centered_time_rank_sums: np.ndarray,
    centered_time_rank_sum_squares: float,
) -> float:
    """Exact event-level Spearman rho without expanding source labels to runs."""

    order = np.argsort(source_values, kind="mergesort")
    sorted_values = source_values[order]
    sorted_counts = source_event_counts[order]
    group_start = np.empty(len(sorted_values), dtype=bool)
    group_start[0] = True
    group_start[1:] = sorted_values[1:] != sorted_values[:-1]
    group_ids = np.cumsum(group_start, dtype=np.int64) - 1
    group_counts = np.bincount(group_ids, weights=sorted_counts)
    counts_before = np.cumsum(group_counts) - group_counts
    group_midrank = counts_before + (group_counts + 1.0) / 2.0
    source_ranks = np.empty(len(source_values), dtype=float)
    source_ranks[order] = group_midrank[group_ids]
    total_events = float(source_event_counts.sum())
    centered_source_ranks = source_ranks - (total_events + 1.0) / 2.0
    numerator = float(np.dot(centered_source_ranks, source_centered_time_rank_sums))
    source_rank_variance = float(
        np.dot(source_event_counts, centered_source_ranks * centered_source_ranks)
    )
    denominator = math.sqrt(source_rank_variance * centered_time_rank_sum_squares)
    return numerator / denominator if denominator > 0 else np.nan


def distance_time_permutation_test(
    events: pd.DataFrame,
    *,
    distance_col: str = "distance_pc",
    time_col: str = "dip_mjd",
    strata: Sequence[str] = ("asassn_field_key",),
    n_permutations: int = 1000,
    random_state: int | np.random.Generator = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Test distance--dip-time Spearman correlation with a stratified null.

    Time ranks are shuffled within field by default, preserving each field's
    seasonal time distribution.  The empirical p-value is two-sided.
    """

    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")
    if isinstance(strata, str):
        strata = (strata,)
    missing = sorted({distance_col, time_col, *strata} - set(events.columns))
    if missing:
        raise KeyError(f"Distance test columns are missing: {missing}")
    distance = pd.to_numeric(events[distance_col], errors="coerce")
    time = pd.to_numeric(events[time_col], errors="coerce")
    valid = np.isfinite(distance) & distance.gt(0) & np.isfinite(time)
    work = events.loc[valid].reset_index(drop=True)
    if len(work) < 3:
        return pd.DataFrame(
            [
                {
                    "n_events": len(work),
                    "spearman_rho": np.nan,
                    "permutation_pvalue": np.nan,
                    "asymptotic_pvalue_unstratified": np.nan,
                    "null_mean_rho": np.nan,
                    "null_std_rho": np.nan,
                    "null_q025_rho": np.nan,
                    "null_q975_rho": np.nan,
                    "z_score": np.nan,
                    "n_permutations": n_permutations,
                    "strata": "+".join(strata) if strata else "global",
                    "n_strata": 0,
                    "n_permutable_events": 0,
                }
            ]
        )
    distance_rank = stats.rankdata(pd.to_numeric(work[distance_col], errors="coerce").to_numpy(dtype=float), method="average")
    time_rank = stats.rankdata(pd.to_numeric(work[time_col], errors="coerce").to_numpy(dtype=float), method="average")
    observed = _pearson_from_ranks(distance_rank, time_rank)
    groups = _permutation_groups(work, strata)
    if not np.isfinite(observed):
        return pd.DataFrame(
            [
                {
                    "n_events": len(work),
                    "spearman_rho": np.nan,
                    "permutation_pvalue": np.nan,
                    "asymptotic_pvalue_unstratified": np.nan,
                    "null_mean_rho": np.nan,
                    "null_std_rho": np.nan,
                    "null_q025_rho": np.nan,
                    "null_q975_rho": np.nan,
                    "z_score": np.nan,
                    "n_permutations": n_permutations,
                    "strata": "+".join(strata) if strata else "global",
                    "n_strata": len(groups),
                    "n_permutable_events": int(sum(len(group) for group in groups if len(group) > 1)),
                }
            ]
        )
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    null = np.empty(n_permutations, dtype=float)
    for permutation in range(n_permutations):
        shuffled = time_rank.copy()
        for indices in groups:
            if len(indices) > 1:
                shuffled[indices] = time_rank[indices][rng.permutation(len(indices))]
        null[permutation] = _pearson_from_ranks(distance_rank, shuffled)
    null_mean = float(np.nanmean(null))
    null_std = float(np.nanstd(null, ddof=1))
    centered_observed = abs(observed - null_mean)
    centered_null = np.abs(null - null_mean)
    asymptotic = stats.spearmanr(
        pd.to_numeric(work[distance_col], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(work[time_col], errors="coerce").to_numpy(dtype=float),
    )
    return pd.DataFrame(
        [
            {
                "n_events": len(work),
                "spearman_rho": observed,
                "permutation_pvalue": (1.0 + float(np.count_nonzero(centered_null >= centered_observed))) / (n_permutations + 1.0),
                "asymptotic_pvalue_unstratified": float(asymptotic.pvalue),
                "null_mean_rho": null_mean,
                "null_std_rho": null_std,
                "null_q025_rho": float(np.nanquantile(null, 0.025)),
                "null_q975_rho": float(np.nanquantile(null, 0.975)),
                "z_score": (observed - null_mean) / null_std if null_std > 0 else np.nan,
                "n_permutations": n_permutations,
                "strata": "+".join(strata) if strata else "global",
                "n_strata": len(groups),
                "n_permutable_events": int(sum(len(group) for group in groups if len(group) > 1)),
            }
        ]
    )


def distance_time_source_block_permutation_test(
    runs: pd.DataFrame,
    *,
    distance_col: str = "distance_pc",
    time_col: str = "dip_mjd",
    source_col: str = "source_key",
    strata: Sequence[str] = ("asassn_field_key",),
    n_permutations: int = 1000,
    random_state: int | np.random.Generator = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Test run epoch versus distance while permuting whole source blocks.

    A recurring source contributes multiple runs, so event-wise shuffling would
    treat those runs as independent. This null instead permutes source distance
    labels within source-level modal strata and keeps every source's complete
    run schedule intact.
    """

    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")
    if isinstance(strata, str):
        strata = (strata,)
    required = {distance_col, time_col, source_col, *strata}
    missing = sorted(required - set(runs.columns))
    if missing:
        raise KeyError(f"Source-block distance-test columns are missing: {missing}")
    distance = pd.to_numeric(runs[distance_col], errors="coerce")
    time = pd.to_numeric(runs[time_col], errors="coerce")
    source = _clean_identifier(runs[source_col])
    valid = np.isfinite(distance) & distance.gt(0) & np.isfinite(time) & source.notna()
    work = runs.loc[valid].copy().reset_index(drop=True)
    if len(work) < 3:
        return pd.DataFrame([{"n_events": len(work), "n_sources": work[source_col].nunique(), "spearman_rho": np.nan, "permutation_pvalue": np.nan}])
    work[source_col] = _clean_identifier(work[source_col]).astype(str)
    source_distance_spread = work.groupby(source_col, observed=True)[distance_col].agg(["min", "max"])
    if bool(source_distance_spread["max"].sub(source_distance_spread["min"]).abs().gt(1e-8).any()):
        raise ValueError("Distance varies between runs of the same source")

    def modal_value(series: pd.Series) -> str:
        cleaned = _clean_identifier(series).dropna().astype(str)
        if cleaned.empty:
            return "<missing>"
        counts = cleaned.value_counts()
        return str(sorted(counts[counts.eq(counts.max())].index)[0])

    aggregations: dict[str, tuple[str, Any]] = {"distance": (distance_col, "first")}
    for column in strata:
        aggregations[column] = (column, modal_value)
    source_table = work.groupby(source_col, observed=True, sort=True).agg(**aggregations).reset_index()
    source_to_position = pd.Series(source_table.index.to_numpy(), index=source_table[source_col].astype(str))
    event_source_position = work[source_col].astype(str).map(source_to_position).astype(int).to_numpy()
    event_time = pd.to_numeric(work[time_col], errors="coerce").to_numpy(float)
    source_distances = pd.to_numeric(source_table["distance"], errors="coerce").to_numpy(float)
    centered_time_ranks = stats.rankdata(event_time).astype(float)
    centered_time_ranks -= centered_time_ranks.mean()
    centered_time_rank_sum_squares = float(np.dot(centered_time_ranks, centered_time_ranks))
    source_event_counts = np.bincount(
        event_source_position, minlength=len(source_table)
    ).astype(float)
    source_centered_time_rank_sums = np.bincount(
        event_source_position,
        weights=centered_time_ranks,
        minlength=len(source_table),
    )
    observed = _event_spearman_from_source_values(
        source_distances,
        source_event_counts,
        source_centered_time_rank_sums,
        centered_time_rank_sum_squares,
    )
    groups = _permutation_groups(source_table, strata)
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    null = np.empty(n_permutations, dtype=float)
    for permutation in range(n_permutations):
        shuffled_source_distances = source_distances.copy()
        for indices in groups:
            if len(indices) > 1:
                shuffled_source_distances[indices] = source_distances[indices][rng.permutation(len(indices))]
        null[permutation] = _event_spearman_from_source_values(
            shuffled_source_distances,
            source_event_counts,
            source_centered_time_rank_sums,
            centered_time_rank_sum_squares,
        )
    null_mean = float(np.nanmean(null))
    null_std = float(np.nanstd(null, ddof=1))
    centered_observed = abs(observed - null_mean)
    centered_null = np.abs(null - null_mean)
    return pd.DataFrame(
        [
            {
                "n_events": len(work),
                "n_sources": len(source_table),
                "spearman_rho": observed,
                "permutation_pvalue": (1.0 + float(np.count_nonzero(centered_null >= centered_observed))) / (n_permutations + 1.0),
                "null_mean_rho": null_mean,
                "null_std_rho": null_std,
                "null_q025_rho": float(np.nanquantile(null, 0.025)),
                "null_q975_rho": float(np.nanquantile(null, 0.975)),
                "z_score": (observed - null_mean) / null_std if null_std > 0 else np.nan,
                "n_permutations": n_permutations,
                "strata": "+".join(strata) if strata else "global",
                "n_strata": len(groups),
                "n_permutable_sources": int(sum(len(group) for group in groups if len(group) > 1)),
                "permutation_unit": "source_distance_block",
            }
        ]
    )


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the nearest parent containing ``pyproject.toml`` and ``malca``."""

    current = (Path.cwd() if start is None else Path(start)).expanduser().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "malca").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find MALCA repository root above {current}")


def resolve_bundle_path(path: str | Path, *, start: str | Path | None = None) -> Path:
    """Resolve a notebook-supplied run/bundle path from cwd or repository root."""

    raw = Path(path).expanduser()
    candidates = [raw, Path.cwd() / raw]
    try:
        candidates.append(find_repo_root(start) / raw)
    except FileNotFoundError:
        pass
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.exists():
            return resolved
    checked = ", ".join(str(candidate.resolve(strict=False)) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve path; checked: {checked}")


def resolve_lightcurve_path(
    row: Mapping[str, Any] | pd.Series,
    *,
    lightcurve_dir: str | Path | None = None,
    bundle_root: str | Path | None = None,
    path_col: str = "lc_path",
) -> Path | None:
    """Resolve an event row's native light curve, including stale bundle paths."""

    raw_path = row.get(path_col)
    names: list[str] = []
    if raw_path is not None and str(raw_path).strip().lower() not in {"", "nan", "none", "<na>"}:
        original = Path(str(raw_path)).expanduser()
        if original.is_file():
            return original.resolve()
        names.append(original.name)
    for key in ("source_key", "asas_sn_id", "candidate_id"):
        value = row.get(key)
        if value is not None and str(value).strip().lower() not in {"", "nan", "none", "<na>"}:
            stem = Path(str(value)).stem
            names.extend(f"{stem}{suffix}" for suffix in (".dat3", ".dat2", ".dat", ".csv"))
    directories: list[Path] = []
    if lightcurve_dir is not None:
        directories.append(Path(lightcurve_dir).expanduser())
    if bundle_root is not None:
        root = Path(bundle_root).expanduser()
        directories.extend((root / "bundle_assets/lightcurves", root / "lightcurves"))
    for directory in directories:
        for name in dict.fromkeys(names):
            candidate = (directory / name).resolve(strict=False)
            if candidate.is_file():
                return candidate
    return None


def scan_event_attribution_and_exposures(
    events: pd.DataFrame,
    *,
    lightcurve_dir: str | Path | None = None,
    bundle_root: str | Path | None = None,
    event_half_window_days: float = 1.0,
    source_col: str = "source_key",
    path_col: str = "lc_path",
    load_kwargs: Mapping[str, Any] | None = None,
    path_resolver: Callable[[pd.Series], str | Path | None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan native light curves for event attribution and actual exposures.

    This intentionally performs no caching or writes; notebooks may cache the
    two returned tables.  Attribution is one row per event/camera/field within
    the event window.  Exposures retain one row per source/night/camera/field,
    allowing callers to compute exact distinct-source denominators.
    """

    if "dip_jd" not in events or source_col not in events:
        raise KeyError(f"events must contain dip_jd and {source_col}")
    if event_half_window_days <= 0:
        raise ValueError("event_half_window_days must be positive")
    kwargs = {"apply_quality": False}
    kwargs.update(dict(load_kwargs or {}))
    attribution_rows: list[dict[str, Any]] = []
    exposure_parts: list[pd.DataFrame] = []
    for _, event in events.iterrows():
        resolved = path_resolver(event) if path_resolver is not None else resolve_lightcurve_path(
            event, lightcurve_dir=lightcurve_dir, bundle_root=bundle_root, path_col=path_col
        )
        path = Path(resolved).expanduser() if resolved is not None else None
        base = {
            source_col: event.get(source_col),
            "candidate_id": event.get("candidate_id"),
            "source_id": event.get("source_id"),
            "dip_jd": event.get("dip_jd"),
            "dip_mjd": event.get("dip_mjd"),
            "dip_night_mjd": event.get("dip_night_mjd"),
            "dip_night": event.get("dip_night"),
            "lightcurve_path": str(path) if path is not None else None,
        }
        if path is None or not path.is_file():
            attribution_rows.append({**base, "scan_status": "missing_lightcurve", "camera_name": pd.NA, "field": pd.NA, "n_window_points": 0, "n_good_window_points": 0})
            continue
        try:
            lightcurve = load_lightcurve_df(path, **kwargs)
        except Exception as exc:
            attribution_rows.append({**base, "scan_status": "load_error", "scan_error": f"{type(exc).__name__}: {exc}", "camera_name": pd.NA, "field": pd.NA, "n_window_points": 0, "n_good_window_points": 0})
            continue
        if lightcurve.empty:
            attribution_rows.append({**base, "scan_status": "empty_lightcurve", "camera_name": pd.NA, "field": pd.NA, "n_window_points": 0, "n_good_window_points": 0})
            continue
        lc = lightcurve.copy()
        lc["night_mjd"] = np.floor(pd.to_numeric(lc["mjd"], errors="coerce")).astype("Int64")
        lc[source_col] = str(event.get(source_col))
        exposure = lc.dropna(subset=["night_mjd"]).groupby(
            [source_col, "night_mjd", "camera_name", "field"], observed=True, dropna=False
        ).agg(n_observations=("jd", "size"), n_good_observations=("is_good", "sum")).reset_index()
        exposure["n_sources"] = 1
        exposure["n_good_sources"] = exposure["n_good_observations"].gt(0).astype(int)
        exposure["candidate_id"] = event.get("candidate_id")
        exposure["source_id"] = event.get("source_id")
        exposure_parts.append(exposure)
        t0 = float(event["dip_jd"])
        window = lc.loc[(pd.to_numeric(lc["jd"], errors="coerce") - t0).abs() <= event_half_window_days].copy()
        if window.empty:
            attribution_rows.append({**base, "scan_status": "no_window_points", "camera_name": pd.NA, "field": pd.NA, "n_window_points": 0, "n_good_window_points": 0})
            continue
        summary = window.groupby(["camera_name", "field"], observed=True, dropna=False).agg(
            n_window_points=("jd", "size"),
            n_good_window_points=("is_good", "sum"),
            closest_time_offset_days=("jd", lambda values: float(np.nanmin(np.abs(pd.to_numeric(values, errors="coerce") - t0)))),
        ).reset_index()
        summary["scan_status"] = "ok"
        for key, value in base.items():
            summary[key] = value
        summary = summary.sort_values(["n_good_window_points", "n_window_points", "closest_time_offset_days"], ascending=[False, False, True], kind="mergesort")
        summary["is_primary_event_group"] = False
        summary.loc[summary.index[0], "is_primary_event_group"] = True
        attribution_rows.extend(summary.to_dict(orient="records"))
    attribution = pd.DataFrame(attribution_rows)
    if exposure_parts:
        exposure_group_cols = list(
            dict.fromkeys((source_col, "candidate_id", "source_id", "night_mjd", "camera_name", "field"))
        )
        exposures = pd.concat(exposure_parts, ignore_index=True).groupby(
            exposure_group_cols,
            observed=True,
            dropna=False,
        ).agg(
            n_observations=("n_observations", "sum"),
            n_good_observations=("n_good_observations", "sum"),
            n_sources=("n_sources", "max"),
            n_good_sources=("n_good_sources", "max"),
        ).reset_index()
        exposures["night"] = pd.to_datetime(exposures["night_mjd"].astype(float), unit="D", origin="1858-11-17", errors="coerce").dt.strftime("%Y-%m-%d")
        exposures = exposures.sort_values(["night_mjd", "camera_name", "field"], kind="mergesort").reset_index(drop=True)
    else:
        exposures = pd.DataFrame(columns=[source_col, "candidate_id", "source_id", "night_mjd", "night", "camera_name", "field", "n_observations", "n_good_observations", "n_sources", "n_good_sources"])
    return attribution, exposures


__all__ = [
    "DEFAULT_CANDIDATE_COLUMNS",
    "DEFAULT_RANDOM_STATE",
    "benjamini_hochberg",
    "build_nearby_pairs",
    "build_nearby_run_pairs",
    "candidate_group_trigger_rates",
    "distance_time_permutation_test",
    "distance_time_source_block_permutation_test",
    "find_repo_root",
    "load_review_candidates",
    "prepare_best_dip_events",
    "prepare_triggered_dip_runs",
    "resolve_bundle_path",
    "resolve_lightcurve_path",
    "scan_event_attribution_and_exposures",
    "select_canonical_distances",
    "summarize_nights",
    "summarize_pair_excess",
    "summarize_run_pair_excess",
    "triggered_run_group_rates",
]
