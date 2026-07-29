from __future__ import annotations

from contextlib import closing
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malca.enrichment.sed_fit import _selected_candidate_ids
from malca.enrichment.sed_model import (
    SED_MODEL_CURVE_COLUMNS,
    SED_MODEL_FIT_COLUMNS,
    SED_MODEL_FIT_VERSION,
    SED_MODEL_POINT_COLUMNS,
    _finalize_point_rows,
    _measurement_id_for_row,
    _prepare_candidate_points,
    fit_sed_models,
    upsert_sed_model_results,
)
from malca.enrichment.synthetic_photometry import FilterResponse
from malca.review.sed_storage import (
    hash_sed_measurement_set,
    load_sed_fit_inputs,
    load_sed_fit_runs,
    load_sed_normalizations,
    prepare_canonical_sed_rows,
    store_sed_fit_results,
    store_sed_fit_run,
    store_sed_measurements,
    store_sed_normalizations,
    store_sed_point_normalizations,
)
from malca.review.store import db_connect


CANDIDATE_ID = "sed-ledger-invariants"


def _insert_candidate(conn: sqlite3.Connection, candidate_id: str = CANDIDATE_ID) -> None:
    conn.execute(
        "INSERT INTO candidates (candidate_id, payload_json, imported_at) VALUES (?, ?, ?)",
        (candidate_id, "{}", "2026-07-18T00:00:00+00:00"),
    )


def _response(
    filter_id: str,
    mag_system: str = "AB",
    *,
    changed: bool = False,
) -> FilterResponse:
    return FilterResponse(
        filter_id=filter_id,
        mag_system=mag_system,
        wavelength_angstrom=np.asarray([4000.0, 5000.0, 6000.0]),
        throughput=np.asarray([0.0, 1.0, 0.15 if changed else 0.0]),
    )


def _canonical_measurement(row: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    measurements, normalizations = prepare_canonical_sed_rows([row])
    return measurements[0], normalizations[0]


def test_response_change_creates_new_normalization_identity_and_both_commit(
    tmp_path: Path,
) -> None:
    row: dict[str, object] = {
        "candidate_id": CANDIDATE_ID,
        "source": "External Survey",
        "catalog_release": "DR1",
        "source_object_id": "external-1",
        "catalog_measurement_id": "external-1-g-mean",
        "band": "g",
        "mag": 15.0,
        "mag_err": 0.02,
        "mag_system": "AB",
        "flux_nu_jy": 3.631e-3,
        "flux_nu_jy_err": 7.262e-5,
        "lambda_pivot_angstrom": 5000.0,
        "plot_lambda_angstrom": 5000.0,
        "plot_lambda_kind": "response_pivot",
        "svo_filter_id": "External/Test.g",
        "fit_policy": "photosphere",
    }
    measurement, _ = _canonical_measurement(row)
    row["measurement_id"] = measurement["measurement_id"]

    first_response = _response("External/Test.g")
    changed_response = _response("External/Test.g", changed=True)
    first = _prepare_candidate_points(
        CANDIDATE_ID,
        {"candidate_id": CANDIDATE_ID},
        pd.DataFrame([row]),
        {("External/Test.g", "AB"): first_response},
        {},
    ).iloc[0]
    changed = _prepare_candidate_points(
        CANDIDATE_ID,
        {"candidate_id": CANDIDATE_ID},
        pd.DataFrame([row]),
        {("External/Test.g", "AB"): changed_response},
        {},
    ).iloc[0]

    assert first["response_hash"] != changed["response_hash"]
    assert first["normalization_version"] != changed["normalization_version"]
    assert first["normalization_hash"] != changed["normalization_hash"]

    # Persistence consumes the public diagnostic-point shape returned by
    # fit_sed_models, after the raw canonical flux columns are renamed.
    first_points = _finalize_point_rows(pd.DataFrame([first]))
    changed_points = _finalize_point_rows(pd.DataFrame([changed]))

    with closing(db_connect(tmp_path / "review.db")) as conn:
        _insert_candidate(conn)
        store_sed_measurements(conn, measurement)
        assert store_sed_point_normalizations(conn, first_points) == 1
        assert store_sed_point_normalizations(conn, changed_points) == 1
        stored = load_sed_normalizations(conn, CANDIDATE_ID)
        assert set(stored["normalization_version"]) == {
            first["normalization_version"],
            changed["normalization_version"],
        }
        assert set(stored["response_hash"]) == {
            first["response_hash"],
            changed["response_hash"],
        }


def test_supplied_normalization_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    row = {
        "candidate_id": CANDIDATE_ID,
        "source": "External Survey",
        "catalog_measurement_id": "external-g",
        "band": "g",
        "native_value": 15.0,
        "native_error": 0.02,
        "native_unit": "mag",
        "observable_kind": "ab_mag",
    }
    measurement, _ = _canonical_measurement(row)
    bad_normalization = {
        "measurement_id": measurement["measurement_id"],
        "normalization_version": "bandpass-response-a",
        "flux_nu_jy": 3.631e-3,
        "flux_nu_jy_err": 7.262e-5,
        "lambda_pivot_angstrom": 5000.0,
        "plot_lambda_angstrom": 5000.0,
        "plot_lambda_kind": "response_pivot",
        "response_hash": "response-a",
        "calibration_hash": "calibration-a",
        "normalization_method": "bandpass",
        "normalization_hash": "0" * 64,
    }

    with closing(db_connect(tmp_path / "review.db")) as conn:
        _insert_candidate(conn)
        store_sed_measurements(conn, measurement)
        with pytest.raises(ValueError, match="normalization_hash"):
            store_sed_normalizations(conn, bad_normalization)
        assert load_sed_normalizations(conn, CANDIDATE_ID).empty


def test_missing_or_nonpositive_diagnostic_does_not_rollback_valid_candidate(
    tmp_path: Path,
) -> None:
    rows = pd.DataFrame(
        [
            {
                "candidate_id": CANDIDATE_ID,
                "source": "External Survey",
                "catalog_measurement_id": "external-g",
                "band": "g",
                "mag": 15.0,
                "mag_err": 0.02,
                "mag_system": "AB",
                "flux_nu_jy": 3.631e-3,
                "flux_nu_jy_err": 7.262e-5,
                "lambda_pivot_angstrom": 5000.0,
                "plot_lambda_angstrom": 5000.0,
                "plot_lambda_kind": "response_pivot",
                "svo_filter_id": "External/Test.g",
                "fit_policy": "photosphere",
            },
            {
                "candidate_id": CANDIDATE_ID,
                "source": "External Survey",
                "catalog_measurement_id": "external-r-zero",
                "band": "r",
                "native_value": 0.0,
                "native_error": 0.1,
                "native_unit": "Jy",
                "observable_kind": "quoted_fnu",
                "mag_system": "Jy",
                "flux_nu_jy": 0.0,
                "flux_nu_jy_err": 0.1,
                "lambda_reference_angstrom": 6000.0,
                "plot_lambda_angstrom": 6000.0,
                "plot_lambda_kind": "mission_reference",
                "svo_filter_id": "External/Test.r",
                "fit_policy": "diagnostic_only",
            },
        ]
    )
    measurements, _ = prepare_canonical_sed_rows(rows)
    id_by_band = {str(row["band"]): str(row["measurement_id"]) for row in measurements}
    rows["measurement_id"] = rows["band"].map(id_by_band)

    def response_loader(filter_id: str, mag_system: str) -> FilterResponse:
        return _response(filter_id, mag_system)

    fits, curves, points = fit_sed_models(
        pd.DataFrame([{"candidate_id": CANDIDATE_ID}]),
        rows,
        library=object(),
        response_loader=response_loader,
        allow_bandpass_download=False,
        return_points=True,
    )
    assert fits.loc[0, "status"] == "insufficient_data"
    assert len(points) == 2

    with closing(db_connect(tmp_path / "review.db")) as conn:
        _insert_candidate(conn)
        assert upsert_sed_model_results(
            conn,
            fits,
            curves,
            points,
            measurement_rows=rows,
        ) == (1, 0)
        assert conn.execute(
            "SELECT status FROM sed_model_fits WHERE candidate_id = ?",
            (CANDIDATE_ID,),
        ).fetchone() == ("insufficient_data",)
        assert conn.execute(
            "SELECT COUNT(*) FROM sed_model_points WHERE candidate_id = ?",
            (CANDIDATE_ID,),
        ).fetchone()[0] == 2
        runs = load_sed_fit_runs(conn, CANDIDATE_ID)
        assert len(runs) == 1
        inputs = load_sed_fit_inputs(conn, str(runs.loc[0, "fit_run_id"]))
        assert id_by_band["g"] in set(inputs["measurement_id"])


@pytest.mark.parametrize("missing_id", [None, np.nan, pd.NA])
def test_alias_and_missing_measurement_ids_canonicalize_identically(
    missing_id: object,
) -> None:
    alias_row: dict[str, object] = {
        "measurement_id": missing_id,
        "candidate_id": CANDIDATE_ID,
        "source": "Gaia",
        "catalog_release": "DR3",
        "source_object_id": "123456789",
        "source_measurement_id": "123456789:G:mean",
        "band": "G",
        "mag": 12.3,
        "mag_err": 0.01,
        "mag_system": "Vega",
    }
    measurement, _ = _canonical_measurement(alias_row)
    expected = str(measurement["measurement_id"])

    assert _measurement_id_for_row(alias_row) == expected


def _valid_fit_run_fixture(
    conn: sqlite3.Connection,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    native = {
        "candidate_id": CANDIDATE_ID,
        "source": "Gaia",
        "catalog_measurement_id": "gaia-g-mean",
        "band": "G",
        "native_value": 12.3,
        "native_error": 0.01,
        "native_unit": "mag",
        "observable_kind": "vega_mag",
    }
    measurement, _ = _canonical_measurement(native)
    store_sed_measurements(conn, measurement)
    normalization = {
        "measurement_id": measurement["measurement_id"],
        "normalization_version": "bandpass-gaia-g-v1",
        "flux_nu_jy": 0.01,
        "flux_nu_jy_err": 0.0001,
        "lambda_pivot_angstrom": 6200.0,
        "plot_lambda_angstrom": 6200.0,
        "plot_lambda_kind": "response_pivot",
        "response_hash": "gaia-g-response",
        "calibration_hash": "gaia-g-calibration",
        "normalization_method": "bandpass",
    }
    store_sed_normalizations(conn, normalization)
    stored = load_sed_normalizations(conn, CANDIDATE_ID).iloc[0]
    inputs = [
        {
            "measurement_id": measurement["measurement_id"],
            "normalization_version": normalization["normalization_version"],
            "fit_role": "photosphere",
            "used": True,
            "exclusion_reason": "",
            "correlation_group": "gaia_broadband",
            "response_hash": normalization["response_hash"],
            "calibration_hash": normalization["calibration_hash"],
            "normalization_hash": stored["normalization_hash"],
        }
    ]
    run = {
        "candidate_id": CANDIDATE_ID,
        "model_family": "Castelli/Kurucz 2004",
        "fit_version": SED_MODEL_FIT_VERSION,
        "photometry_method": "bandpass_integrated",
        "extinction_law": "G23",
        "status": "ok",
        "measurement_set_hash": hash_sed_measurement_set(inputs),
        "candidate_context_hash": "candidate-context",
        "response_manifest_hash": "response-manifest",
        "calibration_manifest_hash": "calibration-manifest",
        "fit_recipe_hash": "fit-recipe",
        "model_grid_hash": "model-grid",
    }
    return run, inputs


def test_public_fit_run_store_accepts_consistent_ledger(tmp_path: Path) -> None:
    with closing(db_connect(tmp_path / "review.db")) as conn:
        _insert_candidate(conn)
        run, inputs = _valid_fit_run_fixture(conn)
        assert store_sed_fit_run(conn, run, inputs) == (1, 1)


@pytest.mark.parametrize(
    "mutation",
    [
        "normalization_hash",
        "response_hash",
        "calibration_hash",
        "input_hash",
        "fit_run_hash",
        "input_count",
        "used_input_count",
    ],
)
def test_public_fit_run_store_rejects_inconsistent_ledger(
    tmp_path: Path,
    mutation: str,
) -> None:
    with closing(db_connect(tmp_path / "review.db")) as conn:
        _insert_candidate(conn)
        run, inputs = _valid_fit_run_fixture(conn)
        if mutation in {"normalization_hash", "response_hash", "calibration_hash"}:
            inputs[0][mutation] = f"wrong-{mutation}"
        elif mutation == "input_hash":
            inputs[0]["input_hash"] = "0" * 64
        elif mutation == "fit_run_hash":
            run["fit_run_hash"] = "0" * 64
        elif mutation == "input_count":
            run["input_count"] = 99
        elif mutation == "used_input_count":
            run["used_input_count"] = 0
        else:  # pragma: no cover - keeps the mutation table exhaustive.
            raise AssertionError(mutation)

        with pytest.raises((ValueError, sqlite3.IntegrityError)):
            store_sed_fit_run(conn, run, inputs)
        assert load_sed_fit_runs(conn, CANDIDATE_ID).empty


def test_fit_failed_is_retried_but_unchanged_insufficient_data_is_not() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE sed_photometry (candidate_id TEXT, source TEXT, band TEXT)")
    conn.execute(
        "CREATE TABLE sed_model_fits ("
        "candidate_id TEXT, fit_version TEXT, status TEXT, measurement_set_hash TEXT, "
        "fit_recipe_hash TEXT, candidate_context_hash TEXT, response_manifest_hash TEXT, "
        "calibration_manifest_hash TEXT, model_grid_hash TEXT)"
    )
    candidate_ids = ("failed", "insufficient")
    conn.executemany(
        "INSERT INTO sed_photometry VALUES (?, 'Gaia', 'G')",
        [(candidate_id,) for candidate_id in candidate_ids],
    )
    conn.executemany(
        "INSERT INTO sed_model_fits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "failed",
                SED_MODEL_FIT_VERSION,
                "fit_failed",
                "measurement",
                "recipe",
                "candidate",
                "response",
                "calibration",
                "grid",
            ),
            (
                "insufficient",
                SED_MODEL_FIT_VERSION,
                "insufficient_data",
                "measurement",
                "recipe",
                "candidate",
                "response",
                "calibration",
                "grid",
            ),
        ],
    )
    current_state = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "measurement_set_hash": "measurement",
                "fit_recipe_hash": "recipe",
                "candidate_context_hash": "candidate",
                "response_manifest_hash": "response",
                "calibration_manifest_hash": "calibration",
                "model_grid_hash": "grid",
            }
            for candidate_id in candidate_ids
        ]
    )

    assert _selected_candidate_ids(
        conn,
        refit_all=False,
        candidate_ids=[],
        limit=None,
        current_state=current_state,
    ) == ["failed"]


def test_hashed_fit_without_points_is_rejected(tmp_path: Path) -> None:
    fit = {
        "candidate_id": CANDIDATE_ID,
        "model_family": "Castelli/Kurucz 2004",
        "fit_version": SED_MODEL_FIT_VERSION,
        "photometry_method": "bandpass_integrated",
        "extinction_law": "G23",
        "status": "ok",
        "n_fit_points": 1,
        "measurement_set_hash": "measurement-set",
        "candidate_context_hash": "candidate-context",
        "response_manifest_hash": "response-manifest",
        "calibration_manifest_hash": "calibration-manifest",
        "fit_recipe_hash": "fit-recipe",
        "model_grid_hash": "model-grid",
    }
    with closing(db_connect(tmp_path / "review.db")) as conn:
        _insert_candidate(conn)
        with pytest.raises(ValueError, match="point|input"):
            store_sed_fit_results(conn, pd.DataFrame([fit]), None)
        assert load_sed_fit_runs(conn, CANDIDATE_ID).empty


def test_empty_replacement_clears_complete_legacy_snapshot(tmp_path: Path) -> None:
    with closing(db_connect(tmp_path / "review.db")) as conn:
        _insert_candidate(conn)
        conn.execute(
            "INSERT INTO sed_model_fits (candidate_id, fit_version, status) VALUES (?, ?, ?)",
            (CANDIDATE_ID, "old-fit", "ok"),
        )
        conn.execute(
            "INSERT INTO sed_model_curves (candidate_id, wavelength_angstrom) VALUES (?, ?)",
            (CANDIDATE_ID, 5000.0),
        )
        conn.execute(
            "INSERT INTO sed_model_points (candidate_id, fit_version, source, band) "
            "VALUES (?, ?, ?, ?)",
            (CANDIDATE_ID, "old-fit", "Gaia", "G"),
        )
        conn.commit()

        assert upsert_sed_model_results(
            conn,
            pd.DataFrame(columns=SED_MODEL_FIT_COLUMNS),
            pd.DataFrame(columns=SED_MODEL_CURVE_COLUMNS),
            pd.DataFrame(columns=SED_MODEL_POINT_COLUMNS),
            replace_candidate_ids=[CANDIDATE_ID],
        ) == (0, 0)
        counts = {
            table: conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE candidate_id = ?",  # noqa: S608 - fixed table names.
                (CANDIDATE_ID,),
            ).fetchone()[0]
            for table in ("sed_model_fits", "sed_model_curves", "sed_model_points")
        }
        assert counts == {
            "sed_model_fits": 0,
            "sed_model_curves": 0,
            "sed_model_points": 0,
        }
