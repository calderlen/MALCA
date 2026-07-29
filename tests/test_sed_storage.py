from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3

import pandas as pd
import numpy as np
import pytest

from malca.review.sed_storage import (
    ImmutableSedRecordError,
    ensure_sed_storage_schema,
    hash_sed_measurement_set,
    load_prepared_sed_measurements,
    load_sed_fit_inputs,
    load_sed_fit_runs,
    load_sed_measurements,
    load_sed_normalizations,
    make_sed_fit_run_hash,
    make_sed_input_hash,
    make_sed_input_manifest_hash,
    make_sed_measurement_id,
    make_sed_normalization_hash,
    migrate_legacy_sed_photometry,
    prepare_canonical_sed_rows,
    SED_STORAGE_SCHEMA_VERSION,
    sed_point_normalization_record,
    store_canonical_sed_rows,
    store_sed_fit_run,
    store_sed_measurements,
    store_sed_normalizations,
    validate_sed_storage_integrity,
)
from malca.review.store import db_connect


def _insert_candidate(conn, candidate_id: str = "sed-v3-candidate") -> None:
    conn.execute(
        "INSERT INTO candidates (candidate_id, payload_json, imported_at) VALUES (?, ?, ?)",
        (candidate_id, "{}", "2026-07-18T00:00:00+00:00"),
    )


def test_db_init_adds_sed_v3_schema_without_replacing_legacy_tables(tmp_path: Path) -> None:
    with closing(db_connect(tmp_path / "review.db")) as conn:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        assert {
            "sed_photometry",
            "sed_model_fits",
            "sed_model_curves",
            "sed_model_points",
            "sed_measurements",
            "sed_measurement_normalizations",
            "sed_fit_runs",
            "sed_fit_inputs",
        }.issubset(tables)
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert int(
            conn.execute(
                "SELECT value FROM sed_storage_meta WHERE key = 'schema_version'"
            ).fetchone()[0]
        ) == SED_STORAGE_SCHEMA_VERSION
        validate_sed_storage_integrity(conn)

        measurement_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(sed_measurements)").fetchall()
        }
        assert {
            "measurement_id",
            "catalog_measurement_id",
            "instrument",
            "exposure_id",
            "epoch_mjd",
            "native_value",
            "native_unit",
            "response_id",
            "calibration_id",
            "passband_fidelity",
            "raw_measurement_json",
        }.issubset(measurement_columns)

        normalization_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(sed_measurement_normalizations)").fetchall()
        }
        assert {
            "lambda_pivot_angstrom",
            "lambda_mean_angstrom",
            "lambda_nominal_angstrom",
            "lambda_reference_angstrom",
            "lambda_isophotal_angstrom",
            "plot_lambda_angstrom",
            "plot_lambda_kind",
            "response_hash",
            "calibration_hash",
            "normalization_hash",
        }.issubset(normalization_columns)


def test_current_sed_schema_check_is_read_only_and_validation_is_explicit(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with closing(db_connect(db_path)) as conn:
        before = conn.execute(
            "SELECT value, updated_at FROM sed_storage_meta WHERE key = 'schema_version'"
        ).fetchone()
        before_changes = conn.total_changes

        changed = ensure_sed_storage_schema(conn)

        after = conn.execute(
            "SELECT value, updated_at FROM sed_storage_meta WHERE key = 'schema_version'"
        ).fetchone()
        assert changed is False
        assert after == before
        assert conn.total_changes == before_changes
        validate_sed_storage_integrity(conn)

        legacy_fit_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(sed_model_fits)").fetchall()
        }
        assert {
            "measurement_set_hash",
            "calibration_manifest_hash",
            "fit_recipe_hash",
            "fit_run_hash",
            "fit_run_id",
        }.issubset(legacy_fit_columns)

        legacy_curve_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(sed_model_curves)").fetchall()
        }
        assert "fit_run_hash" in legacy_curve_columns
        legacy_point_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(sed_model_points)").fetchall()
        }
        assert {
            "fit_run_hash",
            "measurement_id",
            "normalization_version",
            "lambda_pivot_angstrom",
            "lambda_reference_angstrom",
            "plot_lambda_angstrom",
            "plot_lambda_kind",
            "calibration_id",
            "calibration_hash",
            "normalization_hash",
            "passband_fidelity",
            "fit_policy",
        }.issubset(legacy_point_columns)


def test_sed_schema_validation_surfaces_historical_foreign_key_damage(tmp_path: Path) -> None:
    with closing(db_connect(tmp_path / "review.db")) as conn:
        # Simulate a row written by an older client while FK enforcement was
        # disabled; current writes cannot create this state.
        conn.commit()
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            """
            INSERT INTO sed_measurements (
                measurement_id, candidate_id, source, band,
                ingestion_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "orphan-measurement",
                "missing-candidate",
                "test",
                "g",
                "historical-test",
                "2026-07-18T00:00:00+00:00",
            ),
        )
        conn.commit()
        conn.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError, match="foreign-key violation"):
            validate_sed_storage_integrity(conn)


def test_native_measurements_are_multi_epoch_idempotent_and_immutable(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    rows = [
        {
            "candidate_id": "sed-v3-candidate",
            "source": "Gaia",
            "catalog": "Gaia",
            "release": "DR3",
            "catalog_object_id": "123",
            "catalog_measurement_id": f"123:G:{epoch}",
            "band": "G",
            "epoch_mjd": epoch,
            "native_value": value,
            "native_error": 0.01,
            "native_unit": "mag",
            "observable_kind": "vega_mag",
            "is_upper_limit": "0",
            "quality_flags": {"accepted": True},
            "ingestion_version": "gaia-epoch-v1",
        }
        for epoch, value in ((59000.0, 14.2), (59030.0, 14.4))
    ]

    with closing(db_connect(db_path)) as conn:
        _insert_candidate(conn)
        assert store_sed_measurements(conn, rows) == 2
        assert store_sed_measurements(conn, rows) == 0

        loaded = load_sed_measurements(conn, "sed-v3-candidate")
        assert len(loaded) == 2
        assert loaded["measurement_id"].nunique() == 2
        assert loaded["epoch_mjd"].tolist() == [59000.0, 59030.0]
        assert loaded["is_upper_limit"].tolist() == [0, 0]

        changed = dict(rows[0], native_value=99.0)
        with pytest.raises(ImmutableSedRecordError, match="native_value"):
            store_sed_measurements(conn, changed)
        assert load_sed_measurements(conn, "sed-v3-candidate")["native_value"].tolist() == [14.2, 14.4]

        new_row = dict(rows[0], catalog_measurement_id="123:G:59060", epoch_mjd=59060.0)
        with pytest.raises(ImmutableSedRecordError, match="native_value"):
            store_sed_measurements(conn, [new_row, changed])
        # The whole batch rolls back when any immutable row conflicts.
        assert len(load_sed_measurements(conn, "sed-v3-candidate")) == 2


def test_normalizations_are_explicitly_versioned_and_join_to_native_rows(tmp_path: Path) -> None:
    native = {
        "candidate_id": "sed-v3-candidate",
        "source": "Pan-STARRS",
        "release": "DR2",
        "catalog_object_id": "ps1-1",
        "band": "g",
        "native_value": 17.2,
        "native_error": 0.02,
        "native_unit": "mag",
        "observable_kind": "ab_mag",
        "response_id": "PAN-STARRS/PS1.g",
    }
    measurement_id = make_sed_measurement_id(native)
    native["measurement_id"] = measurement_id

    with closing(db_connect(tmp_path / "review.db")) as conn:
        _insert_candidate(conn)
        store_sed_measurements(conn, native)
        normalizations = [
            {
                "measurement_id": measurement_id,
                "normalization_version": "effective-wavelength-v1",
                "flux_nu_jy": 4.70e-4,
                "lambda_effective_angstrom": 4810.0,
                "plot_lambda_angstrom": 4810.0,
                "plot_lambda_kind": "legacy_effective",
                "normalization_method": "legacy",
            },
            {
                "measurement_id": measurement_id,
                "normalization_version": "bandpass-v3",
                "flux_nu_jy": 4.79e-4,
                "lambda_pivot_angstrom": 4849.0,
                "lambda_nominal_angstrom": 4810.0,
                "plot_lambda_angstrom": 4849.0,
                "plot_lambda_kind": "pivot",
                "response_hash": "response-sha256",
                "calibration_hash": "calibration-sha256",
                "normalization_method": "synthetic-photometry-v3",
            },
        ]
        assert store_sed_normalizations(conn, normalizations) == 2
        assert store_sed_normalizations(conn, normalizations) == 0

        loaded = load_sed_normalizations(conn, "sed-v3-candidate")
        assert set(loaded["normalization_version"]) == {"effective-wavelength-v1", "bandpass-v3"}
        v3 = load_prepared_sed_measurements(
            conn,
            "sed-v3-candidate",
            normalization_version="bandpass-v3",
        )
        assert len(v3) == 1
        assert v3.loc[0, "native_value"] == pytest.approx(17.2)
        assert v3.loc[0, "normalized_flux_nu_jy"] == pytest.approx(4.79e-4)
        assert v3.loc[0, "normalized_plot_lambda_kind"] == "pivot"

        changed = dict(normalizations[1], flux_nu_jy=9.0)
        with pytest.raises(ImmutableSedRecordError, match="flux_nu_jy"):
            store_sed_normalizations(conn, changed)


def test_legacy_migration_preserves_rows_and_marks_catalog_ambiguities(tmp_path: Path) -> None:
    with closing(db_connect(tmp_path / "review.db")) as conn:
        _insert_candidate(conn)
        legacy_rows = [
            (
                "sed-v3-candidate",
                "APASS",
                "B",
                16.1,
                0.03,
                "Vega",
                4380.0,
                1.0e-16,
                2.0e-4,
                "Generic/Johnson.B",
            ),
            (
                "sed-v3-candidate",
                "NOIRLab NSC DR2",
                "g",
                17.0,
                0.02,
                "AB",
                4800.0,
                2.0e-16,
                3.0e-4,
                "CTIO/DECam.g",
            ),
            (
                "sed-v3-candidate",
                "Spitzer SEIP",
                "IRAC1",
                12.3,
                0.04,
                "AB",
                35500.0,
                3.0e-16,
                0.025,
                "Spitzer/IRAC.I1",
            ),
        ]
        conn.executemany(
            """
            INSERT INTO sed_photometry (
                candidate_id, source, band, mag, mag_err, mag_system,
                lambda_eff_angstrom, flux_lambda, flux_nu_jy, svo_filter_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            legacy_rows,
        )
        conn.commit()

        assert migrate_legacy_sed_photometry(conn) == (3, 3)
        assert migrate_legacy_sed_photometry(conn) == (0, 0)
        assert conn.execute("SELECT COUNT(*) FROM sed_photometry").fetchone()[0] == 3

        measurements = load_sed_measurements(conn, "sed-v3-candidate").set_index("source")
        assert measurements.loc["APASS", "passband_fidelity"] == "standardized_proxy"
        assert measurements.loc["APASS", "native_value"] == pytest.approx(16.1)
        assert measurements.loc["NOIRLab NSC DR2", "passband_fidelity"] == "mixed_unknown"
        # Direct-Jy missions retain their Jy value instead of the legacy pseudo-AB magnitude.
        assert measurements.loc["Spitzer SEIP", "observable_kind"] == "quoted_fnu"
        assert measurements.loc["Spitzer SEIP", "native_unit"] == "Jy"
        assert measurements.loc["Spitzer SEIP", "native_value"] == pytest.approx(0.025)
        assert "flux_nu_jy" in str(measurements.loc["Spitzer SEIP", "raw_measurement_json"])

        normalizations = load_sed_normalizations(
            conn,
            "sed-v3-candidate",
            normalization_version="legacy-stored-v1",
        )
        assert len(normalizations) == 3
        assert normalizations["lambda_pivot_angstrom"].isna().all()
        assert set(normalizations["plot_lambda_kind"]) == {"legacy_effective"}


def test_canonical_fetch_rows_write_native_and_normalized_values_atomically(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {
                "candidate_id": "sed-v3-candidate",
                "source": "Pan-STARRS",
                "catalog_release": "DR2",
                "source_object_id": "ps1-1",
                "band": "g",
                "mag": 17.2,
                "mag_err": 0.02,
                "mag_system": "AB",
                "observed_flux_nu_jy": 4.79e-4,
                "observed_flux_nu_jy_err": 8.8e-6,
                "lambda_nominal_angstrom": 4810.0,
                "lambda_pivot_angstrom": 4849.0,
                "plot_lambda_angstrom": 4849.0,
                "plot_lambda_kind": "response_pivot",
                "svo_filter_id": "PAN-STARRS/PS1.g",
                "response_hash": "ps1-g-response",
                "calibration_id": "ab-definition",
                "calibration_hash": "ab-calibration",
                "response_kind": "instrument_or_standard_response",
                "fit_policy": "photosphere",
            },
            {
                "candidate_id": "sed-v3-candidate",
                "source": "APASS",
                "band": "B",
                "mag": 16.1,
                "mag_err": 0.03,
                "mag_system": "Vega",
                "flux_nu_jy": 1.55e-3,
                "lambda_pivot_angstrom": 4380.0,
                "plot_lambda_angstrom": 4380.0,
                "plot_lambda_kind": "response_pivot",
                "response_kind": "standardized_proxy",
                "fit_policy": "photosphere_proxy",
            },
            {
                "candidate_id": "sed-v3-candidate",
                "source": "Spitzer SEIP",
                "band": "IRAC1",
                # A compatibility magnitude must not replace the native Jy value.
                "mag": 12.3,
                "mag_system": "Jy",
                "native_flux_unit": "Jy",
                "flux_nu_jy": 0.025,
                "flux_nu_jy_err": 0.001,
                "lambda_reference_angstrom": 35500.0,
                "plot_lambda_angstrom": 35500.0,
                "plot_lambda_kind": "mission_reference",
                "response_hash": "irac1-response",
                "calibration_hash": "irac1-quoted-fnu",
                "fit_policy": "diagnostic_only",
            },
        ]
    )
    prepared_measurements, prepared_normalizations = prepare_canonical_sed_rows(rows)
    assert len({row["measurement_id"] for row in prepared_measurements}) == 3
    prepared_by_source = {row["source"]: row for row in prepared_measurements}
    assert prepared_by_source["Pan-STARRS"]["observable_kind"] == "ab_mag"
    assert prepared_by_source["Pan-STARRS"]["native_value"] == pytest.approx(17.2)
    assert prepared_by_source["APASS"]["observable_kind"] == "vega_mag"
    assert prepared_by_source["APASS"]["native_value"] == pytest.approx(16.1)
    assert prepared_by_source["Spitzer SEIP"]["observable_kind"] == "quoted_fnu"
    assert prepared_by_source["Spitzer SEIP"]["native_value"] == pytest.approx(0.025)

    with closing(db_connect(tmp_path / "review.db")) as conn:
        _insert_candidate(conn)
        assert store_canonical_sed_rows(conn, rows) == (3, 3)
        assert store_canonical_sed_rows(conn, rows) == (0, 0)
        stored = load_sed_measurements(conn, "sed-v3-candidate").set_index("source")
        assert stored.loc["Pan-STARRS", "native_unit"] == "mag"
        assert stored.loc["APASS", "passband_fidelity"] == "standardized_proxy"
        assert stored.loc["Spitzer SEIP", "native_unit"] == "Jy"
        assert stored.loc["Spitzer SEIP", "fit_policy"] == "diagnostic_only"

        normalized = load_sed_normalizations(
            conn,
            "sed-v3-candidate",
            normalization_version="sed-measurement-v3",
        ).set_index("measurement_id")
        spitzer_id = prepared_by_source["Spitzer SEIP"]["measurement_id"]
        assert normalized.loc[spitzer_id, "lambda_reference_angstrom"] == pytest.approx(35500.0)
        assert normalized.loc[spitzer_id, "plot_lambda_kind"] == "mission_reference"
        assert normalized.loc[spitzer_id, "response_hash"] == "irac1-response"
        assert normalized.loc[spitzer_id, "calibration_hash"] == "irac1-quoted-fnu"

        changed = rows.copy()
        changed.loc[changed["source"] == "Pan-STARRS", "observed_flux_nu_jy"] = 9.0
        with pytest.raises(ImmutableSedRecordError, match="flux_nu_jy"):
            store_canonical_sed_rows(conn, changed)
        # The normalization conflict cannot partially replace either table.
        assert load_sed_normalizations(
            conn,
            measurement_ids=[prepared_by_source["Pan-STARRS"]["measurement_id"]],
        ).loc[0, "flux_nu_jy"] == pytest.approx(4.79e-4)


def test_fit_run_records_exact_input_versions_and_hashes_atomically(tmp_path: Path) -> None:
    native_rows = [
        {
            "candidate_id": "sed-v3-candidate",
            "source": "Gaia",
            "catalog_measurement_id": f"gaia-{band}",
            "band": band,
            "native_value": value,
            "native_unit": "mag",
            "observable_kind": "vega_mag",
        }
        for band, value in (("BP", 14.5), ("G", 14.0))
    ]
    for row in native_rows:
        row["measurement_id"] = make_sed_measurement_id(row)

    with closing(db_connect(tmp_path / "review.db")) as conn:
        _insert_candidate(conn)
        store_sed_measurements(conn, native_rows)
        normalizations = [
            {
                "measurement_id": row["measurement_id"],
                "normalization_version": "bandpass-v3",
                "flux_nu_jy": flux,
                "plot_lambda_angstrom": wavelength,
                "plot_lambda_kind": "pivot",
                "response_hash": f"response-{row['band']}",
                "calibration_hash": "gaia-calibration-v3",
            }
            for row, flux, wavelength in zip(native_rows, (0.005, 0.009), (5100.0, 6200.0))
        ]
        store_sed_normalizations(conn, normalizations)
        loaded_norms = load_sed_normalizations(conn, "sed-v3-candidate")
        inputs = [
            {
                "measurement_id": row["measurement_id"],
                "normalization_version": "bandpass-v3",
                "normalization_hash": str(
                    loaded_norms.loc[
                        loaded_norms["measurement_id"] == row["measurement_id"],
                        "normalization_hash",
                    ].iloc[0]
                ),
                "fit_role": "photosphere",
                "used": True,
                "response_hash": f"response-{row['band']}",
                "calibration_hash": "gaia-calibration-v3",
            }
            for row in native_rows
        ]
        # Hashing is independent of input ordering.
        hash_columns = [
            "measurement_id",
            "normalization_version",
            "normalization_hash",
            "response_hash",
            "calibration_hash",
        ]
        measurement_set_hash = hash_sed_measurement_set(loaded_norms[hash_columns])
        reversed_hash = hash_sed_measurement_set(loaded_norms.iloc[::-1][hash_columns])
        assert measurement_set_hash == reversed_hash

        run = {
            "candidate_id": "sed-v3-candidate",
            "model_family": "Castelli/Kurucz 2004",
            "fit_version": "ck04-bandpass-v3",
            "photometry_method": "synthetic-photometry-v3",
            "extinction_law": "F99",
            "status": "ok",
            "measurement_set_hash": measurement_set_hash,
            "response_manifest_hash": "responses-v3",
            "calibration_manifest_hash": "calibrations-v3",
            "fit_recipe_hash": "recipe-v3",
            "result_summary_json": {"teff_k": 5750.0},
        }
        assert store_sed_fit_run(conn, run, inputs) == (1, 2)
        assert store_sed_fit_run(conn, run, inputs) == (0, 0)

        runs = load_sed_fit_runs(conn, "sed-v3-candidate")
        assert len(runs) == 1
        assert runs.loc[0, "input_count"] == 2
        assert runs.loc[0, "used_input_count"] == 2
        stored_inputs = load_sed_fit_inputs(conn, str(runs.loc[0, "fit_run_id"]))
        assert len(stored_inputs) == 2
        assert set(stored_inputs["normalization_version"]) == {"bandpass-v3"}
        assert stored_inputs["normalization_hash"].notna().all()
        assert stored_inputs["input_hash"].notna().all()

        changed_inputs = [*inputs]
        changed_inputs[0] = dict(changed_inputs[0], exclusion_reason="changed after fit")
        assert store_sed_fit_run(conn, run, changed_inputs) == (1, 2)
        assert len(load_sed_fit_runs(conn, "sed-v3-candidate")) == 2
        assert len(load_sed_fit_inputs(conn, str(runs.loc[0, "fit_run_id"]))) == 2


def test_model_result_upsert_atomically_writes_legacy_and_v3_fit_ledgers(tmp_path: Path) -> None:
    from malca.enrichment.sed_model import (
        SED_MODEL_CURVE_COLUMNS,
        SED_MODEL_FIT_COLUMNS,
        SED_MODEL_FIT_VERSION,
        SED_MODEL_POINT_COLUMNS,
        load_sed_model_fits,
        load_sed_model_points,
        upsert_sed_model_results,
    )

    measurement_rows = pd.DataFrame(
        [
            {
                "candidate_id": "sed-v3-candidate",
                "source": "Pan-STARRS",
                "band": band,
                "mag": magnitude,
                "mag_err": 0.02,
                "mag_system": "AB",
                "flux_nu_jy": flux,
                "flux_nu_jy_err": flux * 0.02,
                "svo_filter_id": f"PAN-STARRS/PS1.{band}",
                "plot_lambda_angstrom": wavelength,
                "plot_lambda_kind": "response_pivot",
                "lambda_pivot_angstrom": wavelength,
                "response_hash": f"response-{band}",
                "calibration_hash": "ab-calibration",
                "normalization_hash": f"normalization-{band}",
                "normalization_version": "sed-measurement-v3",
            }
            for band, magnitude, flux, wavelength in (
                ("g", 17.2, 4.79e-4, 4849.0),
                ("r", 16.9, 6.31e-4, 6201.0),
            )
        ]
    )
    prepared_measurements, _ = prepare_canonical_sed_rows(measurement_rows)
    id_by_band = {str(row["band"]): str(row["measurement_id"]) for row in prepared_measurements}
    measurement_rows["measurement_id"] = measurement_rows["band"].map(id_by_band)

    curves = pd.DataFrame(
        [
            {
                **{column: None for column in SED_MODEL_CURVE_COLUMNS},
                "candidate_id": "sed-v3-candidate",
                "model_family": "Castelli/Kurucz 2004",
                "fit_version": SED_MODEL_FIT_VERSION,
                "fit_run_hash": None,
                "wavelength_angstrom": 5000.0,
                "flux_lambda": 1.0e-15,
            }
        ],
        columns=SED_MODEL_CURVE_COLUMNS,
    )
    fit_normalization_version = "sed-measurement-v4-bandpass:ab-calibration"
    points = pd.DataFrame(
        [
            {
                **{column: None for column in SED_MODEL_POINT_COLUMNS},
                "candidate_id": "sed-v3-candidate",
                "fit_version": SED_MODEL_FIT_VERSION,
                "fit_run_hash": "run-v3",
                "measurement_id": id_by_band[band],
                "normalization_version": fit_normalization_version,
                "source": "Pan-STARRS",
                "band": band,
                "fit_role": "photosphere",
                "used": 1,
                "exclusion_reason": "",
                "response_hash": f"response-{band}",
                "calibration_hash": "ab-calibration",
                "normalization_method": "fitter_bandpass_calibrated",
            }
            for band in ("g", "r")
        ],
        columns=SED_MODEL_POINT_COLUMNS,
    )
    expected_normalization_hashes = set()
    for idx, point in points.iterrows():
        normalization_record = sed_point_normalization_record(point)
        normalization_hash = make_sed_normalization_hash(normalization_record)
        points.at[idx, "normalization_hash"] = normalization_hash
        points.at[idx, "input_hash"] = make_sed_input_hash(points.loc[idx])
        expected_normalization_hashes.add(normalization_hash)
    measurement_set_hash = hash_sed_measurement_set(points)
    input_policy_hash = make_sed_input_manifest_hash(points)
    fit = {column: None for column in SED_MODEL_FIT_COLUMNS}
    fit.update(
        {
            "candidate_id": "sed-v3-candidate",
            "model_family": "Castelli/Kurucz 2004",
            "fit_version": SED_MODEL_FIT_VERSION,
            "photometry_method": "bandpass_integrated",
            "extinction_law": "G23",
            "teff_k": 5750.0,
            "measurement_set_hash": measurement_set_hash,
            "candidate_context_hash": "candidate-context-v3",
            "response_manifest_hash": "responses-v3",
            "calibration_manifest_hash": "calibrations-v3",
            "input_policy_manifest_hash": input_policy_hash,
            "fit_recipe_hash": "recipe-v3",
            "model_grid_hash": "model-grid-v3",
            "model_grid_provenance_json": "{}",
            "status": "ok",
            "warning": "",
        }
    )
    fit["fit_run_hash"] = make_sed_fit_run_hash(fit)
    curves["fit_run_hash"] = fit["fit_run_hash"]
    points["fit_run_hash"] = fit["fit_run_hash"]

    with closing(db_connect(tmp_path / "review.db")) as conn:
        _insert_candidate(conn)
        assert upsert_sed_model_results(
            conn,
            pd.DataFrame([fit]),
            curves,
            points,
            measurement_rows=measurement_rows,
        ) == (1, 1)
        runs = load_sed_fit_runs(conn, "sed-v3-candidate")
        assert len(runs) == 1
        assert runs.loc[0, "fit_run_hash"] == fit["fit_run_hash"]
        assert runs.loc[0, "input_count"] == 2
        inputs = load_sed_fit_inputs(conn, str(runs.loc[0, "fit_run_id"]))
        assert len(inputs) == 2
        assert set(inputs["measurement_id"]) == set(id_by_band.values())
        assert set(inputs["normalization_version"]) == {fit_normalization_version}
        assert set(inputs["normalization_hash"]) == expected_normalization_hashes
        legacy_fit = load_sed_model_fits(conn, "sed-v3-candidate")
        assert legacy_fit.loc[0, "fit_run_id"] == runs.loc[0, "fit_run_id"]
        fitted_normalizations = load_sed_normalizations(
            conn,
            "sed-v3-candidate",
            normalization_version=fit_normalization_version,
        )
        assert set(fitted_normalizations["normalization_hash"]) == expected_normalization_hashes
        legacy_snapshot = load_sed_model_points(conn, "sed-v3-candidate")
        assert legacy_snapshot["used"].tolist() == [1, 1]
        assert set(legacy_snapshot["fit_run_id"]) == {str(runs.loc[0, "fit_run_id"])}
        assert set(legacy_snapshot["input_hash"]) == set(inputs["input_hash"])

        changed_points = points.copy()
        changed_points.loc[changed_points["band"] == "g", "used"] = 0
        changed_points.loc[changed_points["band"] == "g", "exclusion_reason"] = "changed_after_fit"
        with pytest.raises(ValueError, match="input_policy_manifest_hash mismatch"):
            upsert_sed_model_results(
                conn,
                pd.DataFrame([fit]),
                curves,
                changed_points,
                measurement_rows=measurement_rows,
            )
        # The v3 manifest conflict rolls the preceding legacy replacements
        # back too.
        legacy_points = load_sed_model_points(conn, "sed-v3-candidate")
        assert legacy_points["used"].tolist() == [1, 1]
        assert set(legacy_points["exclusion_reason"].fillna("")) == {""}


def test_diagnostic_only_and_unregistered_quoted_fnu_are_never_fitted() -> None:
    from malca.enrichment.sed_model import _prepare_candidate_points
    from malca.enrichment.synthetic_photometry import FilterResponse

    response = FilterResponse(
        filter_id="Unknown/Mission.X",
        mag_system="Jy",
        wavelength_angstrom=np.asarray([4000.0, 5000.0, 6000.0]),
        throughput=np.asarray([0.0, 1.0, 0.0]),
        wavelength_ref_angstrom=5000.0,
    )
    row = {
        "candidate_id": "quoted-fnu",
        "source": "Spitzer SEIP",
        "band": "IRAC1",
        "mag_system": "Jy",
        "observable_kind": "quoted_fnu",
        "flux_nu_jy": 1.0,
        "flux_nu_jy_err": 0.1,
        "lambda_eff_angstrom": 5000.0,
        "lambda_reference_angstrom": 5000.0,
        "svo_filter_id": "Spitzer/IRAC.I1",
        "fit_policy": "diagnostic_only",
    }
    points = _prepare_candidate_points(
        "quoted-fnu",
        {"candidate_id": "quoted-fnu"},
        pd.DataFrame([row]),
        {("Spitzer/IRAC.I1", "Jy"): response},
        {},
    )
    assert len(points) == 1
    assert points.loc[0, "fit_role"] == "diagnostic"
    assert points.loc[0, "used"] == 0
    assert points.loc[0, "prediction_status"] == "unavailable"
    assert points.loc[0, "prediction_reason"] == "missing_mission_calibration"


def test_response_registry_overrides_stale_keys_and_uses_nsc_instrument() -> None:
    from malca.enrichment.sed_model import (
        _canonical_response_key,
        _prepare_candidate_points,
        sed_fit_input_state,
    )
    from malca.enrichment.synthetic_photometry import FilterResponse

    stale_apass = {
        "candidate_id": "canonical-key",
        "source": "APASS",
        "band": "V",
        "mag": 12.0,
        "mag_err": 0.03,
        "mag_system": "AB",
        "lambda_eff_angstrom": 5450.0,
        "svo_filter_id": "Wrong/Raw.Filter",
    }
    assert _canonical_response_key(stale_apass) == ("Generic/Johnson.V", "Vega")
    assert _canonical_response_key(
        {
            "source": "NOIRLab NSC DR2",
            "band": "g",
            "instrument": "DECam",
            "svo_filter_id": "Wrong/Raw.Filter",
            "mag_system": "Vega",
        }
    ) == ("CTIO/DECam.g", "AB")
    # An object-level NSC mean has no exact physical response.  A stale DECam
    # label must not silently turn it into exact DECam photometry.
    assert _canonical_response_key(
        {
            "source": "NOIRLab NSC DR2",
            "band": "g",
            "svo_filter_id": "CTIO/DECam.g",
            "mag_system": "AB",
        }
    ) == ("", "AB")

    calls: list[tuple[str, str]] = []

    def response_loader(filter_id: str, mag_system: str) -> FilterResponse:
        calls.append((filter_id, mag_system))
        return FilterResponse(
            filter_id=filter_id,
            mag_system=mag_system,
            zero_point_jy=3636.0,
            wavelength_angstrom=np.asarray([5000.0, 5450.0, 5900.0]),
            throughput=np.asarray([0.0, 1.0, 0.0]),
        )

    state = sed_fit_input_state(
        pd.DataFrame([{"candidate_id": "canonical-key"}]),
        pd.DataFrame([stale_apass]),
        library=object(),
        response_loader=response_loader,
    )
    assert len(state) == 1
    assert calls == [("Generic/Johnson.V", "Vega")]

    response = response_loader("Generic/Johnson.V", "Vega")
    points = _prepare_candidate_points(
        "canonical-key",
        {"candidate_id": "canonical-key"},
        pd.DataFrame([stale_apass]),
        {("Generic/Johnson.V", "Vega"): response},
        {},
    )
    assert points.loc[0, "response_key"] == ("Generic/Johnson.V", "Vega")
    assert points.loc[0, "svo_filter_id"] == "Generic/Johnson.V"
    assert points.loc[0, "mag_system"] == "Vega"
    assert points.loc[0, "prediction_status"] == "pending"


def test_quoted_fnu_failure_reason_precedence_and_strict_reference() -> None:
    from malca.enrichment.sed_model import _prepare_candidate_points
    from malca.enrichment.synthetic_photometry import FilterResponse

    response = FilterResponse(
        filter_id="Spitzer/IRAC.I1",
        mag_system="Jy",
        wavelength_angstrom=np.asarray([30000.0, 35500.0, 41000.0]),
        throughput=np.asarray([0.0, 1.0, 0.0]),
        # This is deliberately populated: it must not be substituted for the
        # catalog mission-reference wavelength below.
        wavelength_ref_angstrom=35500.0,
    )

    def prepare(**overrides: object) -> pd.Series:
        row: dict[str, object] = {
            "candidate_id": "quoted-precedence",
            "source": "External quoted catalog",
            "band": "IRAC1",
            "mag_system": "Jy",
            "observable_kind": "quoted_fnu",
            "flux_nu_jy": 1.0,
            "flux_nu_jy_err": 0.1,
            "lambda_eff_angstrom": 35500.0,
            "svo_filter_id": "Spitzer/IRAC.I1",
            "fit_policy": "diagnostic_only",
        }
        row.update(overrides)
        points = _prepare_candidate_points(
            "quoted-precedence",
            {"candidate_id": "quoted-precedence"},
            pd.DataFrame([row]),
            {("Spitzer/IRAC.I1", "Jy"): response},
            {("Spitzer/IRAC.I1", "Jy"): "not cached"},
        )
        return points.iloc[0]

    missing_filter = prepare(svo_filter_id="", lambda_reference_angstrom=np.nan)
    assert missing_filter["prediction_reason"] == "missing_filter_id"

    missing_mission = prepare(lambda_reference_angstrom=np.nan)
    assert missing_mission["prediction_reason"] == "missing_mission_calibration"
    assert pd.isna(missing_mission["lambda_reference_angstrom"])

    valid_reference_row = {
        "candidate_id": "quoted-precedence",
        "source": "External quoted catalog",
        "band": "IRAC1",
        "mag_system": "Jy",
        "observable_kind": "quoted_fnu",
        "flux_nu_jy": 1.0,
        "flux_nu_jy_err": 0.1,
        "lambda_eff_angstrom": 35500.0,
        "lambda_reference_angstrom": 35500.0,
        "svo_filter_id": "Spitzer/IRAC.I1",
        "fit_policy": "diagnostic_only",
    }
    missing_bandpass = _prepare_candidate_points(
        "quoted-precedence",
        {"candidate_id": "quoted-precedence"},
        pd.DataFrame([valid_reference_row]),
        {},
        {("Spitzer/IRAC.I1", "Jy"): "not cached"},
    ).iloc[0]
    assert missing_bandpass["prediction_reason"] == "missing_bandpass:not cached"
