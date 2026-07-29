from __future__ import annotations

import sqlite3
import types

import numpy as np
import pandas as pd

import malca.enrichment.sed_model as sed_model
from malca.enrichment.sed_fit import (
    _matched_control_candidate_ids,
    _review_event_class_candidate_ids,
    _selected_candidate_ids,
    _stored_photometry,
)
from malca.enrichment.sed_model import (
    SED_MODEL_FIT_VERSION,
    sed_fit_input_state,
    sed_fit_recipe_hash,
    sed_measurement_set_hash,
    sed_model_grid_hash,
)
from malca.enrichment.synthetic_photometry import FilterResponse
from malca.review.sed_storage import store_canonical_sed_rows
from malca.review.store import db_connect


def test_sed_fit_selects_only_missing_and_stale_versions_by_default() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE sed_photometry (candidate_id TEXT)")
    conn.execute("CREATE TABLE sed_model_fits (candidate_id TEXT, fit_version TEXT)")
    conn.executemany("INSERT INTO sed_photometry VALUES (?)", [("current",), ("stale",), ("missing",)])
    conn.executemany(
        "INSERT INTO sed_model_fits VALUES (?, ?)",
        [("current", SED_MODEL_FIT_VERSION), ("stale", "effective-wavelength-v1")],
    )

    selected = _selected_candidate_ids(conn, refit_all=False, candidate_ids=[], limit=None)

    assert selected == ["missing", "stale"]


def test_sed_fit_all_mode_honors_candidate_filter_and_limit() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE sed_photometry (candidate_id TEXT)")
    conn.executemany("INSERT INTO sed_photometry VALUES (?)", [("a",), ("b",), ("c",)])

    selected = _selected_candidate_ids(
        conn,
        refit_all=True,
        candidate_ids=["c", "a"],
        limit=1,
    )

    assert selected == ["a"]


def test_sed_fit_event_class_selector_is_case_insensitive() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE reviews (candidate_id TEXT, event_class TEXT)")
    conn.executemany(
        "INSERT INTO reviews VALUES (?, ?)",
        [("b", "periodic"), ("a", "Dipper"), ("c", " dipper ")],
    )

    selected = _review_event_class_candidate_ids(conn, "dipper")

    assert selected == ["a", "c"]


def test_sed_fit_selects_deterministic_matched_main_sequence_controls() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE candidates (candidate_id TEXT, w1 REAL, gal_b REAL, A_v_3d REAL, yso_class TEXT)"
    )
    conn.execute("CREATE TABLE reviews (candidate_id TEXT, event_class TEXT)")
    conn.execute(
        "CREATE TABLE sed_model_fits (candidate_id TEXT, teff_k REAL, reduced_chi2 REAL, "
        "status TEXT, fit_version TEXT)"
    )
    candidate_rows = [
        ("dip", 9.0, 20.0, 0.2, "Class II"),
        ("near", 9.1, 21.0, 0.25, "Main Sequence"),
        ("far", 13.0, 70.0, 2.0, "Main Sequence"),
        ("reviewed", 9.05, 20.5, 0.2, "Main Sequence"),
        ("yso", 9.0, 20.0, 0.2, "Class II"),
    ]
    conn.executemany("INSERT INTO candidates VALUES (?, ?, ?, ?, ?)", candidate_rows)
    conn.executemany("INSERT INTO reviews VALUES (?, ?)", [("dip", "Dipper"), ("reviewed", "periodic")])
    conn.executemany(
        "INSERT INTO sed_model_fits VALUES (?, ?, ?, ?, ?)",
        [
            ("dip", 6000.0, 1.0, "ok", SED_MODEL_FIT_VERSION),
            ("near", 6100.0, 1.1, "ok", "legacy"),
            ("far", 9000.0, 20.0, "ok", "legacy"),
            ("reviewed", 6050.0, 1.0, "ok", "legacy"),
            ("yso", 6000.0, 1.0, "ok", "legacy"),
        ],
    )

    selected = _matched_control_candidate_ids(conn, "dipper", 2)

    assert selected == ["near", "far"]


def test_sed_fit_refits_current_version_when_photometry_or_recipe_hash_is_stale() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE sed_photometry (candidate_id TEXT, source TEXT, band TEXT, mag REAL)")
    conn.execute(
        "CREATE TABLE sed_model_fits ("
        "candidate_id TEXT, fit_version TEXT, measurement_set_hash TEXT, fit_recipe_hash TEXT)"
    )
    rows = [
        ("current", "APASS", "B", 12.0),
        ("changed", "APASS", "B", 13.0),
        ("recipe", "APASS", "B", 14.0),
    ]
    conn.executemany("INSERT INTO sed_photometry VALUES (?, ?, ?, ?)", rows)
    for candidate_id, source, band, mag in rows:
        measurement_hash = sed_measurement_set_hash(
            pd.DataFrame([{"candidate_id": candidate_id, "source": source, "band": band, "mag": mag}])
        )
        if candidate_id == "changed":
            measurement_hash = "old-measurement"
        recipe_hash = sed_fit_recipe_hash() if candidate_id != "recipe" else "old-recipe"
        conn.execute(
            "INSERT INTO sed_model_fits VALUES (?, ?, ?, ?)",
            (candidate_id, SED_MODEL_FIT_VERSION, measurement_hash, recipe_hash),
        )

    selected = _selected_candidate_ids(conn, refit_all=False, candidate_ids=[], limit=None)

    assert selected == ["changed", "recipe"]


def test_sed_fit_refits_when_candidate_or_cached_manifest_changes() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE sed_photometry (candidate_id TEXT, source TEXT, band TEXT, mag REAL)")
    conn.execute(
        "CREATE TABLE sed_model_fits ("
        "candidate_id TEXT, fit_version TEXT, measurement_set_hash TEXT, fit_recipe_hash TEXT, "
        "candidate_context_hash TEXT, response_manifest_hash TEXT, "
        "calibration_manifest_hash TEXT, model_grid_hash TEXT)"
    )
    conn.execute("INSERT INTO sed_photometry VALUES ('a', 'APASS', 'B', 12.0)")
    conn.execute(
        "INSERT INTO sed_model_fits VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "a",
            SED_MODEL_FIT_VERSION,
            "measurement",
            "recipe",
            "candidate-old",
            "response",
            "calibration",
            "grid",
        ),
    )
    current_state = pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "measurement_set_hash": "measurement",
                "fit_recipe_hash": "recipe",
                "candidate_context_hash": "candidate-new",
                "response_manifest_hash": "response",
                "calibration_manifest_hash": "calibration",
                "model_grid_hash": "grid",
            }
        ]
    )

    selected = _selected_candidate_ids(
        conn,
        refit_all=False,
        candidate_ids=[],
        limit=None,
        current_state=current_state,
    )

    assert selected == ["a"]


def test_completed_fit_replay_uses_v3_epochs_and_only_refits_changed_grid(
    tmp_path,
    monkeypatch,
) -> None:
    package_root = tmp_path / "pystellibs"
    grid_path = package_root / "libs" / "kurucz2004.grid.fits"
    grid_path.parent.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    grid_path.write_bytes(b"stable-test-grid-v1")
    fake_spec = types.SimpleNamespace(origin=str(package_root / "__init__.py"))
    monkeypatch.setattr(
        sed_model.importlib.util,
        "find_spec",
        lambda name: fake_spec if name == "pystellibs" else None,
    )
    monkeypatch.setattr(sed_model.importlib.metadata, "version", lambda name: "test-version")

    actual_kurucz_class = type("Kurucz", (), {"__module__": "pystellibs.kurucz"})
    actual_library = actual_kurucz_class()
    actual_library.source = str(grid_path)

    # Selection-time provenance (no instantiated object) and fit-time
    # provenance (the implementation class) must describe the same grid.
    assert sed_model_grid_hash() == sed_model_grid_hash(actual_library)
    assert sed_fit_recipe_hash() == sed_fit_recipe_hash(actual_library)

    db_path = tmp_path / "review.db"
    conn = db_connect(db_path)
    try:
        conn.execute(
            "INSERT INTO candidates (candidate_id, payload_json, imported_at) VALUES (?, ?, ?)",
            ("nsc-epochs", "{}", "2026-07-18T00:00:00+00:00"),
        )
        canonical_rows = pd.DataFrame(
            [
                {
                    "candidate_id": "nsc-epochs",
                    "source": "NOIRLab NSC DR2",
                    "catalog": "NOIRLab NSC DR2",
                    "catalog_release": "DR2",
                    "catalog_object_id": "nsc-object",
                    "catalog_measurement_id": f"nsc-measurement-{index}",
                    "exposure_id": f"c4d-exposure-{index}",
                    "instrument": "c4d",
                    "band": "g",
                    "epoch_mjd": epoch,
                    "mag": 18.0 + 0.1 * index,
                    "mag_err": 0.02,
                    "mag_system": "AB",
                    "flux_nu_jy": flux,
                    "flux_nu_jy_err": flux * 0.02,
                    "lambda_eff_angstrom": 4770.0,
                    "lambda_pivot_angstrom": 4785.0,
                    "lambda_reference_angstrom": 4801.0,
                    "plot_lambda_angstrom": 4785.0,
                    "plot_lambda_kind": "response_pivot",
                    "svo_filter_id": "CTIO/DECam.g",
                    "passband_fidelity": "exact",
                    "fit_policy": "photosphere",
                    "quality_flags": "",
                    "normalization_version": "sed-measurement-v3",
                }
                for index, (epoch, flux) in enumerate(
                    ((59000.0, 2.1e-4), (59100.0, 1.9e-4)),
                    start=1,
                )
            ]
        )
        store_canonical_sed_rows(conn, canonical_rows)
        # Keep one legacy object-mean row solely as the candidate-discovery
        # bridge.  The fit loader must prefer the two canonical epoch rows.
        conn.execute(
            "INSERT INTO sed_photometry (candidate_id, source, band, mag, mag_system) "
            "VALUES (?, ?, ?, ?, ?)",
            ("nsc-epochs", "NOIRLab NSC DR2", "g", 18.0, "AB"),
        )
        conn.commit()

        photometry = _stored_photometry(conn, ["nsc-epochs"])
        assert len(photometry) == 2
        assert photometry["measurement_id"].nunique() == 2
        assert photometry["epoch_mjd"].tolist() == [59000.0, 59100.0]
        assert set(photometry["instrument"]) == {"c4d"}
        assert set(photometry["fit_policy"]) == {"photosphere"}
        assert set(photometry["lambda_reference_angstrom"]) == {4801.0}
        assert set(photometry["svo_filter_id"]) == {"CTIO/DECam.g"}
        # Re-presenting a canonical load to storage is an exact no-op, proving
        # the native/provenance fields were not lost in the batch mapping.
        assert store_canonical_sed_rows(conn, photometry) == (0, 0)

        def response_loader(filter_id: str, mag_system: str) -> FilterResponse:
            assert (filter_id, mag_system) == ("CTIO/DECam.g", "AB")
            return FilterResponse(
                filter_id=filter_id,
                mag_system=mag_system,
                wavelength_angstrom=np.asarray([4300.0, 4785.0, 5300.0]),
                throughput=np.asarray([0.0, 1.0, 0.0]),
            )

        candidates = pd.DataFrame([{"candidate_id": "nsc-epochs"}])
        current_state = sed_fit_input_state(
            candidates,
            photometry,
            response_loader=response_loader,
        )
        state = current_state.iloc[0]
        repeated_state = sed_fit_input_state(
            candidates,
            photometry.iloc[::-1].reset_index(drop=True),
            response_loader=response_loader,
        ).iloc[0]
        assert state["measurement_set_hash"] == repeated_state["measurement_set_hash"]
        assert state["measurement_set_hash"] != sed_measurement_set_hash(photometry, "nsc-epochs")
        assert state["model_grid_hash"] == sed_model_grid_hash(actual_library)
        assert state["fit_recipe_hash"] == sed_fit_recipe_hash(actual_library)

        conn.execute(
            "INSERT OR REPLACE INTO sed_model_fits ("
            "candidate_id, fit_version, measurement_set_hash, fit_recipe_hash, "
            "candidate_context_hash, response_manifest_hash, calibration_manifest_hash, "
            "input_policy_manifest_hash, model_grid_hash, status"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "nsc-epochs",
                SED_MODEL_FIT_VERSION,
                state["measurement_set_hash"],
                sed_fit_recipe_hash(actual_library),
                state["candidate_context_hash"],
                state["response_manifest_hash"],
                state["calibration_manifest_hash"],
                state["input_policy_manifest_hash"],
                sed_model_grid_hash(actual_library),
                "ok",
            ),
        )
        conn.commit()
        assert _selected_candidate_ids(
            conn,
            refit_all=False,
            candidate_ids=[],
            limit=None,
            current_state=current_state,
        ) == []

        grid_path.write_bytes(b"changed-test-grid-v2-with-different-content")
        changed_state = sed_fit_input_state(
            candidates,
            photometry,
            response_loader=response_loader,
        )
        assert changed_state.loc[0, "model_grid_hash"] != state["model_grid_hash"]
        assert _selected_candidate_ids(
            conn,
            refit_all=False,
            candidate_ids=[],
            limit=None,
            current_state=changed_state,
        ) == ["nsc-epochs"]
    finally:
        conn.close()
