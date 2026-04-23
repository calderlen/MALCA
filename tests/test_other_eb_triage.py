from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from malca.review.other_eb_triage import (
    compute_eb_triage,
    export_eb_triage_products,
    inspect_candidate,
    load_reviewed_other_subset,
    resolve_local_paths,
)


def _write_review_db(
    path: Path,
    *,
    candidate_rows: list[dict[str, object]],
    review_rows: list[dict[str, object]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE candidates (
                candidate_id TEXT,
                asas_sn_id TEXT,
                path TEXT,
                dip_run_count REAL,
                dip_inter_event_spacing_median REAL,
                dip_inter_event_spacing_std REAL,
                dip_amplitude_consistency REAL,
                dip_duration_consistency REAL,
                dip_symmetry_score REAL,
                stats_variability_lomb_scargle_best_period_days REAL,
                stats_variability_lomb_scargle_peak_power REAL,
                stats_variability_lomb_scargle_fap REAL,
                stats_variability_von_neumann_ratio REAL,
                stats_variability_stetson_J REAL,
                dipper_score REAL,
                gaia_var_class TEXT,
                gaia_eb_period REAL,
                gaia_eb_morph TEXT,
                vsx_class TEXT,
                catalog_match INTEGER,
                periodic_flag INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE reviews (
                candidate_id TEXT,
                interest_score INTEGER,
                event_class TEXT,
                review_pass INTEGER,
                notes TEXT,
                status TEXT,
                reviewer TEXT,
                updated_at TEXT
            )
            """
        )
        for row in candidate_rows:
            conn.execute(
                """
                INSERT INTO candidates VALUES (
                    :candidate_id, :asas_sn_id, :path, :dip_run_count,
                    :dip_inter_event_spacing_median, :dip_inter_event_spacing_std,
                    :dip_amplitude_consistency, :dip_duration_consistency, :dip_symmetry_score,
                    :stats_variability_lomb_scargle_best_period_days,
                    :stats_variability_lomb_scargle_peak_power,
                    :stats_variability_lomb_scargle_fap,
                    :stats_variability_von_neumann_ratio,
                    :stats_variability_stetson_J,
                    :dipper_score,
                    :gaia_var_class,
                    :gaia_eb_period,
                    :gaia_eb_morph,
                    :vsx_class,
                    :catalog_match,
                    :periodic_flag
                )
                """,
                row,
            )
        for row in review_rows or []:
            conn.execute(
                """
                INSERT INTO reviews VALUES (
                    :candidate_id, :interest_score, :event_class, :review_pass,
                    :notes, :status, :reviewer, :updated_at
                )
                """,
                row,
            )
        conn.commit()


def test_load_reviewed_other_subset_filters_reviewed_other_from_db(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    _write_review_db(
        db_path,
        candidate_rows=[
            {
                "candidate_id": "C1",
                "asas_sn_id": "1001",
                "path": "/missing/C1.dat2",
                "dip_run_count": 2,
                "dip_inter_event_spacing_median": 10.0,
                "dip_inter_event_spacing_std": 1.0,
                "dip_amplitude_consistency": 0.2,
                "dip_duration_consistency": 0.2,
                "dip_symmetry_score": 0.1,
                "stats_variability_lomb_scargle_best_period_days": 2.5,
                "stats_variability_lomb_scargle_peak_power": 0.3,
                "stats_variability_lomb_scargle_fap": 1e-7,
                "stats_variability_von_neumann_ratio": 1.8,
                "stats_variability_stetson_J": 0.7,
                "dipper_score": 12.0,
                "gaia_var_class": "ECL",
                "gaia_eb_period": 2.5,
                "gaia_eb_morph": "TWOGAUSSIANS",
                "vsx_class": "",
                "catalog_match": 1,
                "periodic_flag": 0,
            },
            {
                "candidate_id": "C2",
                "asas_sn_id": "1002",
                "path": "/missing/C2.dat2",
                "dip_run_count": 1,
                "dip_inter_event_spacing_median": None,
                "dip_inter_event_spacing_std": None,
                "dip_amplitude_consistency": None,
                "dip_duration_consistency": None,
                "dip_symmetry_score": None,
                "stats_variability_lomb_scargle_best_period_days": None,
                "stats_variability_lomb_scargle_peak_power": None,
                "stats_variability_lomb_scargle_fap": None,
                "stats_variability_von_neumann_ratio": None,
                "stats_variability_stetson_J": None,
                "dipper_score": 4.0,
                "gaia_var_class": "",
                "gaia_eb_period": None,
                "gaia_eb_morph": "",
                "vsx_class": "",
                "catalog_match": 0,
                "periodic_flag": 0,
            },
            {
                "candidate_id": "C3",
                "asas_sn_id": "1003",
                "path": "/missing/C3.dat2",
                "dip_run_count": 2,
                "dip_inter_event_spacing_median": 9.0,
                "dip_inter_event_spacing_std": 1.0,
                "dip_amplitude_consistency": 0.3,
                "dip_duration_consistency": 0.3,
                "dip_symmetry_score": 0.2,
                "stats_variability_lomb_scargle_best_period_days": 4.0,
                "stats_variability_lomb_scargle_peak_power": 0.2,
                "stats_variability_lomb_scargle_fap": 1e-4,
                "stats_variability_von_neumann_ratio": 1.4,
                "stats_variability_stetson_J": 0.4,
                "dipper_score": 9.0,
                "gaia_var_class": "",
                "gaia_eb_period": None,
                "gaia_eb_morph": "",
                "vsx_class": "",
                "catalog_match": 0,
                "periodic_flag": 0,
            },
        ],
        review_rows=[
            {
                "candidate_id": "C1",
                "interest_score": 4,
                "event_class": "other",
                "review_pass": 2,
                "notes": "keep",
                "status": "reviewed",
                "reviewer": "tester",
                "updated_at": "2026-04-23T00:00:00Z",
            },
            {
                "candidate_id": "C2",
                "interest_score": 2,
                "event_class": "other",
                "review_pass": 1,
                "notes": "unreviewed",
                "status": "unreviewed",
                "reviewer": "tester",
                "updated_at": "2026-04-23T00:00:00Z",
            },
            {
                "candidate_id": "C3",
                "interest_score": 3,
                "event_class": "dipper",
                "review_pass": 1,
                "notes": "other class",
                "status": "reviewed",
                "reviewer": "tester",
                "updated_at": "2026-04-23T00:00:00Z",
            },
        ],
    )

    subset = load_reviewed_other_subset(db_path)

    assert subset["candidate_id"].tolist() == ["C1"]
    assert subset.loc[0, "event_class"] == "other"
    assert subset.loc[0, "status"] == "reviewed"


def test_load_reviewed_other_subset_joins_exported_labels_by_candidate_id(tmp_path: Path) -> None:
    candidates_path = tmp_path / "candidates.parquet"
    labels_path = tmp_path / "labels.parquet"

    pd.DataFrame(
        [
            {"candidate_id": "A1", "dip_run_count": 2, "dipper_score": 8.0},
            {"candidate_id": "A2", "dip_run_count": 1, "dipper_score": 3.0},
        ]
    ).to_parquet(candidates_path, index=False)
    pd.DataFrame(
        [
            {"candidate_id": "A1", "event_class": "other", "status": "reviewed", "interest_score": 4},
            {"candidate_id": "A2", "event_class": "dipper", "status": "reviewed", "interest_score": 2},
        ]
    ).to_parquet(labels_path, index=False)

    subset = load_reviewed_other_subset(labels_path, candidates_path)

    assert subset["candidate_id"].tolist() == ["A1"]
    assert subset.loc[0, "interest_score"] == 4
    assert subset.loc[0, "dipper_score"] == 8.0


def test_compute_eb_triage_assigns_expected_bins() -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_id": "strong",
                "stats_variability_lomb_scargle_best_period_days": 2.1,
                "stats_variability_lomb_scargle_peak_power": 0.35,
                "stats_variability_lomb_scargle_fap": 1e-8,
                "dip_run_count": 3,
                "dip_inter_event_spacing_median": 10.0,
                "dip_inter_event_spacing_std": 1.0,
                "dip_amplitude_consistency": 0.20,
                "dip_duration_consistency": 0.25,
                "dip_symmetry_score": 0.10,
                "stats_variability_von_neumann_ratio": 1.8,
                "stats_variability_stetson_J": 0.6,
                "dipper_score": 14.0,
            },
            {
                "candidate_id": "possible",
                "stats_variability_lomb_scargle_best_period_days": 5.0,
                "stats_variability_lomb_scargle_peak_power": 0.18,
                "stats_variability_lomb_scargle_fap": 1e-4,
                "dip_run_count": 2,
                "dip_inter_event_spacing_median": 15.0,
                "dip_inter_event_spacing_std": 6.0,
                "dip_amplitude_consistency": 0.40,
                "dip_duration_consistency": 0.45,
                "dip_symmetry_score": 0.20,
                "stats_variability_von_neumann_ratio": 1.2,
                "stats_variability_stetson_J": 0.2,
                "dipper_score": 9.0,
            },
            {
                "candidate_id": "unlikely",
                "stats_variability_lomb_scargle_best_period_days": np.nan,
                "stats_variability_lomb_scargle_peak_power": np.nan,
                "stats_variability_lomb_scargle_fap": np.nan,
                "dip_run_count": 1,
                "dip_inter_event_spacing_median": np.nan,
                "dip_inter_event_spacing_std": np.nan,
                "dip_amplitude_consistency": np.nan,
                "dip_duration_consistency": np.nan,
                "dip_symmetry_score": np.nan,
                "stats_variability_von_neumann_ratio": 1.0,
                "stats_variability_stetson_J": 0.1,
                "dipper_score": 2.0,
            },
        ]
    )

    out = compute_eb_triage(df)
    by_id = out.set_index("candidate_id")

    assert by_id.loc["strong", "eb_bin"] == "strong_eb_candidate"
    assert int(by_id.loc["strong", "eb_score"]) >= 7
    assert bool(by_id.loc["strong", "eb_likely_flag"]) is True
    assert by_id.loc["strong", "eb_likely_label"] == "likely_eb"

    assert by_id.loc["possible", "eb_bin"] == "possible_eb"
    assert int(by_id.loc["possible", "eb_score"]) == 5
    assert bool(by_id.loc["possible", "eb_likely_flag"]) is True
    assert by_id.loc["possible", "eb_likely_label"] == "likely_eb"

    assert by_id.loc["unlikely", "eb_bin"] == "unlikely_eb"
    assert int(by_id.loc["unlikely", "eb_score"]) == 0
    assert bool(by_id.loc["unlikely", "eb_likely_flag"]) is False
    assert by_id.loc["unlikely", "eb_likely_label"] == "not_likely_eb"


def test_compute_eb_triage_handles_missing_metrics() -> None:
    out = compute_eb_triage(pd.DataFrame([{"candidate_id": "missing"}]))

    assert out.loc[0, "candidate_id"] == "missing"
    assert int(out.loc[0, "eb_score"]) == 0
    assert out.loc[0, "eb_bin"] == "unlikely_eb"
    assert bool(out.loc[0, "eb_likely_flag"]) is False
    assert out.loc[0, "eb_likely_label"] == "not_likely_eb"
    assert pd.isna(out.loc[0, "ls_fap_score"])
    assert pd.isna(out.loc[0, "spacing_cv"])


def test_export_eb_triage_products_handles_categorical_bins(tmp_path: Path) -> None:
    subset_df = pd.DataFrame([{"candidate_id": "C1"}])
    triage_df = compute_eb_triage(
        pd.DataFrame(
            [
                {
                    "candidate_id": "C1",
                    "stats_variability_lomb_scargle_best_period_days": 2.0,
                    "stats_variability_lomb_scargle_peak_power": 0.4,
                    "stats_variability_lomb_scargle_fap": 1e-8,
                    "dip_run_count": 3,
                    "dip_inter_event_spacing_median": 10.0,
                    "dip_inter_event_spacing_std": 1.0,
                    "dip_amplitude_consistency": 0.2,
                    "dip_duration_consistency": 0.2,
                    "dip_symmetry_score": 0.1,
                    "stats_variability_von_neumann_ratio": 1.8,
                    "stats_variability_stetson_J": 0.7,
                    "dipper_score": 10.0,
                }
            ]
        )
    )

    paths = export_eb_triage_products(subset_df, triage_df, export_dir=tmp_path / "export")

    assert paths["subset_path"].exists()
    assert paths["triage_path"].exists()
    assert paths["top_candidates_path"].exists()


def test_resolve_local_paths_and_inspect_candidate_metadata_fallback(tmp_path: Path) -> None:
    run_dir = tmp_path / "demo_run"
    lightcurve_dir = run_dir / "bundle_assets" / "lightcurves"
    lightcurve_dir.mkdir(parents=True)
    lightcurve_path = lightcurve_dir / "123.dat2"
    lightcurve_path.write_text("", encoding="ascii")

    df = pd.DataFrame(
        [
            {
                "candidate_id": "123",
                "path": "/cluster/path/123.dat2",
                "dipper_score": 7.0,
            },
            {
                "candidate_id": "no_lc",
                "path": "/cluster/path/no_lc.dat2",
                "dipper_score": 2.0,
            },
        ]
    )

    resolved = resolve_local_paths(df, run_dir=run_dir)
    by_id = resolved.set_index("candidate_id")

    assert bool(by_id.loc["123", "local_lightcurve_exists"]) is True
    assert Path(by_id.loc["123", "local_lightcurve_path"]) == lightcurve_path
    assert bool(by_id.loc["no_lc", "local_lightcurve_exists"]) is False

    inspected = inspect_candidate(resolved, "no_lc", run_dir=run_dir)

    assert inspected["status"] == "metadata_only"
    assert inspected["figure"] is None
    assert inspected["lightcurve_df"] is None
