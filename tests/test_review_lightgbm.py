from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from malca.meta_analysis.ml.review_lightgbm import (
    append_lightcurve_features,
    load_current_schema_training_table,
    load_legacy_march18_training_table,
)


def _create_candidates(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE candidates (
            candidate_id TEXT PRIMARY KEY,
            asas_sn_id TEXT,
            lc_path TEXT,
            dipper_score REAL,
            periodicity_score REAL
        )
        """
    )
    conn.execute(
        "INSERT INTO candidates VALUES (?, ?, ?, ?, ?)",
        ("C1", "C1", "/missing/C1.dat3", 2.5, 0.4),
    )


def test_current_schema_loader_rejects_old_review_only_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        _create_candidates(conn)
        conn.execute(
            """
            CREATE TABLE reviews (
                candidate_id TEXT PRIMARY KEY,
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
        conn.execute(
            "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("C1", 4, "dipper", 2, "", "reviewed", "tester", "2026-03-18T00:00:00+00:00"),
        )

    with pytest.raises(ValueError, match="Current-schema notebook requires"):
        load_current_schema_training_table(db_path, include_lightcurve_features=False)


def test_legacy_loader_derives_taxonomy_from_march18_event_class(tmp_path: Path) -> None:
    db_path = tmp_path / "march18.db"
    with sqlite3.connect(db_path) as conn:
        _create_candidates(conn)
        conn.execute(
            """
            CREATE TABLE reviews (
                candidate_id TEXT PRIMARY KEY,
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
        conn.execute(
            "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("C1", 4, "dipper", 2, "old label", "reviewed", "tester", "2026-03-18T00:00:00+00:00"),
        )

    table = load_legacy_march18_training_table(db_path, include_lightcurve_features=False)

    assert len(table) == 1
    row = table.iloc[0]
    assert row["event_class"] == "dipper"
    assert row["workflow_status"] == "reviewed"
    assert row["morphology_primary"] == "dimming_event"
    assert row["legacy_schema_source"] == "march18"


def test_append_lightcurve_features_reads_flat_export(tmp_path: Path) -> None:
    flat_dir = tmp_path / "bundle_assets" / "lightcurves"
    flat_dir.mkdir(parents=True)
    (flat_dir / "C1.dat3").write_text(
        "\n".join(
            [
                "100.0 12.0 0.02 1 1 0 0 ba/F1",
                "101.0 12.2 0.03 1 2 0 0 bb/F1",
                "102.0 12.4 0.04 1 2 1 0 bb/F1",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    table = pd.DataFrame(
        {
            "candidate_id": ["C1"],
            "asas_sn_id": ["C1"],
            "lc_path": ["/cluster/path/C1.dat3"],
        }
    )

    out = append_lightcurve_features(table, flat_lightcurve_dir=flat_dir)

    assert bool(out.loc[0, "lc_file_exists"]) is True
    assert out.loc[0, "lc_n_points"] == 3
    assert out.loc[0, "lc_n_cameras"] == 2
    assert out.loc[0, "lc_n_bands"] == 2
    assert out.loc[0, "lc_g_n_points"] == 2
    assert out.loc[0, "lc_v_n_points"] == 1
