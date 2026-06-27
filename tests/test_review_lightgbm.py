from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from malca.meta_analysis.ml.review_lightgbm import (
    CURRENT_SCHEMA_REQUIRED_REVIEW_COLUMNS,
    DEFAULT_DROP_COLUMNS,
    TrainingConfig,
    append_lightcurve_features,
    load_target_model,
    load_current_schema_training_table,
    load_legacy_march18_training_table,
    load_recompute_survival_training_table,
    score_target_model,
    train_review_models,
    train_target_model,
)
from malca.review.taxonomy import REVIEW_TAXONOMY_FIELDS
from malca.io.table_io import write_feature_table


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


def _tiny_training_config() -> TrainingConfig:
    return TrainingConfig(
        random_state=11,
        cv_folds=3,
        n_estimators=12,
        num_leaves=7,
        min_child_samples=1,
        min_class_count=3,
        n_jobs=1,
    )


def _review_label_frame(labels: list[str], n_per_class: int = 6) -> pd.DataFrame:
    rows = []
    for class_idx, label in enumerate(labels):
        for idx in range(n_per_class):
            rows.append(
                {
                    "candidate_id": f"{label}-{idx}",
                    "review_label": label,
                    "feature_signal": float(class_idx * 10 + idx),
                    "feature_mod": float(idx % 3),
                    "feature_bucket": f"bucket_{class_idx % 2}",
                }
            )
    return pd.DataFrame(rows)


def test_current_schema_columns_follow_review_taxonomy() -> None:
    assert CURRENT_SCHEMA_REQUIRED_REVIEW_COLUMNS == ("candidate_id", *REVIEW_TAXONOMY_FIELDS)
    assert set(REVIEW_TAXONOMY_FIELDS).issubset(DEFAULT_DROP_COLUMNS)


def test_review_lightgbm_writes_reusable_binary_artifacts(tmp_path: Path) -> None:
    df = _review_label_frame(["artifact", "dipper"], n_per_class=8)
    output_dir = tmp_path / "models"

    results = train_review_models(
        df,
        ["review_label"],
        output_dir=output_dir,
        config=_tiny_training_config(),
    )

    assert "review_label" in results
    target_dir = output_dir / "review_label"
    expected = {
        "model.joblib",
        "lightgbm_model.txt",
        "metadata.json",
        "split_assignments.parquet",
        "test_predictions.parquet",
        "feature_importance.csv",
        "confusion_matrix.csv",
        "calibration_by_bin.csv",
        "cv_metrics.csv",
    }
    assert expected.issubset({path.name for path in target_dir.iterdir()})

    split = pd.read_parquet(target_dir / "split_assignments.parquet")
    assert set(split["split"]) == {"train", "val", "test"}
    preds = pd.read_parquet(target_dir / "test_predictions.parquet")
    assert {"candidate_id", "y_true", "y_pred", "correct", "prob_artifact", "prob_dipper"}.issubset(preds.columns)

    metadata = json.loads((target_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["target_column"] == "review_label"
    assert metadata["config"]["calibration_method"] == "none"
    assert "roc_auc" in metadata["test_metrics"]
    assert "pr_auc" in metadata["test_metrics"]


def test_review_lightgbm_multiclass_metrics_and_artifacts() -> None:
    df = _review_label_frame(["artifact", "dipper", "microlensing"], n_per_class=6)

    result = train_target_model(
        df,
        "review_label",
        config=_tiny_training_config(),
    )

    prob_cols = [col for col in result.test_predictions.columns if col.startswith("prob_")]
    assert prob_cols == ["prob_artifact", "prob_dipper", "prob_microlensing"]
    assert {"roc_auc_ovr_macro", "roc_auc_ovr_weighted", "pr_auc_macro", "pr_auc_weighted"}.issubset(
        result.holdout_metrics
    )
    assert result.confusion_matrix.columns.tolist() == ["y_true", "artifact", "dipper", "microlensing"]
    assert not result.feature_importance.empty
    assert {"class_label", "probability_column", "probability_bin", "observed_rate"}.issubset(
        result.calibration_diagnostics.columns
    )


def test_review_lightgbm_load_and_score_roundtrip(tmp_path: Path) -> None:
    df = _review_label_frame(["artifact", "dipper"], n_per_class=8)
    target_dir = tmp_path / "review_label"
    result = train_target_model(df, "review_label", config=_tiny_training_config())
    from malca.meta_analysis.ml.review_lightgbm import save_target_model

    save_target_model(result, target_dir)

    bundle = load_target_model(target_dir)
    scored_from_bundle = score_target_model(bundle, df.drop(columns=["review_label"]).head(4))
    scored_from_path = score_target_model(target_dir, df.drop(columns=["review_label"]).head(4))

    assert set(scored_from_bundle["y_pred"]).issubset(set(bundle["label_classes"]))
    assert scored_from_bundle[["prob_artifact", "prob_dipper"]].shape == (4, 2)
    pd.testing.assert_series_equal(scored_from_bundle["y_pred"], scored_from_path["y_pred"])


def test_review_lightgbm_raises_clearly_for_too_rare_split_class() -> None:
    df = pd.DataFrame(
        {
            "candidate_id": ["a1", "a2", "b1", "b2", "b3"],
            "review_label": ["rare", "rare", "common", "common", "common"],
            "feature_signal": [1.0, 1.2, 3.0, 3.2, 3.4],
        }
    )
    config = TrainingConfig(
        n_estimators=5,
        min_child_samples=1,
        min_class_count=2,
        n_jobs=1,
    )

    with pytest.raises(ValueError, match="at least 3"):
        train_target_model(df, "review_label", config=config)


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


def test_recompute_survival_loader_labels_old_vetted_candidates(tmp_path: Path) -> None:
    old_a = tmp_path / "old_a.parquet"
    old_b = tmp_path / "old_b.parquet"
    recomputed = tmp_path / "recomputed.parquet"
    write_feature_table(
        pd.DataFrame(
            [
                {
                    "candidate_id": "",
                    "timescale": "stv",
                    "asas_sn_id": "123.0",
                    "lc_path": "/data/lcsv2/12_12.5/lc1_cal/123.dat2",
                    "old_only_feature": 1.5,
                },
                {
                    "candidate_id": "456",
                    "timescale": "stv",
                    "asas_sn_id": "456",
                    "lc_path": "/data/lcsv2/12.5_13/lc1_cal/456.dat2",
                    "old_only_feature": 2.5,
                },
            ]
        ),
        old_a,
    )
    write_feature_table(
        pd.DataFrame(
            [
                {
                    "candidate_id": "789",
                    "timescale": "stv",
                    "lc_path": "/data/lcsv2/13_13.5/lc1_cal/789.dat2",
                    "old_only_feature": 3.5,
                },
                {
                    "candidate_id": "999",
                    "timescale": "stv",
                    "asas_sn_id": "999",
                    "lc_path": "/data/lcsv2/14.5_15/lc1_cal/999.dat2",
                    "old_only_feature": 4.5,
                },
            ]
        ),
        old_b,
    )
    write_feature_table(
        pd.DataFrame(
            [
                {
                    "candidate_id": "123",
                    "timescale": "stv",
                    "asas_sn_id": "123",
                    "lc_path": "/data/lcsv2/12_12.5/lc1_cal/123.dat3",
                    "recomputed_only_feature": 10,
                },
                {
                    "candidate_id": "789",
                    "timescale": "stv",
                    "lc_path": "/data/lcsv2/13_13.5/lc1_cal/789.dat3",
                    "recomputed_only_feature": 11,
                },
            ]
        ),
        recomputed,
    )

    table = load_recompute_survival_training_table(
        [old_a, old_b],
        recomputed,
        mag_bins=("12_12.5", "12.5_13", "13_13.5"),
    )

    assert len(table) == 3
    assert "old_only_feature" in table.columns
    assert "recomputed_only_feature" not in table.columns
    assert table["mag_bin"].tolist() == ["12_12.5", "12.5_13", "13_13.5"]
    assert table["timescale"].unique().tolist() == ["stv"]
    assert table["schema_mode"].unique().tolist() == ["recompute_survival"]

    labels = dict(zip(table["old_only_feature"], table["recompute_survived"]))
    statuses = dict(zip(table["old_only_feature"], table["recompute_status"]))
    assert labels == {1.5: 1, 2.5: 0, 3.5: 1}
    assert statuses[1.5] == "survived_recompute"
    assert statuses[2.5] == "fell_away"
    assert table.loc[table["old_only_feature"] == 3.5, "candidate_id"].iloc[0] == "789"
    assert table.loc[table["old_only_feature"] == 3.5, "asas_sn_id"].iloc[0] == "789"


def test_recompute_survival_loader_rejects_duplicate_old_ids(tmp_path: Path) -> None:
    old = tmp_path / "old.parquet"
    recomputed = tmp_path / "recomputed.parquet"
    write_feature_table(
        pd.DataFrame(
            [
                {"candidate_id": "C1_a", "timescale": "stv", "asas_sn_id": "C1", "lc_path": "/data/lcsv2/12_12.5/a/C1.dat2"},
                {"candidate_id": "C1_b", "timescale": "stv", "asas_sn_id": "C1", "lc_path": "/data/lcsv2/12_12.5/b/C1.dat2"},
            ]
        ),
        old,
    )
    write_feature_table(
        pd.DataFrame(
            [{"candidate_id": "C1", "timescale": "stv", "asas_sn_id": "C1", "lc_path": "/data/lcsv2/12_12.5/a/C1.dat3"}]
        ),
        recomputed,
    )

    with pytest.raises(ValueError, match="not unique"):
        load_recompute_survival_training_table([old], recomputed)
