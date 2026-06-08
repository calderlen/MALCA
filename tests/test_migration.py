from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

from malca.feature_layers import FEATURE_LAYER_COLUMNS
from malca.migration import discover_artifacts, migrate_tree
from malca.review import sync
from malca.review.store import count_queue, db_connect, save_app_state, upsert_candidates_frame


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "assets": []}, sort_keys=True) + "\n", encoding="utf-8")


def _layer(row: pd.Series, name: str) -> dict[str, object]:
    value = row[name]
    assert isinstance(value, str)
    parsed = json.loads(value)
    assert isinstance(parsed, dict)
    return parsed


def test_discover_artifacts_classifies_products_and_copied_assets(tmp_path: Path) -> None:
    root = tmp_path / "output"
    product = root / "runs" / "stv_run" / "results" / "lc_events_filtered.parquet"
    product.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "path": ["C1.dat2"],
            "dip_significant": [True],
            "dipper_score": [7.0],
        }
    ).to_parquet(product, index=False)

    cache = root / "cache" / "catalogs" / "sed" / "catalog.parquet"
    cache.parent.mkdir(parents=True)
    pd.DataFrame({"candidate_id": ["C1"], "dipper_score": [1.0]}).to_parquet(cache, index=False)

    side_table = root / "runs" / "reviewed" / "review" / "review_sed_photometry.parquet"
    side_table.parent.mkdir(parents=True)
    pd.DataFrame({"candidate_id": ["C1"], "source": ["2MASS"], "band": ["J"], "mag": [12.0]}).to_parquet(
        side_table,
        index=False,
    )

    index_table = root / "runs" / "agn" / "milliquas_index_chunks" / "milliquas_index_000000.parquet"
    index_table.parent.mkdir(parents=True)
    pd.DataFrame({"candidate_id": ["C1"], "ra": [1.0], "dec": [2.0]}).to_parquet(index_table, index=False)

    asset = root / "runs" / "stv_run" / "bundle_assets" / "lightcurves" / "C1.dat2"
    asset.parent.mkdir(parents=True)
    asset.write_text("lc\n", encoding="utf-8")

    artifacts = discover_artifacts(root)
    by_rel = {item.relative_path: item for item in artifacts}

    assert by_rel["runs/stv_run/results/lc_events_filtered.parquet"].artifact_type == "product_parquet"
    assert by_rel["runs/stv_run/results/lc_events_filtered.parquet"].action == "migrate"
    assert by_rel["cache/catalogs/sed/catalog.parquet"].action == "copy"
    assert by_rel["runs/reviewed/review/review_sed_photometry.parquet"].action == "copy"
    assert by_rel["runs/agn/milliquas_index_chunks/milliquas_index_000000.parquet"].action == "copy"
    assert by_rel["runs/stv_run/bundle_assets/lightcurves/C1.dat2"].action == "copy"


def test_migrate_tree_rewrites_product_parquet_layer_first_and_preserves_original(tmp_path: Path) -> None:
    root = tmp_path / "output"
    product = root / "runs" / "stv_run" / "results" / "lc_events_filtered.parquet"
    product.parent.mkdir(parents=True)
    original = pd.DataFrame(
        {
            "path": ["C1.dat2"],
            "dip_significant": [True],
            "dipper_score": [7.0],
            "camera_field_key": ["ba/F1"],
            "camera_fields": ["ba/F1,ba/F2,bb/F1"],
            "camera_field_count": [3],
            "camera_field_key_fraction": [0.5],
            "stats_camera_field_key": ["ba/F1"],
            "stats_camera_fields": ["ba/F1,ba/F2"],
            "stats_camera_field_count": [2],
            "stats_camera_field_key_fraction": [0.75],
            "phot_g_mean_mag": [14.2],
            "stats_harmonics_a2": [2.0],
            "stats_harmonics_a4": [6.0],
        }
    )
    original.to_parquet(product, index=False)
    asset = root / "runs" / "stv_run" / "plots" / "C1.png"
    asset.parent.mkdir(parents=True)
    asset.write_text("plot\n", encoding="utf-8")

    out_root = tmp_path / "migrated"
    summary = migrate_tree(root, out_root)

    assert summary.ok is True
    assert product.exists()
    assert "dipper_score" in pd.read_parquet(product).columns
    migrated = pd.read_parquet(out_root / "runs" / "stv_run" / "results" / "lc_events_filtered.parquet")
    assert migrated.loc[0, "candidate_id"] == "stv_C1"
    assert migrated.loc[0, "lc_path"] == "C1.dat2"
    assert migrated.loc[0, "timescale"] == "stv"
    assert "dipper_score" not in migrated.columns
    assert "phot_g_mean_mag" not in migrated.columns
    assert set(FEATURE_LAYER_COLUMNS).issubset(migrated.columns)
    assert _layer(migrated.iloc[0], "lc_stats")["dipper_score"] == 7.0
    assert _layer(migrated.iloc[0], "lc_stats")["stats_camera_name_key"] == "ba"
    assert _layer(migrated.iloc[0], "lc_stats")["stats_camera_names"] == "ba"
    assert "stats_camera_field_key" not in _layer(migrated.iloc[0], "lc_stats")
    assert _layer(migrated.iloc[0], "external_stats")["phot_g_mean_mag"] == 14.2
    assert _layer(migrated.iloc[0], "external_stats")["camera_name_key"] == "ba"
    assert _layer(migrated.iloc[0], "external_stats")["camera_names"] == "ba,bb"
    assert _layer(migrated.iloc[0], "external_stats")["camera_name_count"] == 2
    assert _layer(migrated.iloc[0], "external_stats")["camera_name_key_fraction"] == 0.5
    assert "camera_field_key" not in _layer(migrated.iloc[0], "external_stats")
    assert _layer(migrated.iloc[0], "derived_stats")["derived_harmonics_a4_a2"] == 3.0
    assert (out_root / "runs" / "stv_run" / "plots" / "C1.png").read_text(encoding="utf-8") == "plot\n"
    assert (out_root / "migration_report.json").exists()
    assert (out_root / "unclassified_columns.json").exists()


def test_migrate_tree_rewrites_product_csv_camera_field_columns(tmp_path: Path) -> None:
    root = tmp_path / "output"
    product = root / "runs" / "stv_run" / "results" / "lc_events_filtered.csv"
    product.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "path": ["C1.dat2"],
            "dip_significant": [True],
            "camera_field_key": ["ba/F1"],
            "camera_fields": ["ba/F1,bb/F1"],
            "camera_field_count": [2],
            "camera_field_key_fraction": [0.25],
        }
    ).to_csv(product, index=False)

    out_root = tmp_path / "migrated"
    summary = migrate_tree(root, out_root)

    assert summary.ok is True
    migrated = pd.read_csv(out_root / "runs" / "stv_run" / "results" / "lc_events_filtered.csv")
    assert "camera_field_key" not in migrated.columns
    external = json.loads(migrated.loc[0, "external_stats"])
    assert external["camera_name_key"] == "ba"
    assert external["camera_names"] == "ba,bb"
    assert external["camera_name_count"] == 2


def test_migrate_tree_rewrites_camera_fields_when_timescale_inference_falls_back(tmp_path: Path) -> None:
    root = tmp_path / "output"
    product = root / "runs" / "ltv_march18" / "results" / "already_layered.parquet"
    product.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "candidate_id": ["C1"],
            "lc_stats": ["{}"],
            "external_stats": ["{}"],
            "derived_stats": ["{}"],
            "camera_field_key": ["ba/F1"],
            "camera_fields": ["ba/F1,bb/F1"],
            "camera_field_count": [2],
            "camera_field_key_fraction": [0.25],
        }
    ).to_parquet(product, index=False)

    out_root = tmp_path / "migrated"
    summary = migrate_tree(root, out_root)

    assert summary.ok is True
    migrated = pd.read_parquet(out_root / "runs" / "ltv_march18" / "results" / "already_layered.parquet")
    assert "camera_field_key" not in migrated.columns
    external = _layer(migrated.iloc[0], "external_stats")
    assert external["camera_name_key"] == "ba"
    assert external["camera_names"] == "ba,bb"


def test_migrate_tree_preserves_dangling_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "output"
    product = root / "runs" / "stv_run" / "results" / "lc_events_filtered.parquet"
    product.parent.mkdir(parents=True)
    pd.DataFrame({"path": ["C1.dat2"], "dip_significant": [True]}).to_parquet(product, index=False)

    latest = root / "runs" / "ltv" / "latest"
    latest.parent.mkdir(parents=True)
    target = tmp_path / "pytest-temp-run-that-no-longer-exists"
    latest.symlink_to(target)
    assert latest.is_symlink()
    assert not latest.exists()

    out_root = tmp_path / "migrated"
    summary = migrate_tree(root, out_root)

    mirrored = out_root / "runs" / "ltv" / "latest"
    assert summary.ok is True
    assert mirrored.is_symlink()
    assert mirrored.readlink() == target


def test_migrate_tree_rewrites_review_db_payload_but_keeps_sql_filter_columns(tmp_path: Path) -> None:
    root = tmp_path / "output"
    db_path = root / "review" / "review.db"
    with db_connect(db_path) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "C1",
                        "asas_sn_id": "C1",
                        "dipper_score": 6.0,
                        "camera_field_key": "ba/F1",
                        "camera_fields": "ba/F1,ba/F2,bb/F1",
                        "camera_field_count": 3,
                        "camera_field_key_fraction": 0.5,
                        "phot_g_mean_mag": 14.2,
                    }
                ]
            ),
        )
        conn.execute("ALTER TABLE candidates ADD COLUMN camera_field_key TEXT")
        conn.execute("ALTER TABLE candidates ADD COLUMN camera_fields TEXT")
        conn.execute("ALTER TABLE candidates ADD COLUMN camera_field_count REAL")
        conn.execute("ALTER TABLE candidates ADD COLUMN camera_field_key_fraction REAL")
        conn.execute(
            """
            UPDATE candidates
            SET camera_field_key = 'ba/F1',
                camera_fields = 'ba/F1,ba/F2,bb/F1',
                camera_field_count = 3,
                camera_field_key_fraction = 0.5
            WHERE candidate_id = 'C1'
            """
        )
        conn.commit()

    out_root = tmp_path / "migrated"
    summary = migrate_tree(root, out_root)

    assert summary.ok is True
    migrated_db = out_root / "review" / "review.db"
    with sqlite3.connect(migrated_db) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(candidates)").fetchall()}
        row = conn.execute(
            """
            SELECT dipper_score, payload_json, lc_stats, external_stats,
                   camera_name_key, camera_names, camera_name_count, camera_name_key_fraction
            FROM candidates WHERE candidate_id='C1'
            """
        ).fetchone()
    assert "camera_field_key" not in columns
    assert row[0] == 6.0
    payload = json.loads(row[1])
    assert "dipper_score" not in payload
    assert payload["lc_stats"]["dipper_score"] == 6.0
    assert payload["external_stats"]["phot_g_mean_mag"] == 14.2
    assert payload["external_stats"]["camera_name_key"] == "ba"
    assert "camera_field_key" not in payload["external_stats"]
    assert json.loads(row[2])["dipper_score"] == 6.0
    assert json.loads(row[3])["phot_g_mean_mag"] == 14.2
    assert json.loads(row[3])["camera_name_key"] == "ba"
    assert row[4] == "ba"
    assert row[5] == "ba,bb"
    assert row[6] == 2
    assert row[7] == 0.5


def test_migrate_tree_rewrites_review_db_paths_for_migrated_queue_scope(tmp_path: Path) -> None:
    root = tmp_path / "output"
    run_rel = Path("runs") / "march18_bundle"
    db_path = root / run_rel / "review" / "review.taxonomy_filled.db"
    old_output_root = Path("/home/calder/code/malca/output")
    old_run_dir = old_output_root / run_rel
    old_lc_path = old_run_dir / "bundle_assets" / "lightcurves" / "C1.dat3"
    old_input_file = old_run_dir / "results" / "lc_events_vetted.parquet"

    with db_connect(db_path) as conn:
        upsert_candidates_frame(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "C1",
                        "asas_sn_id": "C1",
                        "source_path": str(old_run_dir),
                        "lc_path": str(old_lc_path),
                        "path": str(old_lc_path),
                    }
                ]
            ),
        )
        save_app_state(conn, "last_input_file", str(old_input_file))

    out_root = tmp_path / "migrated_output"
    summary = migrate_tree(root, out_root)

    assert summary.ok is True
    migrated_db = out_root / run_rel / "review" / "review.taxonomy_filled.db"
    new_run_dir = (out_root / run_rel).resolve()
    new_lc_path = new_run_dir / "bundle_assets" / "lightcurves" / "C1.dat3"
    new_input_file = new_run_dir / "results" / "lc_events_vetted.parquet"

    with db_connect(migrated_db) as conn:
        source_path, lc_path, payload_json, camera_name_count = conn.execute(
            """
            SELECT source_path, lc_path, payload_json, camera_name_count
            FROM candidates WHERE candidate_id='C1'
            """
        ).fetchone()
        saved_input = conn.execute("SELECT value FROM app_state WHERE key='last_input_file'").fetchone()[0]
        assert source_path == str(new_run_dir)
        assert lc_path == str(new_lc_path)
        assert saved_input == str(new_input_file)
        assert "/home/calder/code/malca/output" not in payload_json
        assert camera_name_count is None
        assert count_queue(conn, filters={"source_paths": [str(new_run_dir)]}) == 1


def test_migrate_tree_rewrites_candidates_jsonl_and_import_indexes_layers(tmp_path: Path) -> None:
    root = tmp_path / "output"
    transfer = root / "transfer" / "bundle"
    _write_jsonl(
        transfer / "candidates.jsonl",
        [
            {
                "schema_version": 2,
                "candidate_id": "C1",
                "asas_sn_id": "C1",
                "dipper_score": 6.0,
                "camera_field_key": "ba/F1",
                "payload": {
                    "phot_g_mean_mag": 14.2,
                    "camera_fields": "ba/F1,bb/F1",
                    "camera_field_count": 2,
                    "camera_field_key_fraction": 0.5,
                    "custom_nested": {"ok": True},
                },
            }
        ],
    )
    _write_jsonl(transfer / "reviews.jsonl", [])
    _write_manifest(transfer / "assets_manifest.json")

    out_root = tmp_path / "migrated"
    summary = migrate_tree(root, out_root)

    assert summary.ok is True
    migrated_transfer = out_root / "transfer" / "bundle"
    record = json.loads((migrated_transfer / "candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "dipper_score" not in record
    assert "camera_field_key" not in record
    assert record["payload"]["lc_stats"]["dipper_score"] == 6.0
    assert record["payload"]["external_stats"]["phot_g_mean_mag"] == 14.2
    assert record["payload"]["external_stats"]["camera_name_key"] == "ba"
    assert record["payload"]["external_stats"]["camera_names"] == "ba,bb"
    assert "camera_field_key" not in record["payload"]["external_stats"]
    assert record["payload"]["custom_nested"]["ok"] is True

    rebuilt = tmp_path / "rebuilt.db"
    sync.import_review_bundle(migrated_transfer, db_path=rebuilt, replace=True)
    with sqlite3.connect(rebuilt) as conn:
        sql_value = conn.execute("SELECT dipper_score FROM candidates WHERE candidate_id='C1'").fetchone()[0]
        payload = json.loads(conn.execute("SELECT payload_json FROM candidates WHERE candidate_id='C1'").fetchone()[0])
    assert sql_value == 6.0
    assert payload["lc_stats"]["dipper_score"] == 6.0


def test_python_migrate_scan_only_writes_report_without_mirror(tmp_path: Path) -> None:
    root = tmp_path / "output"
    product = root / "runs" / "stv_run" / "results" / "lc_events_filtered.parquet"
    product.parent.mkdir(parents=True)
    pd.DataFrame({"path": ["C1.dat2"], "dip_significant": [True]}).to_parquet(product, index=False)
    report = tmp_path / "scan.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "migrate",
            "--input",
            str(root),
            "--scan-only",
            "--report",
            str(report),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert report.exists()
    payload = json.loads(report.read_text(encoding="ascii"))
    assert payload[0]["action"] == "would_migrate"
