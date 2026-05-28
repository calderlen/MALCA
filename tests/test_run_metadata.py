from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import json

from malca.run_metadata import (
    build_fingerprint,
    build_run_summary,
    fingerprint_digest,
    json_stable,
    load_summary_state,
    preserve_imported_run_snapshots,
)


def test_preserve_imported_run_snapshots_copies_imported_metadata_once(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    run_params_file = out_dir / "run_params.json"
    run_summary_file = out_dir / "run_summary.json"
    run_params_file.write_text('{"stage":"cluster"}', encoding="utf-8")
    run_summary_file.write_text('{"detection_stats":{"total_detections":42}}', encoding="utf-8")

    params_snapshot, summary_snapshot = preserve_imported_run_snapshots(
        stage="home",
        import_bundle=Path("bundle.zip"),
        out_dir=out_dir,
        run_params_file=run_params_file,
        run_summary_file=run_summary_file,
    )

    assert params_snapshot == out_dir / "run_params_imported.json"
    assert summary_snapshot == out_dir / "run_summary_imported.json"
    assert params_snapshot.read_text(encoding="utf-8") == run_params_file.read_text(encoding="utf-8")
    assert summary_snapshot.read_text(encoding="utf-8") == run_summary_file.read_text(encoding="utf-8")

    params_snapshot.write_text('{"stage":"preserved"}', encoding="utf-8")
    summary_snapshot.write_text('{"detection_stats":{"total_detections":99}}', encoding="utf-8")

    preserve_imported_run_snapshots(
        stage="home",
        import_bundle=Path("bundle.zip"),
        out_dir=out_dir,
        run_params_file=run_params_file,
        run_summary_file=run_summary_file,
    )

    assert params_snapshot.read_text(encoding="utf-8") == '{"stage":"preserved"}'
    assert summary_snapshot.read_text(encoding="utf-8") == '{"detection_stats":{"total_detections":99}}'


def test_load_summary_state_preserves_existing_summary_fields(tmp_path: Path) -> None:
    run_summary_file = tmp_path / "run_summary.json"
    run_summary_file.write_text(
        json.dumps({"detection_stats": {"total_detections": 123}}),
        encoding="utf-8",
    )

    start = datetime(2026, 3, 18, 12, 0, 0)
    summary_state = load_summary_state(
        run_summary_file=run_summary_file,
        run_start_time=start,
        stage="home",
    )

    assert summary_state["detection_stats"] == {"total_detections": 123}
    assert summary_state["run_info"] == {
        "start_time": start.isoformat(),
        "stage": "home",
    }


def test_build_run_summary_preserves_previous_manifest_and_detection_stats_for_home_replay() -> None:
    start = datetime(2026, 3, 18, 12, 0, 0)
    end = start + timedelta(minutes=5)
    previous_summary = {
        "manifest_stats": {
            "total_sources": 500,
            "filtered_sources": 125,
            "kept_fraction": 0.25,
        },
        "detection_stats": {
            "total_detections": 125,
            "unique_sources": 125,
        },
    }

    summary = build_run_summary(
        previous_summary=previous_summary,
        run_start_time=start,
        run_end_time=end,
        config_fingerprint={"filter": {"min_bayes_factor": 10.0}},
        run_upstream=False,
        manifest_total_sources=None,
        manifest_filtered_sources=None,
        artifact_context={"stage": "home", "bundle_lightcurve_count": 125},
    )

    assert summary["detection_stats"] == previous_summary["detection_stats"]
    assert summary["manifest_stats"] == previous_summary["manifest_stats"]
    assert summary["artifact_context"]["bundle_lightcurve_count"] == 125
    assert summary["run_info"]["duration_seconds"] == 300.0


def test_build_run_summary_uses_upstream_manifest_counts_when_available() -> None:
    start = datetime(2026, 3, 11, 8, 0, 0)
    end = start + timedelta(hours=2)

    summary = build_run_summary(
        previous_summary={},
        run_start_time=start,
        run_end_time=end,
        config_fingerprint={"filter": {"min_bayes_factor": 10.0}},
        run_upstream=True,
        manifest_total_sources=1000,
        manifest_filtered_sources=250,
        artifact_context={"stage": "cluster", "bundle_lightcurve_count": 0},
    )

    assert summary["manifest_stats"] == {
        "total_sources": 1000,
        "filtered_sources": 250,
        "kept_fraction": 0.25,
    }


def test_fingerprint_digest_is_stable_after_json_round_trip(tmp_path: Path) -> None:
    fingerprint = {
        "version": 1,
        "params": {"path": tmp_path / "input", "mag_bin": ("13_13.5", "13.5_14")},
        "code": {"stv/pipeline.py": "abc123"},
    }

    normalized = json_stable(fingerprint)
    round_tripped = json.loads(json.dumps(normalized))

    assert fingerprint_digest(fingerprint) == fingerprint_digest(round_tripped)


def test_build_fingerprint_stabilizes_params_and_hashes_code(tmp_path: Path) -> None:
    code = tmp_path / "module.py"
    code.write_text("print('ok')\n", encoding="ascii")

    fingerprint = build_fingerprint(
        version="1",
        params={"path": Path("output/runs/stv/example"), "bins": ("13_13.5",)},
        code_base=tmp_path,
        code_paths=["module.py", "missing.py"],
    )

    assert fingerprint["version"] == "1"
    assert fingerprint["params"] == {"bins": ["13_13.5"], "path": "output/runs/stv/example"}
    assert set(fingerprint["code"]) == {"module.py"}
