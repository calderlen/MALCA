from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.enrichment import vetting
from malca.external_lc_manifest import (
    EXTERNAL_LC_MANIFEST_FILE,
    clear_external_lc_manifest_caches,
    index_external_lc_paths_from_manifest,
    lookup_external_lc_paths_from_manifest,
    read_external_lc_manifest,
    upsert_external_lc_manifest_entry,
    write_external_lc_manifest,
)
from malca.review.lightcurve_sources import clear_external_lc_discovery_caches, discover_external_lcs


def _write_lc(path: Path, rows: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "mjd": [59000.0 + i for i in range(rows)],
            "mag": [12.0 + i * 0.1 for i in range(rows)],
            "mag_err": [0.03 for _ in range(rows)],
        }
    ).to_parquet(path, index=False)
    return path


def test_manifest_lookup_avoids_rglob_when_manifest_exists(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "results"
    lc_path = _write_lc(root / "neowise_lc_123.parquet")
    assert upsert_external_lc_manifest_entry(
        root,
        candidate_id="123",
        source="neowise",
        file_prefix="neowise",
        path=lc_path,
    )
    clear_external_lc_manifest_caches()

    def fail_rglob(*_args, **_kwargs):
        raise AssertionError("manifest lookup should not scan")

    monkeypatch.setattr(Path, "rglob", fail_rglob)

    mapping = index_external_lc_paths_from_manifest(str(root), "neowise")

    assert mapping == {"123": str(lc_path)}


def test_targeted_manifest_lookup_never_scans_unrelated_files(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "results"
    lc_path = _write_lc(root / "external_lcs" / "tess_lc_C1.parquet")
    assert upsert_external_lc_manifest_entry(
        root,
        candidate_id="C1",
        source="tess",
        file_prefix="tess",
        path=lc_path,
    )
    clear_external_lc_manifest_caches()

    def fail_rglob(*_args, **_kwargs):
        raise AssertionError("targeted review lookup must not scan the results tree")

    monkeypatch.setattr(Path, "rglob", fail_rglob)

    mapping = lookup_external_lc_paths_from_manifest(
        root,
        ["tess", "atlas"],
        ["C1", "missing"],
    )

    assert mapping == {"tess": {"C1": str(lc_path)}, "atlas": {}}


def test_missing_manifest_falls_back_to_scan_and_writes_manifest(tmp_path: Path) -> None:
    root = tmp_path / "results"
    lc_path = _write_lc(root / "crts_lc_123.parquet")
    clear_external_lc_manifest_caches()

    mapping = index_external_lc_paths_from_manifest(str(root), "crts")

    assert mapping == {"123": str(lc_path)}
    assert (root / EXTERNAL_LC_MANIFEST_FILE).exists()
    manifest = read_external_lc_manifest(root)
    assert manifest[["candidate_id", "file_prefix", "path_relative"]].to_dict("records") == [
        {"candidate_id": "123", "file_prefix": "crts", "path_relative": "crts_lc_123.parquet"}
    ]


def test_stale_manifest_row_falls_back_to_scan_and_repairs_manifest(tmp_path: Path) -> None:
    root = tmp_path / "results"
    old_path = root / "old" / "neowise_lc_123.parquet"
    new_path = _write_lc(root / "neowise_lc_123.parquet")
    write_external_lc_manifest(
        root,
        pd.DataFrame(
            [
                {
                    "candidate_id": "123",
                    "source": "neowise",
                    "file_prefix": "neowise",
                    "path": str(old_path),
                    "path_relative": "old/neowise_lc_123.parquet",
                    "size_bytes": 1,
                    "mtime_ns": 1,
                    "updated_unix": 1.0,
                }
            ]
        ),
    )
    clear_external_lc_manifest_caches()

    mapping = index_external_lc_paths_from_manifest(str(root), "neowise")

    assert mapping == {"123": str(new_path)}
    manifest = read_external_lc_manifest(root)
    assert manifest.loc[0, "path_relative"] == "neowise_lc_123.parquet"
    assert int(manifest.loc[0, "size_bytes"]) == new_path.stat().st_size
    assert int(manifest.loc[0, "mtime_ns"]) == new_path.stat().st_mtime_ns


def test_neowise_aliases_resolve_same_cached_file(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    root = run_dir / "results"
    lc_path = _write_lc(root / "neowise_lc_123.parquet")
    assert upsert_external_lc_manifest_entry(
        root,
        candidate_id="123",
        source="neowise",
        file_prefix="neowise",
        path=lc_path,
    )
    clear_external_lc_discovery_caches()

    def fail_rglob(*_args, **_kwargs):
        raise AssertionError("alias lookup should use the manifest")

    monkeypatch.setattr(Path, "rglob", fail_rglob)

    found = discover_external_lcs(
        "123",
        {"candidate_id": "123"},
        run_dir,
        ["neowise", "neowise_w1", "neowise_w2", "neowise_color"],
    )

    assert found == {
        "neowise": lc_path,
        "neowise_w1": lc_path,
        "neowise_w2": lc_path,
        "neowise_color": lc_path,
    }


def test_external_lc_writer_upserts_manifest_without_duplicate_rows(tmp_path: Path) -> None:
    df = pd.DataFrame([{"candidate_id": "123"}])
    vetting._write_external_lc_file(
        tmp_path,
        "neowise_lc",
        df,
        0,
        pd.DataFrame({"mjd": [59000.0], "w1mpro": [12.0], "w1sigmpro": [0.03]}),
    )

    manifest = read_external_lc_manifest(tmp_path)
    assert len(manifest) == 1
    assert manifest.loc[0, "candidate_id"] == "123"
    assert manifest.loc[0, "file_prefix"] == "neowise"

    vetting._write_external_lc_file(
        tmp_path,
        "neowise_lc",
        df,
        0,
        pd.DataFrame(
            {
                "mjd": [59000.0, 59001.0],
                "w1mpro": [12.0, 12.2],
                "w1sigmpro": [0.03, 0.04],
            }
        ),
    )

    path = tmp_path / "neowise_lc_123.parquet"
    manifest = read_external_lc_manifest(tmp_path)
    assert len(manifest) == 1
    assert int(manifest.loc[0, "size_bytes"]) == path.stat().st_size
    assert int(manifest.loc[0, "mtime_ns"]) == path.stat().st_mtime_ns
