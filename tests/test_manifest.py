from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from malca.io.manifest import build_manifest


def _write_mock_lc(path: Path) -> None:
    lines = [
        "1000.0 14.0 0.05 1 1 0 0 cam1/field1",
        "1010.0 14.1 0.05 1 1 0 0 cam1/field1",
        "1020.0 14.2 0.05 1 1 0 0 cam1/field1",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_build_manifest_supports_flat_lightcurve_directory_without_index(tmp_path: Path) -> None:
    flat_dir = tmp_path / "bundle_assets" / "lightcurves"
    _write_mock_lc(flat_dir / "1001.dat3")
    _write_mock_lc(flat_dir / "1002.dat3")

    manifest = build_manifest(
        None,
        None,
        mag_bins=["13_13.5"],
        id_column="asas_sn_id",
        flat_lc_dir=flat_dir,
        show_progress=False,
    )

    assert manifest["source_id"].tolist() == ["1001", "1002"]
    assert manifest["mag_bin"].tolist() == ["13_13.5", "13_13.5"]
    assert manifest["lc_dir"].tolist() == [str(flat_dir), str(flat_dir)]
    assert manifest["dat_exists"].tolist() == [True, True]
    assert manifest["dat_path"].map(lambda value: Path(value).name).tolist() == ["1001.dat3", "1002.dat3"]


def test_build_manifest_supports_flat_lightcurve_directory_with_index_metadata(tmp_path: Path) -> None:
    flat_dir = tmp_path / "bundle_assets" / "lightcurves"
    _write_mock_lc(flat_dir / "1001.dat3")
    _write_mock_lc(flat_dir / "1002.dat3")

    index_file = tmp_path / "concatenated_index.parquet"
    pd.DataFrame(
        {
            "asas_sn_id": ["1001", "1002"],
            "mag_bin": ["13_13.5", "14_14.5"],
            "index_num": [1, 2],
        }
    ).to_parquet(index_file, index=False)

    manifest = build_manifest(
        None,
        None,
        mag_bins=["14_14.5"],
        id_column="asas_sn_id",
        flat_lc_dir=flat_dir,
        index_file=index_file,
        show_progress=False,
    )

    assert manifest["source_id"].tolist() == ["1002"]
    assert manifest["mag_bin"].tolist() == ["14_14.5"]
    assert manifest["index_num"].tolist() == [2]
    assert manifest["index_csv"].tolist() == [str(index_file)]
    assert manifest["dat_path"].map(lambda value: Path(value).name).tolist() == ["1002.dat3"]


def test_flat_manifest_rejects_empty_input_directory(tmp_path: Path) -> None:
    flat_dir = tmp_path / "empty"
    flat_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No .* light curves"):
        build_manifest(
            None,
            None,
            mag_bins=["13_13.5"],
            id_column="asas_sn_id",
            flat_lc_dir=flat_dir,
            show_progress=False,
        )


def test_flat_manifest_rejects_conflicting_duplicate_index_metadata(tmp_path: Path) -> None:
    flat_dir = tmp_path / "lightcurves"
    _write_mock_lc(flat_dir / "1001.dat3")
    index_file = tmp_path / "index.parquet"
    pd.DataFrame(
        {
            "asas_sn_id": ["1001", "1001"],
            "mag_bin": ["13_13.5", "14_14.5"],
            "index_num": [1, 2],
        }
    ).to_parquet(index_file, index=False)

    with pytest.raises(ValueError, match="conflicting duplicate"):
        build_manifest(
            None,
            None,
            mag_bins=["13_13.5", "14_14.5"],
            id_column="asas_sn_id",
            flat_lc_dir=flat_dir,
            index_file=index_file,
            show_progress=False,
        )


def test_hierarchical_manifest_rejects_source_mapped_to_two_locations(tmp_path: Path) -> None:
    index_root = tmp_path / "indexes"
    lc_root = tmp_path / "lcs"
    for mag_bin in ("13_13.5", "14_14.5"):
        index_dir = index_root / mag_bin
        index_dir.mkdir(parents=True)
        pd.DataFrame({"asas_sn_id": ["1001"]}).to_csv(index_dir / "index1.csv", index=False)
        _write_mock_lc(lc_root / mag_bin / "lc1_cal" / "1001.dat3")

    with pytest.raises(ValueError, match="conflicting light-curve locations"):
        build_manifest(
            index_root,
            lc_root,
            mag_bins=["13_13.5", "14_14.5"],
            id_column="asas_sn_id",
            show_progress=False,
        )
