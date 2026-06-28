from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

from malca.io import manifest as manifest_cli
from malca.io.manifest import build_manifest


def _write_mock_lc(path: Path) -> None:
    lines = [
        "1000.0 14.0 0.05 1 1 0 0 cam1/field1",
        "1010.0 14.1 0.05 1 1 0 0 cam1/field1",
        "1020.0 14.2 0.05 1 1 0 0 cam1/field1",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_mock_lcsv2_root(root: Path, *, source_id: str = "1001") -> None:
    index_path = root / "13_13.5" / "index1.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"asas_sn_id": [source_id]}).to_csv(index_path, index=False)
    _write_mock_lc(root / "13_13.5" / "lc1_cal" / f"{source_id}.dat3")


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


def test_manifest_cli_requires_lcv2_root_without_flat_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MALCA_LCV2_ROOT", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifest",
            "--mag-bin",
            "13_13.5",
            "--output",
            str(tmp_path / "manifest.parquet"),
            "--workers",
            "1",
        ],
    )

    with pytest.raises(SystemExit, match="MALCA_LCV2_ROOT"):
        manifest_cli.main()


def test_manifest_cli_uses_lcv2_root_env(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "lcsv2"
    _write_mock_lcsv2_root(root)
    output = tmp_path / "manifest.parquet"
    monkeypatch.setenv("MALCA_LCV2_ROOT", str(root))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifest",
            "--mag-bin",
            "13_13.5",
            "--output",
            str(output),
            "--no-progress",
            "--workers",
            "1",
        ],
    )

    manifest_cli.main()

    manifest = pd.read_parquet(output)
    assert manifest["source_id"].tolist() == ["1001"]
    assert manifest["dat_exists"].tolist() == [True]
    assert manifest["index_csv"].map(lambda value: Path(value).parent.parent).tolist() == [root]


def test_manifest_cli_explicit_roots_override_env(tmp_path: Path, monkeypatch) -> None:
    env_root = tmp_path / "env_lcsv2"
    explicit_root = tmp_path / "explicit_lcsv2"
    _write_mock_lcsv2_root(env_root, source_id="env")
    _write_mock_lcsv2_root(explicit_root, source_id="explicit")
    output = tmp_path / "manifest.parquet"
    monkeypatch.setenv("MALCA_LCV2_ROOT", str(env_root))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifest",
            "--index-root",
            str(explicit_root),
            "--lc-root",
            str(explicit_root),
            "--mag-bin",
            "13_13.5",
            "--output",
            str(output),
            "--no-progress",
            "--workers",
            "1",
        ],
    )

    manifest_cli.main()

    manifest = pd.read_parquet(output)
    assert manifest["source_id"].tolist() == ["explicit"]
    assert Path(manifest.loc[0, "dat_path"]).is_relative_to(explicit_root)
