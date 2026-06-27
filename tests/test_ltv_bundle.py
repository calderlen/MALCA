from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from malca.ltv.bundle import export_ltv_bundle
from malca.io.table_io import write_feature_table


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="ascii")


def test_export_ltv_bundle_single_file(tmp_path: Path) -> None:
    lc_dir = tmp_path / "lc"
    kept = lc_dir / "KEEP-1.dat2"
    rejected = lc_dir / "DROP-1.dat2"
    _write_text(kept, "kept\n")
    _write_text(rejected, "drop\n")

    input_path = tmp_path / "LTvar13-13.5.parquet"
    write_feature_table(
        pd.DataFrame(
            {
                "candidate_id": ["ltv_KEEP-1", "ltv_DROP-1"],
                "timescale": ["ltv", "ltv"],
                "asas_sn_id": ["KEEP-1", "DROP-1"],
                "ltv_slope": [0.08, 0.01],
                "ltv_max_diff": [0.4, 0.4],
                "lc_path": [str(kept), str(rejected)],
            }
        ),
        input_path,
    )

    bundle_zip = tmp_path / "single_bundle.zip"
    export_ltv_bundle(input_path, bundle_zip)

    with zipfile.ZipFile(bundle_zip, "r") as zf:
        names = set(zf.namelist())
        assert "lightcurves/KEEP-1.dat2" in names
        assert "lightcurves/DROP-1.dat2" not in names


def test_export_ltv_bundle_directory_collects_all_mag_bins(tmp_path: Path) -> None:
    out_dir = tmp_path / "output" / "ltv"
    out_dir.mkdir(parents=True, exist_ok=True)
    lc_dir = tmp_path / "lc"
    first = lc_dir / "BIN1-1.dat2"
    second = lc_dir / "BIN2-1.dat2"
    shared = lc_dir / "SHARED-1.dat2"
    _write_text(first, "first\n")
    _write_text(second, "second\n")
    _write_text(shared, "shared\n")

    write_feature_table(
        pd.DataFrame(
            {
                "candidate_id": ["ltv_BIN1-1", "ltv_SHARED-1a"],
                "timescale": ["ltv", "ltv"],
                "asas_sn_id": ["BIN1-1", "SHARED-1"],
                "ltv_slope": [0.08, 0.09],
                "ltv_max_diff": [0.4, 0.5],
                "lc_path": [str(first), str(shared)],
            }
        ),
        out_dir / "LTvar12-12.5_pipeline.parquet",
    )
    write_feature_table(
        pd.DataFrame(
            {
                "candidate_id": ["ltv_BIN2-1", "ltv_SHARED-1b"],
                "timescale": ["ltv", "ltv"],
                "asas_sn_id": ["BIN2-1", "SHARED-1"],
                "ltv_slope": [0.11, 0.12],
                "ltv_max_diff": [0.6, 0.7],
                "lc_path": [str(second), str(shared)],
            }
        ),
        out_dir / "LTvar12.5-13_pipeline.parquet",
    )

    bundle_zip = tmp_path / "all_bins_bundle.zip"
    export_ltv_bundle(out_dir, bundle_zip)

    with zipfile.ZipFile(bundle_zip, "r") as zf:
        names = set(zf.namelist())
        assert names == {
            "lightcurves/BIN1-1.dat2",
            "lightcurves/BIN2-1.dat2",
            "lightcurves/SHARED-1.dat2",
        }


def test_export_ltv_bundle_directory_pattern_override(tmp_path: Path) -> None:
    out_dir = tmp_path / "output" / "ltv"
    out_dir.mkdir(parents=True, exist_ok=True)
    lc_dir = tmp_path / "lc"
    wanted = lc_dir / "WANTED-1.dat2"
    skipped = lc_dir / "SKIPPED-1.dat2"
    _write_text(wanted, "wanted\n")
    _write_text(skipped, "skipped\n")

    write_feature_table(
        pd.DataFrame(
            {
                "candidate_id": ["ltv_WANTED-1"],
                "timescale": ["ltv"],
                "asas_sn_id": ["WANTED-1"],
                "ltv_slope": [0.08],
                "ltv_max_diff": [0.4],
                "lc_path": [str(wanted)],
            }
        ),
        out_dir / "custom_a.parquet",
    )
    write_feature_table(
        pd.DataFrame(
            {
                "candidate_id": ["ltv_SKIPPED-1"],
                "timescale": ["ltv"],
                "asas_sn_id": ["SKIPPED-1"],
                "ltv_slope": [0.08],
                "ltv_max_diff": [0.4],
                "lc_path": [str(skipped)],
            }
        ),
        out_dir / "custom_b.parquet",
    )

    bundle_zip = tmp_path / "custom_bundle.zip"
    export_ltv_bundle(out_dir, bundle_zip, pattern="custom_a.parquet")

    with zipfile.ZipFile(bundle_zip, "r") as zf:
        names = set(zf.namelist())
        assert names == {"lightcurves/WANTED-1.dat2"}
