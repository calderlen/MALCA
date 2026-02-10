from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from malca.detect import export_bundle_zip


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_export_bundle_includes_candidate_dat2_and_raw2(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    lc_dir = tmp_path / "lc_source"
    dat2_path = lc_dir / "ASASSN-TEST-001.dat2"
    raw2_path = lc_dir / "ASASSN-TEST-001.raw2"
    _write_text(dat2_path, "dat2 content\n")
    _write_text(raw2_path, "raw2 content\n")

    pd.DataFrame({"path": [str(dat2_path)]}).to_parquet(results_dir / "lc_events_filtered.parquet", index=False)

    bundle_zip = tmp_path / "bundle.zip"
    bundled = export_bundle_zip(bundle_zip, out_dir)

    expected_dat2 = "bundle_assets/lightcurves/ASASSN-TEST-001.dat2"
    expected_raw2 = "bundle_assets/lightcurves/ASASSN-TEST-001.raw2"

    assert expected_dat2 in bundled
    assert expected_raw2 in bundled

    with zipfile.ZipFile(bundle_zip, "r") as zf:
        names = set(zf.namelist())
        assert "results/lc_events_filtered.parquet" in names
        assert expected_dat2 in names
        assert expected_raw2 in names
