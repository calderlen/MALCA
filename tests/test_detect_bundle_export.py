from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from malca.config import ASASSN_INDEX_PATH
from malca.stv.pipeline import _resolve_asassn_index_path, export_bundle_zip
from malca.io.table_io import write_feature_table


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

    write_feature_table(
        pd.DataFrame(
            {
                "candidate_id": ["stv_ASASSN-TEST-001"],
                "timescale": ["stv"],
                "lc_path": [str(dat2_path)],
                "failed_any": [False],
            }
        ),
        results_dir / "lc_events_filtered.parquet",
    )

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


def test_export_bundle_includes_chunked_mag_bin_event_results(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    result_dir = out_dir / "results" / "lc_events_results_13_13.5"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "chunk_000000.parquet").write_bytes(b"chunk bytes\n")

    bundle_zip = tmp_path / "bundle.zip"
    bundled = export_bundle_zip(bundle_zip, out_dir, mag_bin_tag="13_13.5")

    expected = "results/lc_events_results_13_13.5/chunk_000000.parquet"
    assert expected in bundled
    with zipfile.ZipFile(bundle_zip, "r") as zf:
        assert expected in set(zf.namelist())


def test_export_bundle_includes_vetted_and_extended_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    results_dir = out_dir / "results"
    external_dir = results_dir / "external_lcs"
    external_dir.mkdir(parents=True, exist_ok=True)
    for rel in [
        "results/lc_events_vetted.parquet",
        "results/lc_events_external_lcs.parquet",
        "results/lc_events_multi_survey_features.parquet",
        "results/external_lcs/ztf_lc_C1.parquet",
    ]:
        (out_dir / rel).write_bytes(b"placeholder\n")

    bundle_zip = tmp_path / "bundle.zip"
    bundled = export_bundle_zip(bundle_zip, out_dir)

    expected = {
        "results/lc_events_vetted.parquet",
        "results/lc_events_external_lcs.parquet",
        "results/lc_events_multi_survey_features.parquet",
        "results/external_lcs/ztf_lc_C1.parquet",
    }
    assert expected.issubset(set(bundled))
    with zipfile.ZipFile(bundle_zip, "r") as zf:
        assert expected.issubset(set(zf.namelist()))


def test_resolve_index_prefers_bundle_assets_file(tmp_path: Path) -> None:
    out_dir = tmp_path / "output" / "runs" / "run_a"
    bundled_index = out_dir / "bundle_assets" / "asassn_index_full.parquet"
    _write_text(bundled_index, "bundle index\n")

    output_root_index = out_dir.parents[1] / ASASSN_INDEX_PATH.name
    _write_text(output_root_index, "output root index\n")

    resolved, _ = _resolve_asassn_index_path(out_dir)

    assert resolved == bundled_index


def test_resolve_index_falls_back_to_output_root(tmp_path: Path) -> None:
    out_dir = tmp_path / "output" / "runs" / "run_b"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_root_index = out_dir.parents[1] / ASASSN_INDEX_PATH.name
    _write_text(output_root_index, "output root index\n")

    resolved, _ = _resolve_asassn_index_path(out_dir)

    assert resolved == output_root_index


def test_resolve_index_honors_explicit_override(tmp_path: Path) -> None:
    out_dir = tmp_path / "output" / "runs" / "run_c"
    out_dir.mkdir(parents=True, exist_ok=True)

    override_index = tmp_path / "custom" / "manual_index.parquet"
    _write_text(override_index, "manual index\n")

    resolved, _ = _resolve_asassn_index_path(out_dir, index_override=override_index)

    assert resolved == override_index
