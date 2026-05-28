from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from malca.schema_migration import (
    convert_product_file,
    convert_product_frame,
    scan_product,
)


def test_convert_stv_product_frame_adds_canonical_identity() -> None:
    out = convert_product_frame(
        pd.DataFrame({"path": ["/data/ASASSN-1.dat2"], "dip_significant": [True]}),
        "stv",
    )

    assert out["lc_path"].tolist() == ["/data/ASASSN-1.dat2"]
    assert out["candidate_id"].tolist() == ["stv_ASASSN-1"]
    assert out["timescale"].tolist() == ["stv"]
    assert "path" not in out.columns


def test_convert_ltv_product_frame_maps_legacy_core_and_context_columns() -> None:
    out = convert_product_frame(
        pd.DataFrame(
            {
                "asas_sn_id": ["123"],
                "lc_path": ["/data/123.dat2"],
                "ra_deg": [10.0],
                "dec_deg": [-5.0],
                "Pstarss gmag": [13.2],
                "Slope": [0.4],
                "Quad Slope": [0.01],
                "max diff": [0.6],
                "Median": [13.1],
                "Median_err": [0.02],
                "Dispersion": [0.1],
                "n_seasons": [4],
                "ltv_filter_reason": ["passed"],
                "ltv_passed_filters": [1],
                "neowise_n_epochs": [3],
                "M_G": [4.5],
                "M_G0": [4.1],
                "bp_rp0": [1.2],
            }
        ),
        "ltv",
    )

    assert out.loc[0, "candidate_id"] == "ltv_123"
    assert out.loc[0, "timescale"] == "ltv"
    assert out.loc[0, "ra"] == 10.0
    assert out.loc[0, "dec"] == -5.0
    assert out.loc[0, "baseline_mag"] == 13.2
    assert out.loc[0, "ltv_slope"] == 0.4
    assert out.loc[0, "ltv_slope_quad"] == 0.01
    assert out.loc[0, "ltv_max_diff"] == 0.6
    assert out.loc[0, "ltv_n_seasons"] == 4
    assert out.loc[0, "filter_reason"] == "passed"
    assert out.loc[0, "ltv_neowise_n_epochs"] == 3
    assert out.loc[0, "mg"] == 4.5
    assert out.loc[0, "mg0"] == 4.1
    assert out.loc[0, "bprp0"] == 1.2
    assert "ltv_passed_filters" not in out.columns


def test_convert_product_frame_conflicting_aliases_fail_by_default() -> None:
    df = pd.DataFrame(
        {
            "asas_sn_id": ["123"],
            "lc_path": ["/data/123.dat2"],
            "ra": [11.0],
            "ra_deg": [10.0],
            "dec": [-5.0],
        }
    )

    with pytest.raises(ValueError, match="Conflicting legacy/canonical columns"):
        convert_product_frame(df, "ltv")


def test_scan_product_detects_legacy_columns(tmp_path: Path) -> None:
    product = tmp_path / "output" / "runs" / "stv" / "old" / "results" / "lc_events_filtered.parquet"
    product.parent.mkdir(parents=True)
    pd.DataFrame({"path": ["a.dat2"], "dip_significant": [True]}).to_parquet(product, index=False)

    scan = scan_product(product)

    assert scan.timescale == "stv"
    assert scan.needs_conversion is True
    assert scan.legacy_columns == ["path"]


def test_convert_product_file_handles_chunked_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "output" / "runs" / "stv" / "old" / "results" / "lc_events_results"
    dataset.mkdir(parents=True)
    pd.DataFrame({"path": ["a.dat2"], "dip_significant": [True]}).to_parquet(
        dataset / "chunk_000000.parquet",
        index=False,
    )
    output = tmp_path / "migrated" / "lc_events_results"

    result = convert_product_file(dataset, output, overwrite=True)

    assert result.wrote is True
    converted = pd.read_parquet(output)
    assert converted["lc_path"].tolist() == ["a.dat2"]
    assert converted["candidate_id"].tolist() == ["stv_a"]


def test_convert_product_schema_dry_run_writes_report_only(tmp_path: Path) -> None:
    product = tmp_path / "run" / "results" / "LTvar13-13.5_pipeline.parquet"
    product.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "asas_sn_id": ["123"],
            "lc_path": ["/data/123.dat2"],
            "ra_deg": [1.0],
            "dec_deg": [2.0],
            "Slope": [0.1],
        }
    ).to_parquet(product, index=False)
    report = tmp_path / "report.json"
    output_dir = tmp_path / "migrated"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/convert_product_schema.py",
            str(product.parent.parent),
            "--timescale",
            "ltv",
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
    assert not output_dir.exists()
    payload = json.loads(report.read_text(encoding="ascii"))
    assert payload[0]["needs_conversion"] is True


def test_convert_product_schema_in_place_requires_backup_dir(tmp_path: Path) -> None:
    product = tmp_path / "run" / "results" / "lc_events_filtered.parquet"
    product.parent.mkdir(parents=True)
    pd.DataFrame({"path": ["a.dat2"], "dip_significant": [True]}).to_parquet(product, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/convert_product_schema.py",
            str(product.parent.parent),
            "--write",
            "--in-place",
            "--timescale",
            "stv",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "--in-place requires --backup-dir" in completed.stderr
