from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from malca.vsx.filter import colspecs, vsx_columns
from scripts import download_vsx_catalog
from scripts.download_vsx_catalog import convert_vsx_raw_to_parquet


def _vsx_fwf_line(values: dict[str, object]) -> str:
    chars = [" "] * max(stop for _start, stop in colspecs)
    for (start, stop), name in zip(colspecs, vsx_columns):
        width = stop - start
        text = str(values.get(name, ""))[:width]
        chars[start:stop] = list(text.ljust(width))
    return "".join(chars)


def test_convert_vsx_raw_to_parquet_filters_missing_coordinates(tmp_path: Path) -> None:
    raw_path = tmp_path / "vsx.dat"
    output_path = tmp_path / "vsx_all.parquet"
    raw_path.write_text(
        "\n".join(
            [
                _vsx_fwf_line(
                    {
                        "id_vsx": 101,
                        "name": "VSX Good",
                        "var_flag": 0,
                        "ra": "10.000000",
                        "dec": "20.00000",
                        "class": "EA",
                        "period": "1.250000",
                    }
                ),
                _vsx_fwf_line(
                    {
                        "id_vsx": 202,
                        "name": "VSX Missing RA",
                        "var_flag": 1,
                        "dec": "21.00000",
                        "class": "VAR",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = convert_vsx_raw_to_parquet(
        raw_path=raw_path,
        output_path=output_path,
        chunk_rows=1,
    )
    out = pd.read_parquet(output_path)

    assert rows == 1
    assert out[["id_vsx", "name", "var_flag", "ra", "dec", "class", "period"]].to_dict("records") == [
        {
            "id_vsx": 101,
            "name": "VSX Good",
            "var_flag": 0,
            "ra": 10.0,
            "dec": 20.0,
            "class": "EA",
            "period": 1.25,
        }
    ]
    marker = json.loads(output_path.with_name("vsx_all.parquet.complete.json").read_text())
    assert marker["row_count"] == 1
    assert marker["raw_path"] == str(raw_path)


def test_convert_vsx_raw_to_parquet_requires_overwrite(tmp_path: Path) -> None:
    raw_path = tmp_path / "vsx.dat"
    output_path = tmp_path / "vsx_all.parquet"
    raw_path.write_text(_vsx_fwf_line({"id_vsx": 101, "ra": "10.0", "dec": "20.0"}) + "\n")
    output_path.write_text("placeholder")

    with pytest.raises(FileExistsError):
        convert_vsx_raw_to_parquet(raw_path=raw_path, output_path=output_path)


def test_verify_vsx_raw_complete_rejects_partial_file(monkeypatch, tmp_path: Path) -> None:
    raw_path = tmp_path / "vsx.dat"
    raw_path.write_bytes(b"x" * 10)

    monkeypatch.setattr(download_vsx_catalog, "_remote_size", lambda session, url, timeout: 20)

    with pytest.raises(RuntimeError, match="raw file is incomplete"):
        download_vsx_catalog.verify_vsx_raw_complete(raw_path=raw_path)
