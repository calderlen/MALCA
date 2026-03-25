from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_rewrite_ltv_id_columns_script_rewrites_parquet_in_place(tmp_path: Path) -> None:
    target = tmp_path / "LTvar12-12.5_pipeline.parquet"
    pd.DataFrame(
        {
            "ASAS-SN ID": [123, 456],
            "Slope": [0.1, 0.2],
        }
    ).to_parquet(target, index=False)

    script = Path(__file__).resolve().parents[1] / "scripts" / "rewrite_ltv_id_columns.py"
    subprocess.run(
        [sys.executable, str(script), str(tmp_path), "--write"],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    out = pd.read_parquet(target)
    assert "asas_sn_id" in out.columns
    assert "ASAS-SN ID" not in out.columns
    assert out["asas_sn_id"].tolist() == [123, 456]
