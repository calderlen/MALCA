from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from malca import tag


def _write_mock_dat(path: Path) -> None:
    lines = [
        "1000.0 14.0 0.05 1 1 0 0 cam1/field1",
        "1010.0 14.1 0.05 1 1 0 0 cam1/field1",
        "1020.0 14.2 0.05 1 1 0 0 cam1/field1",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_apply_tags_vsx_tag_mode_keeps_rows() -> None:
    df = pd.DataFrame(
        {
            "source_id": ["1001", "1002"],
            "path": ["/tmp/lc_a", "/tmp/lc_b"],
            "vsx_sep_arcsec": [0.3, 5.0],
            "vsx_class": ["EA", ""],
        }
    )

    out = tag.apply_tags(
        df,
        apply_vsx=True,
        vsx_mode="tag",
        apply_sparse=False,
        apply_multi_camera=False,
        apply_mag_range=False,
        show_tqdm=False,
    )

    assert len(out) == 2
    assert "vsx_sep_arcsec" in out.columns
    assert "vsx_class" in out.columns
    assert "failed_vsx_match" not in out.columns
    assert out["source_id"].astype(str).tolist() == ["1001", "1002"]


def test_apply_tags_rejects_legacy_vsx_filter_mode() -> None:
    df = pd.DataFrame({"source_id": ["1001"], "path": ["/tmp/lc_a"]})

    with pytest.raises(ValueError, match="vsx_mode must be 'tag'"):
        tag.apply_tags(
            df,
            apply_vsx=False,
            vsx_mode="filter",
            apply_sparse=False,
            apply_multi_camera=False,
            apply_mag_range=False,
            show_tqdm=False,
        )


def test_apply_tags_honors_file_extension_override(tmp_path: Path) -> None:
    flat_dir = tmp_path / "flat"
    _write_mock_dat(flat_dir / "2001.dat2")

    df = pd.DataFrame(
        {
            "source_id": ["2001"],
            "path": [str(flat_dir)],
            "mag_bin": ["13_13.5"],
        }
    )

    out = tag.apply_tags(
        df,
        apply_vsx=False,
        apply_sparse=True,
        min_time_span=1.0,
        min_points_per_day=0.01,
        apply_multi_camera=False,
        apply_mag_range=False,
        show_tqdm=False,
        file_ext="dat2",
    )

    assert bool(out.loc[0, "failed_sparse"]) is False
