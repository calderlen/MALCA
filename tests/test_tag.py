from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.stv import tag


def _write_mock_dat(path: Path) -> None:
    lines = [
        "1000.0 14.0 0.05 1 1 0 0 cam1/field1",
        "1010.0 14.1 0.05 1 1 0 0 cam1/field1",
        "1020.0 14.2 0.05 1 1 0 0 cam1/field1",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_mock_raw2(lc_path: Path, lines: list[str]) -> None:
    lc_path.parent.mkdir(parents=True, exist_ok=True)
    lc_path.write_text("", encoding="ascii")
    lc_path.with_suffix(".raw2").write_text("\n".join(lines) + "\n", encoding="ascii")


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
    assert out.loc[0, "asassn_field_key"] == "field1"
    assert out.loc[0, "asassn_fields"] == "field1"
    assert int(out.loc[0, "asassn_field_count"]) == 1
    assert out.loc[0, "camera_name_key"] == "cam1"


def test_compute_stats_parallel_writes_incremental_checkpoint_parts(tmp_path: Path) -> None:
    flat_dir = tmp_path / "flat"
    _write_mock_dat(flat_dir / "2002.dat2")

    checkpoint = tmp_path / "stats.parquet"
    pd.DataFrame(
        {
            "source_id": ["2001"],
            "time_span_days": [123.0],
            "points_per_day": [4.5],
            "n_cameras": [2],
        }
    ).to_parquet(checkpoint, index=False)

    df = pd.DataFrame(
        {
            "source_id": ["2001", "2002"],
            "path": [str(flat_dir), str(flat_dir)],
        }
    )

    out = tag._compute_stats_parallel(
        df,
        "source_id",
        "path",
        compute_time=True,
        compute_cameras=True,
        compute_fields=True,
        file_ext="dat2",
        n_workers=1,
        checkpoint_path=checkpoint,
        chunk_size=1,
    )

    assert out.loc[out["source_id"] == "2001", "time_span_days"].item() == 123.0
    assert out.loc[out["source_id"] == "2002", "time_span_days"].item() == 20.0
    assert out.loc[out["source_id"] == "2002", "asassn_field_key"].item() == "field1"

    # The legacy checkpoint remains read-only; new progress is appended as a part.
    assert pd.read_parquet(checkpoint)["source_id"].astype(str).tolist() == ["2001"]
    parts_dir = checkpoint.with_name(f"{checkpoint.name}.parts")
    parts = sorted(parts_dir.glob("part-*.parquet"))
    assert len(parts) == 1
    assert pd.read_parquet(parts[0])["source_id"].astype(str).tolist() == ["2002"]

    out_resume = tag._compute_stats_parallel(
        df,
        "source_id",
        "path",
        compute_time=True,
        compute_cameras=True,
        compute_fields=True,
        file_ext="dat2",
        n_workers=1,
        checkpoint_path=checkpoint,
        chunk_size=1,
    )

    assert out_resume["time_span_days"].tolist() == [123.0, 20.0]
    assert sorted(parts_dir.glob("part-*.parquet")) == parts


def test_filter_camera_medians_marks_raw_suspects_without_exclusions(tmp_path: Path) -> None:
    lc_path = tmp_path / "C1.dat2"
    _write_mock_raw2(
        lc_path,
        [
            "# camera median 1siglow 1sighigh p90low p90high",
            "1 14.5 14.4 14.6 14.3 14.7",
            "bad line",
            "2 13.2 13.1 13.3 13.0 13.4",
        ],
    )
    df = pd.DataFrame(
        {
            "source_id": ["C1"],
            "path": [str(lc_path)],
            "mag_bin": ["13_13.5"],
        }
    )

    out = tag.filter_camera_medians(df, mag_tolerance=0.2, n_workers=1)

    assert out.loc[0, tag.RAW_MEDIAN_SUSPECT_COL] == "1"
    assert "excluded_cameras" not in out.columns


def test_filter_camera_medians_parallel_batches_match_sequential(tmp_path: Path) -> None:
    rows = [
        ("C1", "13_13.5", ["1 13.2 0 0 0 0", "2 14.0 0 0 0 0"]),
        ("C2", "12_12.5", ["1 12.1 0 0 0 0", "2 12.6 0 0 0 0"]),
        ("C3", "13.5_14", ["1 15.0 0 0 0 0", "2 13.8 0 0 0 0"]),
        ("C4", "not_a_bin", ["1 99.0 0 0 0 0"]),
    ]
    records = []
    for source_id, mag_bin, raw_lines in rows:
        lc_path = tmp_path / f"{source_id}.dat2"
        _write_mock_raw2(lc_path, raw_lines)
        records.append({"source_id": source_id, "path": str(lc_path), "mag_bin": mag_bin})
    df = pd.DataFrame.from_records(records)

    sequential = tag.filter_camera_medians(df, mag_tolerance=0.2, n_workers=1, chunk_size=2)
    parallel = tag.filter_camera_medians(df, mag_tolerance=0.2, n_workers=2, chunk_size=2)

    seq_values = sequential.sort_values("source_id")[tag.RAW_MEDIAN_SUSPECT_COL].tolist()
    par_values = parallel.sort_values("source_id")[tag.RAW_MEDIAN_SUSPECT_COL].tolist()
    assert par_values == seq_values
    assert seq_values == ["2", "", "1", ""]


def test_filter_camera_medians_reads_part_checkpoints_and_skips_completed(tmp_path: Path) -> None:
    c1_path = tmp_path / "C1.dat2"
    c2_path = tmp_path / "C2.dat2"
    _write_mock_raw2(c1_path, ["1 14.5 0 0 0 0"])
    _write_mock_raw2(c2_path, ["2 14.5 0 0 0 0"])
    df = pd.DataFrame(
        {
            "source_id": ["C1", "C2"],
            "path": [str(c1_path), str(c2_path)],
            "mag_bin": ["13_13.5", "13_13.5"],
        }
    )
    checkpoint = tmp_path / "camera.parquet"
    parts_dir = checkpoint.with_name(f"{checkpoint.name}.parts")
    parts_dir.mkdir()
    pd.DataFrame(
        {
            "source_id": ["C1"],
            tag.RAW_MEDIAN_SUSPECT_COL: ["7"],
        }
    ).to_parquet(parts_dir / "part-000.parquet", index=False)

    out = tag.filter_camera_medians(
        df,
        mag_tolerance=0.2,
        n_workers=1,
        checkpoint_path=checkpoint,
        chunk_size=1,
    )

    assert out[tag.RAW_MEDIAN_SUSPECT_COL].tolist() == ["7", "2"]
    parts = sorted(parts_dir.glob("part-*.parquet"))
    assert len(parts) == 2
    assert pd.read_parquet(parts[-1])["source_id"].astype(str).tolist() == ["C2"]

    out_resume = tag.filter_camera_medians(
        df,
        mag_tolerance=0.2,
        n_workers=1,
        checkpoint_path=checkpoint,
        chunk_size=1,
    )

    assert out_resume[tag.RAW_MEDIAN_SUSPECT_COL].tolist() == ["7", "2"]
    assert sorted(parts_dir.glob("part-*.parquet")) == parts


def test_filter_camera_medians_reads_legacy_checkpoint_and_writes_parts(tmp_path: Path) -> None:
    c1_path = tmp_path / "C1.dat2"
    c2_path = tmp_path / "C2.dat2"
    _write_mock_raw2(c1_path, ["1 14.5 0 0 0 0"])
    _write_mock_raw2(c2_path, ["2 14.5 0 0 0 0"])
    checkpoint = tmp_path / "camera.parquet"
    pd.DataFrame(
        {
            "source_id": ["C1"],
            tag.RAW_MEDIAN_SUSPECT_COL: ["9"],
        }
    ).to_parquet(checkpoint, index=False)
    df = pd.DataFrame(
        {
            "source_id": ["C1", "C2"],
            "path": [str(c1_path), str(c2_path)],
            "mag_bin": ["13_13.5", "13_13.5"],
        }
    )

    out = tag.filter_camera_medians(
        df,
        mag_tolerance=0.2,
        n_workers=1,
        checkpoint_path=checkpoint,
        chunk_size=1,
    )

    assert out[tag.RAW_MEDIAN_SUSPECT_COL].tolist() == ["9", "2"]
    assert pd.read_parquet(checkpoint)["source_id"].astype(str).tolist() == ["C1"]
    parts_dir = checkpoint.with_name(f"{checkpoint.name}.parts")
    parts = sorted(parts_dir.glob("part-*.parquet"))
    assert len(parts) == 1
    assert pd.read_parquet(parts[0])["source_id"].astype(str).tolist() == ["C2"]
