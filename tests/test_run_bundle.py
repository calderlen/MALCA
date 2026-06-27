from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd
import pytest

from malca.products.run_bundle import collect_candidate_lightcurve_files, export_run_bundle, import_bundle_zip


def test_import_bundle_zip_rejects_missing_and_invalid_zip(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        import_bundle_zip(tmp_path / "missing.zip", tmp_path / "run")

    invalid = tmp_path / "invalid.zip"
    invalid.write_text("not a zip", encoding="ascii")
    with pytest.raises(ValueError):
        import_bundle_zip(invalid, tmp_path / "run")


def test_import_bundle_zip_honors_overwrite(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("results/new.txt", "new\n")

    run_dir = tmp_path / "run"
    stale = run_dir / "stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale\n", encoding="ascii")

    import_bundle_zip(bundle, run_dir, overwrite=True)

    assert not stale.exists()
    assert (run_dir / "results" / "new.txt").read_text(encoding="ascii") == "new\n"


def test_export_run_bundle_includes_run_and_external_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "results").mkdir(parents=True)
    (run_dir / "run_params.json").write_text("{}", encoding="ascii")
    (run_dir / "results" / "table.parquet").write_text("table\n", encoding="ascii")
    external = tmp_path / "lc.dat2"
    external.write_text("lc\n", encoding="ascii")

    bundle = tmp_path / "bundle.zip"
    names = export_run_bundle(
        bundle,
        run_dir,
        include_files=["run_params.json"],
        include_dirs=["results"],
        external_files=[(external, "bundle_assets/lightcurves/lc.dat2")],
    )

    assert names == [
        "results/table.parquet",
        "run_params.json",
        "bundle_assets/lightcurves/lc.dat2",
    ]


def test_collect_candidate_lightcurve_files_uses_first_available_path_column(tmp_path: Path) -> None:
    dat = tmp_path / "lc" / "candidate.dat2"
    dat.parent.mkdir()
    dat.write_text("lc\n", encoding="ascii")

    collection = collect_candidate_lightcurve_files(
        pd.DataFrame({"dat_path": [str(dat)]}),
        path_cols=("path", "dat_path"),
        arc_prefix="bundle_assets/lightcurves",
        allowed_suffix_prefixes=("dat",),
    )

    assert collection.rows == 1
    assert collection.candidate_paths == 1
    assert collection.added == 1
    assert collection.files == [(dat, "bundle_assets/lightcurves/candidate.dat2")]


def test_collect_candidate_lightcurve_files_dedupes_paths_and_colliding_basenames(tmp_path: Path) -> None:
    first = tmp_path / "a" / "same.dat2"
    second = tmp_path / "b" / "same.dat2"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first\n", encoding="ascii")
    second.write_text("second\n", encoding="ascii")

    collection = collect_candidate_lightcurve_files(
        pd.DataFrame({"path": [str(first), str(first), str(second)]}),
        path_cols=("path",),
        arc_prefix="bundle_assets/lightcurves",
    )

    assert collection.added == 2
    assert collection.duplicate_arcname == 1
    assert collection.files == [
        (first, "bundle_assets/lightcurves/same.dat2"),
        (second, "bundle_assets/lightcurves/000001_same.dat2"),
    ]


def test_collect_candidate_lightcurve_files_includes_sidecars_and_skips_suffixes(tmp_path: Path) -> None:
    dat = tmp_path / "candidate.dat3"
    raw = tmp_path / "candidate.raw2"
    txt = tmp_path / "notes.txt"
    dat.write_text("dat\n", encoding="ascii")
    raw.write_text("raw\n", encoding="ascii")
    txt.write_text("txt\n", encoding="ascii")

    collection = collect_candidate_lightcurve_files(
        pd.DataFrame({"path": [str(dat), str(txt)]}),
        path_cols=("path",),
        arc_prefix="bundle_assets/lightcurves",
        allowed_suffix_prefixes=("dat",),
        sidecar_suffixes=(".raw2",),
    )

    assert collection.skipped_suffix == 1
    assert collection.files == [
        (dat, "bundle_assets/lightcurves/candidate.dat3"),
        (raw, "bundle_assets/lightcurves/candidate.raw2"),
    ]


def test_collect_candidate_lightcurve_files_counts_missing_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.dat2"

    collection = collect_candidate_lightcurve_files(
        pd.DataFrame({"path": [str(missing)]}),
        path_cols=("path",),
        arc_prefix="bundle_assets/lightcurves",
        allowed_suffix_prefixes=("dat",),
    )

    assert collection.missing == 1
    assert collection.added == 0
    assert collection.files == []
