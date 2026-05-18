from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd

from malca import external_lcs
from malca.review.pipeline import _run_external_lcs_stage, detect_pipeline_status, run_missing_stages
from malca.review.store import db_connect, get_candidate_payload, import_candidates


def _install_fake_vetting(monkeypatch, calls: list[dict]) -> None:
    module = types.ModuleType("malca.vetting")

    def fake_fetch_external_lcs(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        calls.append(kwargs)
        out = df.copy()
        out["atlas_has_phot"] = False
        out["ztf_lc_n_det"] = 0
        out["gaia_epoch_lc_n_g"] = 0
        out["tess_n_sectors"] = 1
        out["tess_total_points"] = 25
        out["tess_flux_range"] = 0.02
        out["neowise_n_epochs"] = 3
        out["neowise_w1_range"] = 0.1
        out["neowise_w2_range"] = 0.2
        out["ps1_lc_n_points"] = 0
        out["crts_lc_n_points"] = 0
        return out

    module.fetch_external_lcs = fake_fetch_external_lcs
    monkeypatch.setitem(sys.modules, "malca.vetting", module)


def test_external_lcs_cli_runs_tess_by_default(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    _install_fake_vetting(monkeypatch, calls)

    monkeypatch.setattr(
        external_lcs,
        "read_parquet_table",
        lambda _path: pd.DataFrame([{"asas_sn_id": "C1", "ra": 1.0, "dec": 2.0}]),
    )
    written: dict[str, object] = {}
    monkeypatch.setattr(
        external_lcs,
        "write_parquet_table",
        lambda df, path: written.update({"df": df.copy(), "path": path}),
    )

    args = external_lcs.build_arg_parser().parse_args(
        [str(tmp_path / "candidates.parquet"), "--output-dir", str(tmp_path), "--no-checkpoint"]
    )
    external_lcs.run(args)

    assert calls[-1]["run_tess"] is True
    assert calls[-1]["run_neowise"] is True
    assert calls[-1]["run_atlas"] is False
    assert "tess_n_sectors" in written["df"].columns
    assert "neowise_n_epochs" in written["df"].columns


def test_external_lcs_cli_can_skip_tess(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    _install_fake_vetting(monkeypatch, calls)
    monkeypatch.setattr(
        external_lcs,
        "read_parquet_table",
        lambda _path: pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}]),
    )
    monkeypatch.setattr(external_lcs, "write_parquet_table", lambda _df, _path: None)

    args = external_lcs.build_arg_parser().parse_args(
        [
            str(tmp_path / "candidates.parquet"),
            "--output-dir",
            str(tmp_path),
            "--no-checkpoint",
            "--no-tess",
        ]
    )
    external_lcs.run(args)

    assert calls[-1]["run_tess"] is False


def test_external_lcs_cli_can_skip_neowise(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    _install_fake_vetting(monkeypatch, calls)
    monkeypatch.setattr(
        external_lcs,
        "read_parquet_table",
        lambda _path: pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}]),
    )
    monkeypatch.setattr(external_lcs, "write_parquet_table", lambda _df, _path: None)

    args = external_lcs.build_arg_parser().parse_args(
        [
            str(tmp_path / "candidates.parquet"),
            "--output-dir",
            str(tmp_path),
            "--no-checkpoint",
            "--no-neowise",
        ]
    )
    external_lcs.run(args)

    assert calls[-1]["run_neowise"] is False


def test_review_external_lcs_stage_runs_tess(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    _install_fake_vetting(monkeypatch, calls)
    payload = {"candidate_id": "C1", "ra_deg": 1.0, "dec_deg": 2.0}

    _run_external_lcs_stage(payload, tmp_path)

    assert calls[-1]["run_tess"] is True
    assert calls[-1]["run_neowise"] is True
    assert calls[-1]["run_atlas"] is False
    assert payload["tess_n_sectors"] == 1
    assert payload["tess_total_points"] == 25
    assert payload["neowise_n_epochs"] == 3
    assert calls[-1]["refresh_cache"] is False


def test_review_external_lcs_stage_can_refresh_cache(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    _install_fake_vetting(monkeypatch, calls)
    payload = {"candidate_id": "C1", "ra_deg": 1.0, "dec_deg": 2.0}

    _run_external_lcs_stage(payload, tmp_path, refresh_cache=True)

    assert calls[-1]["refresh_cache"] is True


def test_review_forced_external_lcs_stage_accepts_ra_dec_aliases(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    _install_fake_vetting(monkeypatch, calls)
    run_dir = tmp_path / "run"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)
    source_path = results_dir / "candidates.parquet"
    source_path.write_text("", encoding="ascii")
    db_path = tmp_path / "review.db"
    log_lines: list[str] = []

    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}]),
            source_path=str(source_path),
            characterize_before_import=False,
            vet_before_import=False,
        )
        stages = run_missing_stages(
            conn,
            "C1",
            progress_callback=log_lines.append,
            force_stages=["external_lcs"],
            only_force=True,
        )
        payload = get_candidate_payload(conn, "C1")

    assert stages == ["external_lcs"]
    assert calls[-1]["refresh_cache"] is True
    assert calls[-1]["output_dir"] == results_dir
    assert payload["ra_deg"] == 1.0
    assert payload["dec_deg"] == 2.0
    assert payload["tess_n_sectors"] == 1
    assert any("Fetching external LCs" in line for line in log_lines)


def test_review_external_lcs_failure_is_not_marked_complete(monkeypatch, tmp_path: Path) -> None:
    module = types.ModuleType("malca.vetting")

    def fake_fetch_external_lcs(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = df.copy()
        out["tess_n_sectors"] = 1
        out.attrs["external_lc_failures"] = ["ZTF LCs failed: test failure"]
        return out

    module.fetch_external_lcs = fake_fetch_external_lcs
    monkeypatch.setitem(sys.modules, "malca.vetting", module)
    results_dir = tmp_path / "run" / "results"
    results_dir.mkdir(parents=True)
    source_path = results_dir / "candidates.parquet"
    source_path.write_text("", encoding="ascii")
    db_path = tmp_path / "review.db"
    log_lines: list[str] = []
    completed: list[str] = []

    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame([{"candidate_id": "C1", "ra_deg": 1.0, "dec_deg": 2.0}]),
            source_path=str(source_path),
            characterize_before_import=False,
            vet_before_import=False,
        )
        stages = run_missing_stages(
            conn,
            "C1",
            progress_callback=log_lines.append,
            stage_complete_callback=completed.append,
            force_stages=["external_lcs"],
            only_force=True,
        )
        payload = get_candidate_payload(conn, "C1")

    assert stages == ["external_lcs"]
    assert completed == []
    assert payload["tess_n_sectors"] == 1
    assert any("External LCs finished with failures" in line for line in log_lines)


def test_external_lcs_status_requires_tess_signature() -> None:
    payload = {
        "ztf_lc_n_det": 0,
        "gaia_epoch_lc_n_g": 0,
        "neowise_n_epochs": 0,
        "ps1_lc_n_points": 0,
        "crts_lc_n_points": 0,
    }

    assert detect_pipeline_status(payload)["external_lcs"] == "partial"
    payload["tess_n_sectors"] = 0
    assert detect_pipeline_status(payload)["external_lcs"] == "complete"
