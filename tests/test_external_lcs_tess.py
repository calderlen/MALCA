from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd

from malca.enrichment import external_lcs, vetting
from malca.external_lc_manifest import (
    clear_external_lc_manifest_caches,
    read_external_lc_manifest,
    upsert_external_lc_manifest_entry,
)
from malca.review.pipeline import _run_external_lcs_stage, detect_pipeline_status, run_missing_stages
from malca.review.store import db_connect, get_candidate_payload, import_candidates


def _install_fake_vetting(monkeypatch, calls: list[dict]) -> None:
    module = types.ModuleType("malca.enrichment.vetting")

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
        out["kepler_n_quarters"] = 0
        out["kepler_total_points"] = 0
        out["kepler_flux_range"] = 0.0
        out["aavso_lc_n_points"] = 0
        out["ogle_lc_n_points"] = 0
        out["ogle_lc_i_range"] = 0.0
        out["ogle_lc_v_range"] = 0.0
        out["stripe82_lc_n_points"] = 0
        out["stripe82_lc_u_range"] = 0.0
        out["stripe82_lc_g_range"] = 0.0
        out["stripe82_lc_r_range"] = 0.0
        out["stripe82_lc_i_range"] = 0.0
        out["stripe82_lc_z_range"] = 0.0
        out["allwise_mep_n_epochs"] = 0
        out["allwise_mep_w1_range"] = 0.0
        out["allwise_mep_w2_range"] = 0.0
        out["allwise_mep_w3_range"] = 0.0
        out["allwise_mep_w4_range"] = 0.0
        out["vvvx_virac_n_epochs"] = 0
        out["vvvx_virac_z_range"] = 0.0
        out["vvvx_virac_y_range"] = 0.0
        out["vvvx_virac_j_range"] = 0.0
        out["vvvx_virac_h_range"] = 0.0
        out["vvvx_virac_ks_range"] = 0.0
        out["ps1_lc_n_points"] = 0
        out["crts_lc_n_points"] = 0
        return out

    module.fetch_external_lcs = fake_fetch_external_lcs
    monkeypatch.setitem(sys.modules, "malca.enrichment.vetting", module)


def test_external_lcs_cli_runs_tess_by_default(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    _install_fake_vetting(monkeypatch, calls)

    monkeypatch.setattr(
        external_lcs,
        "read_feature_table",
        lambda _path: pd.DataFrame([{"asas_sn_id": "C1", "ra": 1.0, "dec": 2.0}]),
    )
    written: dict[str, object] = {}
    monkeypatch.setattr(
        external_lcs,
        "write_feature_table",
        lambda df, path: written.update({"df": df.copy(), "path": path}),
    )

    args = external_lcs.build_arg_parser().parse_args(
        [str(tmp_path / "candidates.parquet"), "--output-dir", str(tmp_path), "--no-checkpoint"]
    )
    external_lcs.run(args)

    assert calls[-1]["run_tess"] is True
    assert calls[-1]["run_neowise"] is True
    assert calls[-1]["run_kepler"] is True
    assert calls[-1]["run_aavso"] is True
    assert calls[-1]["run_ogle"] is True
    assert calls[-1]["run_stripe82"] is True
    assert calls[-1]["run_allwise_mep"] is True
    assert calls[-1]["run_vvvx_virac"] is True
    assert calls[-1]["run_atlas"] is False
    assert "tess_n_sectors" in written["df"].columns
    assert "neowise_n_epochs" in written["df"].columns


def test_external_lcs_cli_hydrates_coordinates_from_layer_first_input(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    seen: dict[str, pd.DataFrame] = {}

    module = types.ModuleType("malca.enrichment.vetting")

    def fake_fetch_external_lcs(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        calls.append(kwargs)
        seen["df"] = df.copy()
        out = df.copy()
        out["neowise_n_epochs"] = 0
        return out

    module.fetch_external_lcs = fake_fetch_external_lcs
    monkeypatch.setitem(sys.modules, "malca.enrichment.vetting", module)

    input_path = tmp_path / "candidates.parquet"
    external_lcs.write_feature_table(
        pd.DataFrame(
            [
                {
                    "candidate_id": "C1",
                    "asas_sn_id": "C1",
                    "ra": 1.25,
                    "dec": -2.5,
                    "gaia_epoch_available": True,
                    "gaia_epoch_n_obs": 7,
                    "gaia_epoch_g_range": 0.42,
                    "failed_any": 0,
                }
            ]
        ),
        input_path,
    )

    args = external_lcs.build_arg_parser().parse_args(
        [
            str(input_path),
            "--output",
            str(tmp_path / "external.parquet"),
            "--output-dir",
            str(tmp_path / "external_lcs"),
            "--no-checkpoint",
            "--all-candidates",
        ]
    )
    external_lcs.run(args)

    assert calls
    assert float(seen["df"].loc[0, "ra"]) == 1.25
    assert float(seen["df"].loc[0, "dec"]) == -2.5
    assert bool(seen["df"].loc[0, "gaia_epoch_available"]) is True
    assert int(seen["df"].loc[0, "gaia_epoch_n_obs"]) == 7
    assert float(seen["df"].loc[0, "gaia_epoch_g_range"]) == 0.42


def test_external_lcs_cli_accepts_review_db_input_without_implicit_merge(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    seen: dict[str, pd.DataFrame] = {}
    module = types.ModuleType("malca.enrichment.vetting")

    def fake_fetch_external_lcs(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        calls.append(kwargs)
        seen["df"] = df.copy()
        out = df.copy()
        out["tess_n_sectors"] = 1
        out["tess_total_points"] = 25
        out["tess_flux_range"] = 0.02
        return out

    module.fetch_external_lcs = fake_fetch_external_lcs
    monkeypatch.setitem(sys.modules, "malca.enrichment.vetting", module)

    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame([{"candidate_id": "C1", "asas_sn_id": "A1"}]),
            source_path="candidates.parquet",
            characterize_before_import=False,
            vet_before_import=False,
        )
        conn.execute(
            "UPDATE candidates SET ra=NULL, dec=NULL, payload_json=? WHERE candidate_id='C1'",
            (
                json.dumps(
                    {
                        "candidate_id": "C1",
                        "asas_sn_id": "A1",
                        "payload_json": json.dumps(
                            {
                                "candidate_id": "C1",
                                "asas_sn_id": "A1",
                                "ra_deg": 1.25,
                                "dec_deg": -2.5,
                            }
                        ),
                    }
                ),
            ),
        )
        conn.commit()

    args = external_lcs.build_arg_parser().parse_args(
        [
            str(db_path),
            "--output",
            str(tmp_path / "external.parquet"),
            "--output-dir",
            str(tmp_path / "external_lcs"),
            "--no-checkpoint",
            "--all-candidates",
        ]
    )
    external_lcs.run(args)

    assert calls
    assert float(seen["df"].loc[0, "ra"]) == 1.25
    assert float(seen["df"].loc[0, "dec"]) == -2.5
    assert (tmp_path / "external.parquet").exists()
    with db_connect(db_path) as conn:
        payload = get_candidate_payload(conn, "C1")
    assert "tess_n_sectors" not in payload
    assert "tess_total_points" not in payload


def test_external_lcs_cli_explicit_review_db_merges_results(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    module = types.ModuleType("malca.enrichment.vetting")

    def fake_fetch_external_lcs(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        calls.append(kwargs)
        out = df.copy()
        out["tess_n_sectors"] = 1
        out["tess_total_points"] = 25
        out["tess_flux_range"] = 0.02
        return out

    module.fetch_external_lcs = fake_fetch_external_lcs
    monkeypatch.setitem(sys.modules, "malca.enrichment.vetting", module)

    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame([{"candidate_id": "C1", "asas_sn_id": "A1", "ra": 1.25, "dec": -2.5}]),
            source_path="candidates.parquet",
            characterize_before_import=False,
            vet_before_import=False,
        )

    args = external_lcs.build_arg_parser().parse_args(
        [
            str(db_path),
            "--output",
            str(tmp_path / "external.parquet"),
            "--output-dir",
            str(tmp_path / "external_lcs"),
            "--review-db",
            str(db_path),
            "--no-checkpoint",
            "--all-candidates",
        ]
    )
    external_lcs.run(args)

    assert calls
    with db_connect(db_path) as conn:
        payload = get_candidate_payload(conn, "C1")
    assert payload["tess_n_sectors"] == 1
    assert payload["tess_total_points"] == 25


def test_external_lcs_cli_can_skip_tess(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    _install_fake_vetting(monkeypatch, calls)
    monkeypatch.setattr(
        external_lcs,
        "read_feature_table",
        lambda _path: pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}]),
    )
    monkeypatch.setattr(external_lcs, "write_feature_table", lambda _df, _path: None)

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
        "read_feature_table",
        lambda _path: pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}]),
    )
    monkeypatch.setattr(external_lcs, "write_feature_table", lambda _df, _path: None)

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


def test_external_lcs_cli_new_default_sources_can_be_skipped(monkeypatch, tmp_path: Path) -> None:
    flag_to_kw = {
        "--no-kepler": "run_kepler",
        "--no-aavso": "run_aavso",
        "--no-ogle": "run_ogle",
        "--no-stripe82": "run_stripe82",
        "--no-allwise-mep": "run_allwise_mep",
        "--no-vvvx-virac": "run_vvvx_virac",
    }

    for flag, kw in flag_to_kw.items():
        calls: list[dict] = []
        _install_fake_vetting(monkeypatch, calls)
        monkeypatch.setattr(
            external_lcs,
            "read_feature_table",
            lambda _path: pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}]),
        )
        monkeypatch.setattr(external_lcs, "write_feature_table", lambda _df, _path: None)

        args = external_lcs.build_arg_parser().parse_args(
            [
                str(tmp_path / f"{kw}.parquet"),
                "--output-dir",
                str(tmp_path),
                "--no-checkpoint",
                flag,
            ]
        )
        external_lcs.run(args)

        assert calls[-1][kw] is False
        for other_kw in flag_to_kw.values():
            if other_kw != kw:
                assert calls[-1][other_kw] is True


def test_external_lcs_cache_only_rebuilds_summary_from_lc_file(monkeypatch, tmp_path: Path) -> None:
    module = types.ModuleType("malca.enrichment.vetting")

    def fake_fetch_external_lcs(*_args, **_kwargs):
        raise AssertionError("cache-only mode should not call remote fetchers")

    def fake_read_status(_output_dir: Path) -> pd.DataFrame:
        return pd.DataFrame()

    def fake_candidate_cache_id(df: pd.DataFrame, idx: object) -> str:
        return str(df.loc[idx, "candidate_id"])

    def fake_external_lc_path(output_dir: Path, prefix: str, df: pd.DataFrame, idx: object) -> Path:
        return Path(output_dir) / f"{prefix}_{fake_candidate_cache_id(df, idx)}.parquet"

    def fake_read_external_lc_file(path: Path) -> pd.DataFrame | None:
        return pd.read_parquet(path) if path.exists() else None

    def fake_flux_summary(
        lc: pd.DataFrame,
        group_col: str,
        n_col: str,
        points_col: str,
        range_col: str,
    ) -> dict[str, float]:
        flux = pd.to_numeric(lc["flux"], errors="coerce").dropna()
        return {
            n_col: int(lc[group_col].nunique()),
            points_col: int(len(lc)),
            range_col: float(flux.max() - flux.min()),
        }

    def fake_count_summary(lc: pd.DataFrame, col: str) -> dict[str, int]:
        return {col: int(len(lc))}

    module.fetch_external_lcs = fake_fetch_external_lcs
    module._read_external_lc_status = fake_read_status
    module._candidate_cache_id = fake_candidate_cache_id
    module._external_lc_path = fake_external_lc_path
    module._read_external_lc_file = fake_read_external_lc_file
    module._summarize_flux_lc = fake_flux_summary
    module._summarize_count_lc = fake_count_summary
    module._summarize_atlas_lc = lambda _lc: {}
    module._summarize_ztf_lc = lambda _lc: {}
    module._summarize_gaia_epoch_lc = lambda _lc: {}
    module._summarize_neowise_lc = lambda _lc: {}
    module._summarize_ogle_lc = lambda _lc: {}
    module._summarize_stripe82_lc = lambda _lc: {}
    module._summarize_allwise_mep_lc = lambda _lc: {}
    module._summarize_vvvx_virac_lc = lambda _lc: {}
    monkeypatch.setitem(sys.modules, "malca.enrichment.vetting", module)

    output_dir = tmp_path / "external_lcs"
    output_dir.mkdir()
    pd.DataFrame({"quarter": [1, 1, 2], "flux": [1.0, 1.2, 0.8]}).to_parquet(
        output_dir / "kepler_lc_C1.parquet"
    )
    monkeypatch.setattr(
        external_lcs,
        "read_feature_table",
        lambda _path: pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}]),
    )

    output_path = tmp_path / "external.parquet"
    args = external_lcs.build_arg_parser().parse_args(
        [
            str(tmp_path / "candidates.parquet"),
            "--cache-only",
            "--output",
            str(output_path),
            "--output-dir",
            str(output_dir),
            "--no-checkpoint",
            "--no-atlas",
            "--no-ztf",
            "--no-gaia-epoch",
            "--no-tess",
            "--no-neowise",
            "--no-aavso",
            "--no-ogle",
            "--no-stripe82",
            "--no-allwise-mep",
            "--no-vvvx-virac",
            "--no-ps1",
            "--no-crts",
        ]
    )

    external_lcs.run(args)

    saved = pd.read_parquet(output_path)
    external_stats = json.loads(saved.loc[0, "external_stats"])
    assert external_stats["kepler_n_quarters"] == 2
    assert external_stats["kepler_total_points"] == 3
    assert external_stats["kepler_flux_range"] == 0.3999999999999999


def test_external_lcs_cache_only_uses_manifest_for_nested_ps1_file(tmp_path: Path) -> None:
    root = tmp_path / "results"
    lc_path = root / "external_lcs_staging" / "ps1" / "ps1_lc_C1.parquet"
    lc_path.parent.mkdir(parents=True)
    pd.DataFrame({"mjd": [59000.0, 59001.0, 59002.0], "mag": [15.0, 15.2, 15.1]}).to_parquet(lc_path, index=False)
    assert upsert_external_lc_manifest_entry(root, candidate_id="C1", source="ps1", file_prefix="ps1", path=lc_path)
    clear_external_lc_manifest_caches()

    df = pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}])
    out = external_lcs.rebuild_external_lc_table_from_cache(df, root, {"ps1": True})

    assert int(out.loc[0, "ps1_lc_n_points"]) == 3
    frames = external_lcs._cache_only_source_merge_frames(out, {"ps1": True})
    assert len(frames) == 1
    assert frames[0][0] == "ps1"
    assert frames[0][1].to_dict("records") == [{"candidate_id": "C1", "ps1_lc_n_points": 3}]


def test_external_lcs_cache_only_scans_nested_files_when_manifest_missing(tmp_path: Path) -> None:
    root = tmp_path / "results"
    lc_path = root / "external_lcs_staging" / "ps1" / "ps1_lc_C1.parquet"
    lc_path.parent.mkdir(parents=True)
    pd.DataFrame({"mjd": [59000.0, 59001.0], "mag": [15.0, 15.2]}).to_parquet(lc_path, index=False)
    clear_external_lc_manifest_caches()

    df = pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}])
    out = external_lcs.rebuild_external_lc_table_from_cache(df, root, {"ps1": True})

    assert int(out.loc[0, "ps1_lc_n_points"]) == 2
    manifest = read_external_lc_manifest(root)
    assert manifest[["candidate_id", "file_prefix", "path_relative"]].to_dict("records") == [
        {
            "candidate_id": "C1",
            "file_prefix": "ps1",
            "path_relative": "external_lcs_staging/ps1/ps1_lc_C1.parquet",
        }
    ]


def test_external_lcs_cache_only_reads_staged_status_without_touching_unattempted_rows(tmp_path: Path) -> None:
    root = tmp_path / "results"
    status_path = root / "external_lcs_staging" / "ps1" / vetting.EXTERNAL_LC_STATUS_FILE
    status_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "module": "Pan-STARRS LCs",
                "candidate_id": "C1",
                "cache_key": "synthetic",
                "status": "no_data",
                "updated_unix": 1.0,
                "ps1_lc_n_points": 0,
            }
        ]
    ).to_parquet(status_path, index=False)

    df = pd.DataFrame(
        [
            {"candidate_id": "C1", "ra": 1.0, "dec": 2.0},
            {"candidate_id": "C2", "ra": 3.0, "dec": 4.0},
        ]
    )
    out = external_lcs.rebuild_external_lc_table_from_cache(df, root, {"ps1": True})

    assert int(out.loc[0, "ps1_lc_n_points"]) == 0
    frames = external_lcs._cache_only_source_merge_frames(out, {"ps1": True})
    assert frames[0][1].to_dict("records") == [{"candidate_id": "C1", "ps1_lc_n_points": 0}]


def test_external_lcs_cache_only_source_merge_preserves_unrelated_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "review.db"
    with db_connect(db_path) as conn:
        import_candidates(
            conn,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "C1",
                        "asas_sn_id": "C1",
                        "neowise_n_epochs": 9,
                        "ps1_lc_n_points": 0,
                    }
                ]
            ),
            source_path=str(tmp_path),
            characterize_before_import=False,
            vet_before_import=False,
        )

    updates = external_lcs._merge_source_frames_into_review_db_with_retries(
        db_path,
        [("ps1", pd.DataFrame([{"candidate_id": "C1", "ps1_lc_n_points": 5}]))],
    )

    assert updates == {"ps1": 1}
    with db_connect(db_path) as conn:
        payload = get_candidate_payload(conn, "C1")
    assert payload["ps1_lc_n_points"] == 5
    assert payload["neowise_n_epochs"] == 9


def test_external_lcs_cache_only_repairs_ztf_status_from_existing_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "external_lcs"
    output_dir.mkdir()
    df = pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}])
    pd.DataFrame(
        {
            "mjd": [59000.0, 59001.0, 59002.0],
            "mag": [15.0, 15.4, 15.1],
            "band": ["zg", "zg", "zr"],
        }
    ).to_parquet(output_dir / "ztf_lc_C1.parquet", index=False)
    cache_key = vetting._coord_lookup_cache_key(df, 0, 2.0, vetting.ZTF_LC_COLLECTION)
    pd.DataFrame(
        [
            {
                "module": "ZTF LCs",
                "candidate_id": "C1",
                "cache_key": cache_key,
                "status": "failed",
                "ztf_lc_n_det": 0,
            }
        ]
    ).to_parquet(output_dir / vetting.EXTERNAL_LC_STATUS_FILE, index=False)

    out = external_lcs.rebuild_external_lc_table_from_cache(df, output_dir, {"ztf": True})

    assert int(out.loc[0, "ztf_lc_n_det"]) == 3
    assert round(float(out.loc[0, "ztf_lc_g_range"]), 6) == 0.4
    status = pd.read_parquet(output_dir / vetting.EXTERNAL_LC_STATUS_FILE)
    row = status[status["candidate_id"] == "C1"].iloc[-1]
    assert row["status"] == "fetched"
    assert int(row["ztf_lc_n_det"]) == 3


def test_review_external_lcs_stage_runs_tess(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    _install_fake_vetting(monkeypatch, calls)
    payload = {"candidate_id": "C1", "ra": 1.0, "dec": 2.0}

    _run_external_lcs_stage(payload, tmp_path)

    assert calls[-1]["run_tess"] is True
    assert calls[-1]["run_neowise"] is True
    assert calls[-1]["run_kepler"] is True
    assert calls[-1]["run_aavso"] is True
    assert calls[-1]["run_ogle"] is True
    assert calls[-1]["run_stripe82"] is True
    assert calls[-1]["run_allwise_mep"] is True
    assert calls[-1]["run_vvvx_virac"] is True
    assert calls[-1]["run_atlas"] is False
    assert payload["tess_n_sectors"] == 1
    assert payload["tess_total_points"] == 25
    assert payload["neowise_n_epochs"] == 3
    assert calls[-1]["refresh_cache"] is False


def test_review_external_lcs_stage_can_refresh_cache(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    _install_fake_vetting(monkeypatch, calls)
    payload = {"candidate_id": "C1", "ra": 1.0, "dec": 2.0}

    _run_external_lcs_stage(payload, tmp_path, refresh_cache=True)

    assert calls[-1]["refresh_cache"] is True


def test_review_forced_external_lcs_stage_uses_canonical_coordinates(monkeypatch, tmp_path: Path) -> None:
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
    assert payload["ra"] == 1.0
    assert payload["dec"] == 2.0
    assert payload["tess_n_sectors"] == 1
    assert any("Fetching external LCs" in line for line in log_lines)


def test_review_external_lcs_failure_is_not_marked_complete(monkeypatch, tmp_path: Path) -> None:
    module = types.ModuleType("malca.enrichment.vetting")

    def fake_fetch_external_lcs(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = df.copy()
        out["tess_n_sectors"] = 1
        out.attrs["external_lc_failures"] = ["ZTF LCs failed: test failure"]
        return out

    module.fetch_external_lcs = fake_fetch_external_lcs
    monkeypatch.setitem(sys.modules, "malca.enrichment.vetting", module)
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
            pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}]),
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
        "kepler_n_quarters": 0,
        "aavso_lc_n_points": 0,
        "ogle_lc_n_points": 0,
        "stripe82_lc_n_points": 0,
        "allwise_mep_n_epochs": 0,
        "vvvx_virac_n_epochs": 0,
        "ps1_lc_n_points": 0,
        "crts_lc_n_points": 0,
    }

    assert detect_pipeline_status(payload)["external_lcs"] == "partial"
    payload["tess_n_sectors"] = 0
    assert detect_pipeline_status(payload)["external_lcs"] == "complete"
