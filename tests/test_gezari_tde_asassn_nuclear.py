from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_gezari2021_tde_asassn_nuclear.py"
SPEC = importlib.util.spec_from_file_location("run_gezari2021_tde_asassn_nuclear", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = script
SPEC.loader.exec_module(script)


def _tde_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "gezari2021_test_tde",
                "name": "Test TDE",
                "survey": "ZTF",
                "waveband": "O",
                "redshift": 0.1,
                "log_lbb_erg_s": 44.0,
                "log_tbb_k": 4.5,
                "paper_reference": "Example et al. 2021",
                "ra_deg": 10.0,
                "dec_deg": 20.0,
            }
        ]
    )


def _index_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _write_dat3(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "2459000.0 17.00 0.03 1 1 0 0 ba/F1",
                "2459001.0 16.80 0.03 1 1 0 0 ba/F1",
                "2459002.0 16.95 0.04 1 2 1 0 bb/F1",
                "2459003.0 17.10 0.04 1 2 1 0 bb/F1",
            ]
        )
        + "\n",
        encoding="ascii",
    )


def test_strict_crossmatch_requires_exactly_one_match() -> None:
    out = script.strict_crossmatch_asassn(
        _tde_frame(),
        _index_frame([{"asas_sn_id": "1001", "ra_deg": 10.0001, "dec_deg": 20.0}]),
        radius_arcsec=5.0,
    )

    assert len(out) == 1
    assert out.loc[0, "candidate_id"] == "gezari2021_test_tde"
    assert out.loc[0, "asas_sn_id"] == "1001"
    assert 0.0 < out.loc[0, "asassn_sep_arcsec"] < 5.0


def test_strict_crossmatch_fails_on_zero_matches() -> None:
    with pytest.raises(script.StrictPreflightError) as exc_info:
        script.strict_crossmatch_asassn(
            _tde_frame(),
            _index_frame([{"asas_sn_id": "1001", "ra_deg": 11.0, "dec_deg": 20.0}]),
            radius_arcsec=5.0,
        )

    diagnostics = exc_info.value.diagnostics
    assert diagnostics.loc[0, "reason"] == "unmatched"
    assert diagnostics.loc[0, "match_count"] == 0


def test_strict_crossmatch_fails_on_multiple_matches() -> None:
    with pytest.raises(script.StrictPreflightError) as exc_info:
        script.strict_crossmatch_asassn(
            _tde_frame(),
            _index_frame(
                [
                    {"asas_sn_id": "1001", "ra_deg": 10.0001, "dec_deg": 20.0},
                    {"asas_sn_id": "1002", "ra_deg": 10.0002, "dec_deg": 20.0},
                ]
            ),
            radius_arcsec=5.0,
        )

    diagnostics = exc_info.value.diagnostics
    assert diagnostics.loc[0, "reason"] == "ambiguous"
    assert diagnostics.loc[0, "match_count"] == 2
    assert diagnostics.loc[0, "matched_asas_sn_ids"] == "1001;1002"


def test_attach_lightcurve_manifest_fails_when_match_absent() -> None:
    matches = _tde_frame().assign(asas_sn_id="1001", asassn_sep_arcsec=0.2)
    manifest = pd.DataFrame([{"source_id": "9999", "dat_path": "/tmp/9999.dat3", "dat_exists": True}])

    with pytest.raises(script.StrictPreflightError) as exc_info:
        script.attach_lightcurve_manifest(matches, manifest)

    assert exc_info.value.diagnostics.loc[0, "reason"] == "missing_manifest_row"


def test_attach_lightcurve_manifest_fails_when_dat_missing(tmp_path: Path) -> None:
    lc_path = tmp_path / "1001.dat3"
    _write_dat3(lc_path)
    matches = _tde_frame().assign(asas_sn_id="1001", asassn_sep_arcsec=0.2)
    manifest = pd.DataFrame([{"source_id": "1001", "dat_path": str(lc_path), "dat_exists": False}])

    with pytest.raises(script.StrictPreflightError) as exc_info:
        script.attach_lightcurve_manifest(matches, manifest)

    assert exc_info.value.diagnostics.loc[0, "reason"] == "dat_exists_not_true"


def test_tiny_end_to_end_recovery_writes_reports_and_plots(monkeypatch, tmp_path: Path) -> None:
    tde_csv = tmp_path / "gezari.csv"
    index_path = tmp_path / "asassn_index.parquet"
    manifest_path = tmp_path / "lc_manifest.parquet"
    run_dir = tmp_path / "run"
    lc_path = tmp_path / "1001.dat3"
    _write_dat3(lc_path)

    _tde_frame().drop(columns=["candidate_id"]).to_csv(tde_csv, index=False)
    _index_frame([{"asas_sn_id": "1001", "ra_deg": 10.0001, "dec_deg": 20.0, "gaia_id": "123"}]).to_parquet(index_path)
    pd.DataFrame([{"source_id": "1001", "dat_path": str(lc_path), "dat_exists": True}]).to_parquet(manifest_path)

    def fake_run_nuclear_context(targets: pd.DataFrame, config: object) -> pd.DataFrame:
        out = targets.copy()
        out["lc_feature_status"] = "ok"
        out["nuc_n_points"] = 4
        out["nuc_time_span_days"] = 3.0
        out["agn_prior_score"] = 0.05
        out["tde_candidate_score"] = 0.82
        out["clagn_photometric_score"] = 0.15
        out["tde_candidate_reasons"] = "single flare; quiet baseline"
        out["agn_prior_reasons"] = ""
        out["clagn_reasons"] = ""
        return out

    monkeypatch.setattr(script, "run_nuclear_context", fake_run_nuclear_context)

    out = script.run_recovery(
        tde_csv=tde_csv,
        asassn_index=index_path,
        lc_manifest=manifest_path,
        run_dir=run_dir,
        workers=1,
        chunk_size=1,
    )

    assert len(out) == 1
    assert out.loc[0, "nuclear_primary_hypothesis"] == "tde"
    assert (run_dir / "results" / "gezari2021_asassn_matches.csv").exists()
    assert (run_dir / "results" / "gezari2021_asassn_matches.parquet").exists()
    assert (run_dir / "results" / "gezari2021_nuclear_arbitrated.parquet").exists()

    summary = pd.read_csv(run_dir / "results" / "gezari2021_nuclear_recovery_summary.csv")
    assert summary.loc[0, "nuclear_primary_hypothesis"] == "tde"
    assert summary.loc[0, "tde_candidate_score"] == pytest.approx(0.82)

    candidate_id = out.loc[0, "candidate_id"]
    for path in (
        run_dir / "plots" / "individual" / f"{candidate_id}.png",
        run_dir / "plots" / "individual" / f"{candidate_id}.pdf",
        run_dir / "plots" / "gezari2021_tde_score_grid.png",
        run_dir / "plots" / "gezari2021_tde_score_grid.pdf",
        run_dir / "plots" / "gezari2021_pickups_grid.png",
        run_dir / "plots" / "gezari2021_pickups_grid.pdf",
    ):
        assert path.exists()
        assert path.stat().st_size > 0
