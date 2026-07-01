from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malca.products.feature_layers import to_layer_first_frame

post_filter = pytest.importorskip("malca.stv.filter")


@pytest.mark.parametrize(
    ("pdm_snr", "ce_snr", "expected"),
    [
        (5.0, 5.0, True),
        (7.2, 5.1, True),
        (4.99, 8.0, False),
        (8.0, 4.99, False),
        (np.nan, 6.0, False),
        (6.0, np.nan, False),
    ],
)
def test_is_periodic_by_snr_requires_both_metrics(
    pdm_snr: float,
    ce_snr: float,
    expected: bool,
) -> None:
    assert post_filter._is_periodic_by_snr(pdm_snr, ce_snr) is expected


def test_filter_run_robustness_respects_max_run_count() -> None:
    df = pd.DataFrame(
        {
            "lc_path": ["a.csv", "b.csv"],
            "dip_run_count": [2, 5],
            "jump_run_count": [0, 0],
            "dip_max_run_points": [4, 4],
            "jump_max_run_points": [0, 0],
            "dip_max_run_cameras": [2, 2],
            "jump_max_run_cameras": [0, 0],
        }
    )

    out = post_filter.filter_run_robustness(
        df,
        min_run_count=1,
        max_run_count=3,
        min_run_points=2,
        min_run_cameras=2,
    )

    assert list(out["lc_path"]) == ["a.csv"]


def test_filter_score_branch_specific_thresholds() -> None:
    df = pd.DataFrame(
        {
            "lc_path": ["dip.csv", "jump.csv", "none.csv"],
            "dipper_score": [1.2, -1.0, -2.0],
            "jumper_score": [-1.0, 0.8, -2.0],
        }
    )

    out = post_filter.filter_score(df, min_dip_score=0.5, min_jump_score=0.5)
    assert set(out["lc_path"]) == {"dip.csv", "jump.csv"}


def test_filter_significant_detection_explicit_gate() -> None:
    df = pd.DataFrame(
        {
            "lc_path": ["dip_ok.csv", "no_peak.csv", "flag_false.csv", "jump_ok.csv"],
            "dip_significant": [True, True, False, False],
            "jump_significant": [False, False, False, True],
            "dip_count": [1, 0, 1, 0],
            "jump_count": [0, 0, 0, 1],
            "dip_run_count": [1, 1, 1, 0],
            "jump_run_count": [0, 0, 0, 1],
        }
    )

    out = post_filter.filter_significant_detection(
        df,
        require_significant_flag=True,
        min_peak_count=1,
        min_run_count=1,
    )

    assert set(out["lc_path"]) == {"dip_ok.csv", "jump_ok.csv"}


def test_filter_evidence_strength_skips_missing_bayes_factor_columns() -> None:
    df = pd.DataFrame(
        {
            "lc_path": ["a.csv", "b.csv"],
            "dip_max_log_bf_local": [12.0, 1.0],
            "jump_max_log_bf_local": [np.nan, np.nan],
        }
    )

    out = post_filter.filter_evidence_strength(df)

    assert out["lc_path"].tolist() == ["a.csv", "b.csv"]


def test_apply_filters_hydrates_layer_first_event_metrics() -> None:
    df = pd.DataFrame(
        {
            "candidate_id": ["stv-a", "stv-b"],
            "timescale": ["stv", "stv"],
            "lc_path": ["pass.csv", "fail.csv"],
            "dip_bayes_factor": [20.0, 2.0],
            "jump_bayes_factor": [0.0, 0.0],
            "dip_max_log_bf_local": [8.0, 8.0],
            "jump_max_log_bf_local": [np.nan, np.nan],
        }
    )
    layer_first = to_layer_first_frame(df)

    out = post_filter.apply_filters(
        layer_first,
        apply_evidence_strength=True,
        min_bayes_factor=10.0,
        require_finite_local_bf=True,
        apply_significant_detection=False,
        apply_run_robustness=False,
        apply_morphology=False,
        apply_score=False,
        apply_periodicity_validation=False,
        apply_gaia_ruwe_validation=False,
        apply_gaia_pm_validation=False,
        apply_periodic_catalog_validation=False,
        show_tqdm=False,
        verbose=False,
    )

    out_by_path = out.set_index("lc_path")
    assert int(out_by_path.loc["pass.csv", "failed_posterior_strength"]) == 0
    assert int(out_by_path.loc["fail.csv", "failed_posterior_strength"]) == 1
    assert int(out_by_path.loc["pass.csv", "failed_any"]) == 0
    assert int(out_by_path.loc["fail.csv", "failed_any"]) == 1


def test_apply_filters_tags_significant_detection_failures() -> None:
    df = pd.DataFrame(
        {
            "lc_path": ["pass.csv", "fail.csv"],
            "dip_significant": [True, False],
            "jump_significant": [False, False],
            "dip_count": [1, 0],
            "jump_count": [0, 0],
            "dip_run_count": [1, 0],
            "jump_run_count": [0, 0],
        }
    )
    df.index = pd.Index([10, 20], name="source_pos")

    out = post_filter.apply_filters(
        df,
        apply_evidence_strength=False,
        apply_significant_detection=True,
        apply_run_robustness=False,
        apply_morphology=False,
        apply_score=False,
        apply_periodicity_validation=False,
        apply_gaia_ruwe_validation=False,
        apply_gaia_pm_validation=False,
        apply_periodic_catalog_validation=False,
        show_tqdm=False,
        verbose=False,
    )

    assert "failed_significant_detection" in out.columns
    assert len(out) == len(df)
    assert out.index.equals(df.index)
    assert out.set_index("lc_path").loc["pass.csv", "failed_significant_detection"] == 0
    assert out.set_index("lc_path").loc["fail.csv", "failed_significant_detection"] == 1
    assert out.set_index("lc_path").loc["pass.csv", "failed_any"] == 0
    assert out.set_index("lc_path").loc["fail.csv", "failed_any"] == 1


def test_apply_filters_periodicity_checks_only_prereq_passers(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _fake_validate_periodicity(df: pd.DataFrame, **_kwargs) -> pd.DataFrame:
        seen["paths"] = list(df["lc_path"].astype(str))
        seen["kwargs"] = dict(_kwargs)
        out = df.copy()
        out["lsp_power"] = 0.42
        out["lsp_period"] = 2.5
        out["lsp_bootstrap_sig"] = 0.2
        out["lsp_is_alias"] = False
        out["lsp_is_significant"] = False
        out["periodicity_score"] = 0.7
        out["periodic_flag"] = False
        return out

    monkeypatch.setattr(post_filter, "validate_periodicity", _fake_validate_periodicity)

    df = pd.DataFrame(
        {
            "lc_path": ["pass.csv", "prefail.csv"],
            "dip_significant": [True, False],
            "jump_significant": [False, False],
            "dip_count": [1, 0],
            "jump_count": [0, 0],
            "dip_run_count": [1, 0],
            "jump_run_count": [0, 0],
        }
    )

    out = post_filter.apply_filters(
        df,
        apply_evidence_strength=False,
        apply_significant_detection=True,
        apply_run_robustness=False,
        apply_morphology=False,
        apply_score=False,
        apply_periodicity_validation=True,
        apply_gaia_ruwe_validation=False,
        apply_gaia_pm_validation=False,
        apply_periodic_catalog_validation=False,
        show_tqdm=False,
        verbose=False,
    )

    assert seen["paths"] == ["pass.csv"]
    assert seen["kwargs"]["pdm_method"] == "plavchan"
    out_by_path = out.set_index("lc_path")
    assert float(out_by_path.loc["pass.csv", "lsp_period"]) == 2.5
    assert pd.isna(out_by_path.loc["prefail.csv", "lsp_period"])
    assert int(out_by_path.loc["pass.csv", "failed_periodicity"]) == 0
    assert int(out_by_path.loc["prefail.csv", "failed_periodicity"]) == 0


def test_apply_filters_periodicity_reject_only_marks_checked_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, list[str]] = {}

    def _fake_validate_periodicity(df: pd.DataFrame, **_kwargs) -> pd.DataFrame:
        seen["paths"] = list(df["lc_path"].astype(str))
        out = df.copy()
        out["lsp_power"] = 0.5
        out["lsp_period"] = 1.5
        out["lsp_bootstrap_sig"] = 1e-4
        out["lsp_is_alias"] = False
        out["lsp_is_significant"] = True
        out["periodicity_score"] = 4.0
        out["periodic_flag"] = out["lc_path"].astype(str).eq("reject.csv")
        return out

    monkeypatch.setattr(post_filter, "validate_periodicity", _fake_validate_periodicity)

    df = pd.DataFrame(
        {
            "lc_path": ["reject.csv", "keep.csv", "prefail.csv"],
            "dip_significant": [True, True, False],
            "jump_significant": [False, False, False],
            "dip_count": [1, 1, 0],
            "jump_count": [0, 0, 0],
            "dip_run_count": [1, 1, 0],
            "jump_run_count": [0, 0, 0],
        }
    )

    out = post_filter.apply_filters(
        df,
        apply_evidence_strength=False,
        apply_significant_detection=True,
        apply_run_robustness=False,
        apply_morphology=False,
        apply_score=False,
        apply_periodicity_validation=True,
        periodicity_flag_only=False,
        apply_gaia_ruwe_validation=False,
        apply_gaia_pm_validation=False,
        apply_periodic_catalog_validation=False,
        show_tqdm=False,
        verbose=False,
    )

    assert seen["paths"] == ["reject.csv", "keep.csv"]
    out_by_path = out.set_index("lc_path")
    assert int(out_by_path.loc["reject.csv", "failed_periodicity"]) == 1
    assert int(out_by_path.loc["keep.csv", "failed_periodicity"]) == 0
    assert int(out_by_path.loc["prefail.csv", "failed_periodicity"]) == 0


def test_apply_filters_periodicity_all_candidates_overrides_prereq_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, list[str]] = {}

    def _fake_validate_periodicity(df: pd.DataFrame, **_kwargs) -> pd.DataFrame:
        seen["paths"] = list(df["lc_path"].astype(str))
        out = df.copy()
        out["lsp_power"] = 0.42
        out["lsp_period"] = 2.5
        out["lsp_bootstrap_sig"] = 0.2
        out["lsp_is_alias"] = False
        out["lsp_is_significant"] = False
        out["periodicity_score"] = 0.7
        out["periodic_flag"] = False
        return out

    monkeypatch.setattr(post_filter, "validate_periodicity", _fake_validate_periodicity)

    df = pd.DataFrame(
        {
            "lc_path": ["pass.csv", "prefail.csv"],
            "dip_significant": [True, False],
            "jump_significant": [False, False],
            "dip_count": [1, 0],
            "jump_count": [0, 0],
            "dip_run_count": [1, 0],
            "jump_run_count": [0, 0],
        }
    )

    out = post_filter.apply_filters(
        df,
        apply_evidence_strength=False,
        apply_significant_detection=True,
        apply_run_robustness=False,
        apply_morphology=False,
        apply_score=False,
        apply_periodicity_validation=True,
        periodicity_all_candidates=True,
        apply_gaia_ruwe_validation=False,
        apply_gaia_pm_validation=False,
        apply_periodic_catalog_validation=False,
        show_tqdm=False,
        verbose=False,
    )

    assert seen["paths"] == ["pass.csv", "prefail.csv"]
    out_by_path = out.set_index("lc_path")
    assert float(out_by_path.loc["pass.csv", "lsp_period"]) == 2.5
    assert float(out_by_path.loc["prefail.csv", "lsp_period"]) == 2.5


def test_validate_periodicity_uses_local_bundle_lightcurves(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle_assets" / "lightcurves"
    bundle_dir.mkdir(parents=True)
    local_lc = bundle_dir / "123.dat3"
    local_lc.write_text("stub", encoding="utf-8")

    seen: dict[str, object] = {}

    def _fake_lsp_worker(args: tuple) -> dict[str, object]:
        seen["args"] = args
        original_path, resolved_path, *_ = args
        return {
            "lc_path": original_path,
            "resolved_path": resolved_path,
            "lsp_power": np.nan,
            "lsp_period": 2.5,
            "lsp_bootstrap_sig": 1e-3,
            "lsp_is_alias": False,
            "lsp_is_significant": True,
            "pdm_period": 2.5,
            "pdm_min_theta": 0.2,
            "pdm_snr": 7.0,
            "pdm_bootstrap_sig": 1e-3,
            "pdm_is_significant": True,
            "ce_period": 2.5,
            "ce_min_entropy": 0.25,
            "ce_snr": 6.0,
            "ce_bootstrap_sig": 2e-3,
            "ce_is_significant": True,
            "periodicity_bootstrap_sig": 1e-3,
            "periodicity_is_significant": True,
            "periodicity_is_rejected": True,
            "error": None,
        }

    monkeypatch.setattr(post_filter, "_lsp_worker", _fake_lsp_worker)

    df = pd.DataFrame(
        {
            "lc_path": ["/data/poohbah/cluster/123.dat3"],
            "asas_sn_id": ["123"],
            "n_points": [100],
        }
    )

    out = post_filter.validate_periodicity(
        df,
        n_bootstrap=8,
        significance_level=0.01,
        workers=1,
        skip_if_consensus=False,
        lightcurve_bundle_dir=bundle_dir,
        show_tqdm=False,
        verbose=False,
    )

    assert seen["args"][0] == "/data/poohbah/cluster/123.dat3"
    assert seen["args"][1] == str(local_lc)
    assert seen["args"][5] == "plavchan"
    row = out.iloc[0]
    assert float(row["pdm_snr"]) == 7.0
    assert float(row["ce_snr"]) == 6.0
    assert float(row["lsp_period"]) == 2.5


def test_lsp_worker_aligns_simple_v_minus_g_median_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    jd_g = np.linspace(0.0, 120.0, 80)
    jd_v = jd_g + 0.15
    signal_g = 14.0 + 0.10 * np.sin(2.0 * np.pi * jd_g / 4.0)
    signal_v = 14.0 + 0.10 * np.sin(2.0 * np.pi * jd_v / 4.0) + 0.8

    df_g = pd.DataFrame(
        {
            "JD": jd_g,
            "mag": signal_g,
            "error": np.full(jd_g.size, 0.03, dtype=float),
            "v_g_band": np.zeros(jd_g.size, dtype=int),
        }
    )
    df_v = pd.DataFrame(
        {
            "JD": jd_v,
            "mag": signal_v,
            "error": np.full(jd_v.size, 0.03, dtype=float),
            "v_g_band": np.ones(jd_v.size, dtype=int),
        }
    )

    captured: dict[str, np.ndarray] = {}

    def _fake_read_lc_dat2(*_args: object, **_kwargs: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        captured["file_ext"] = _kwargs.get("file_ext")
        return df_g.copy(), df_v.copy()

    def _fake_pdm_stats(_jd: np.ndarray, mag: np.ndarray, _err: np.ndarray, **_kwargs: object) -> dict[str, float]:
        captured["pdm_mag"] = np.asarray(mag, dtype=float).copy()
        return {
            "pdm_period": 4.0,
            "pdm_min_theta": 0.2,
            "pdm_snr": 7.0,
            "pdm_bootstrap_sig": 1e-3,
            "pdm_is_significant": True,
        }

    def _fake_ce_stats(_jd: np.ndarray, mag: np.ndarray, _err: np.ndarray, **_kwargs: object) -> dict[str, float]:
        captured["ce_mag"] = np.asarray(mag, dtype=float).copy()
        return {
            "ce_period": 4.0,
            "ce_min_entropy": 0.25,
            "ce_snr": 6.0,
            "ce_bootstrap_sig": 2e-3,
            "ce_is_significant": True,
        }

    monkeypatch.setattr(post_filter, "read_lc_dat2", _fake_read_lc_dat2)
    monkeypatch.setattr(post_filter, "compute_pdm_stats", _fake_pdm_stats)
    monkeypatch.setattr(post_filter, "compute_ce_stats", _fake_ce_stats)

    out = post_filter._lsp_worker(
        ("/data/poohbah/cluster/123.dat3", "/tmp/123.dat3", 8, 0.01, True, "plavchan")
    )

    g_count = len(df_g)
    expected_offset = float(np.median(signal_v) - np.median(signal_g))
    for key in ("pdm_mag", "ce_mag"):
        mag = captured[key]
        assert np.isclose(np.median(mag[:g_count]), np.median(mag[g_count:]), atol=1e-10)
    assert out["pdm_method"] == "plavchan"
    assert captured["file_ext"] == "dat3"
    assert np.isclose(expected_offset, 0.8, atol=5e-3)


def test_validate_periodicity_recomputes_stale_checkpoint_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle_assets" / "lightcurves"
    bundle_dir.mkdir(parents=True)
    local_lc = bundle_dir / "123.dat3"
    local_lc.write_text("stub", encoding="utf-8")

    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "lsp_checkpoint.parquet"
    pd.DataFrame(
        [
            {
                "lc_path": "/data/poohbah/cluster/123.dat3",
                "resolved_path": "/data/poohbah/cluster/123.dat3",
                "lsp_period": np.nan,
                "lsp_bootstrap_sig": np.nan,
                "periodicity_bootstrap_sig": np.nan,
                "periodicity_is_significant": False,
                "periodicity_is_rejected": False,
                "pdm_period": np.nan,
                "pdm_min_theta": np.nan,
                "pdm_snr": np.nan,
                "pdm_bootstrap_sig": np.nan,
                "pdm_is_significant": False,
                "ce_period": np.nan,
                "ce_min_entropy": np.nan,
                "ce_snr": np.nan,
                "ce_bootstrap_sig": np.nan,
                "ce_is_significant": False,
                "error": "Light curve file not found",
            }
        ]
    ).to_parquet(checkpoint_path, index=False)

    seen: dict[str, int] = {"calls": 0}

    def _fake_lsp_worker(args: tuple) -> dict[str, object]:
        seen["calls"] += 1
        original_path, resolved_path, *_ = args
        return {
            "lc_path": original_path,
            "resolved_path": resolved_path,
            "lsp_power": np.nan,
            "lsp_period": 3.5,
            "lsp_bootstrap_sig": 5e-4,
            "lsp_is_alias": False,
            "lsp_is_significant": True,
            "pdm_period": 3.5,
            "pdm_min_theta": 0.3,
            "pdm_snr": 5.5,
            "pdm_bootstrap_sig": 5e-4,
            "pdm_is_significant": True,
            "ce_period": 3.5,
            "ce_min_entropy": 0.35,
            "ce_snr": 5.8,
            "ce_bootstrap_sig": 6e-4,
            "ce_is_significant": True,
            "periodicity_bootstrap_sig": 5e-4,
            "periodicity_is_significant": True,
            "periodicity_is_rejected": True,
            "error": None,
        }

    monkeypatch.setattr(post_filter, "_lsp_worker", _fake_lsp_worker)

    df = pd.DataFrame(
        {
            "lc_path": ["/data/poohbah/cluster/123.dat3"],
            "asas_sn_id": ["123"],
            "n_points": [250],
        }
    )

    out = post_filter.validate_periodicity(
        df,
        n_bootstrap=8,
        significance_level=0.01,
        workers=1,
        checkpoint_dir=checkpoint_dir,
        skip_if_consensus=False,
        lightcurve_bundle_dir=bundle_dir,
        show_tqdm=False,
        verbose=False,
    )

    assert seen["calls"] == 1
    row = out.iloc[0]
    assert float(row["pdm_snr"]) == 5.5
    assert float(row["ce_snr"]) == 5.8
    assert bool(row["periodic_flag"]) is True
    saved = pd.read_parquet(checkpoint_path)
    assert int(saved["error"].notna().sum()) == 0
    assert saved.loc[0, "pdm_method"] == "plavchan"
