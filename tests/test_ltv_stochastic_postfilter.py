from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca.ltv import pipeline as ltv_pipeline
from malca.ltv import stochastic as ltv_stochastic


def _write_dat2(path: Path) -> None:
    lines = [
        "1000.0 14.0 0.05 1 1 0 0 cam1/field1",
        "1010.0 14.1 0.05 1 1 0 0 cam1/field1",
        "1020.0 14.2 0.05 1 1 0 0 cam1/field1",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_add_stochastic_postfilter_features_merges_columns(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dat2_path = tmp_path / "123.dat2"
    _write_dat2(dat2_path)

    def _fake_bundle(jd, mag, err, *, include_drw):
        assert len(jd) == 3
        assert len(mag) == 3
        assert len(err) == 3
        return {
            "stoch_sf_ml_amplitude": 0.4,
            "stoch_sf_ml_gamma": 0.7,
            "stoch_iar_phi": 0.9,
            "stoch_mhps_high": 1.0,
            "stoch_mhps_low": 2.0,
            "stoch_mhps_non_zero": 3.0,
            "stoch_mhps_pn_flag": 0.0,
            "stoch_mhps_ratio": 2.0,
            "stoch_gp_drw_sigma": 0.05 if include_drw else float("nan"),
            "stoch_gp_drw_tau": 120.0 if include_drw else float("nan"),
        }

    monkeypatch.setattr(ltv_stochastic, "_load_stochastic_functions", lambda include_drw: {})
    monkeypatch.setattr(ltv_stochastic, "_compute_feature_bundle", _fake_bundle)

    df = pd.DataFrame({"lc_path": [str(dat2_path)]})
    out = ltv_stochastic.add_stochastic_postfilter_features(
        df,
        include_drw=True,
        n_workers=1,
        verbose=False,
    )

    assert out.loc[0, "stoch_sf_ml_amplitude"] == 0.4
    assert out.loc[0, "stoch_sf_ml_gamma"] == 0.7
    assert out.loc[0, "stoch_iar_phi"] == 0.9
    assert out.loc[0, "stoch_mhps_ratio"] == 2.0
    assert out.loc[0, "stoch_gp_drw_tau"] == 120.0


def test_run_full_pipeline_invokes_stochastic_stage(monkeypatch) -> None:
    seen: dict[str, bool] = {"called": False}

    def _fake_stochastic(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        seen["called"] = True
        out = df.copy()
        out["stoch_sf_ml_amplitude"] = 1.23
        out["stoch_iar_phi"] = 0.88
        return out

    monkeypatch.setattr(ltv_pipeline, "add_stochastic_postfilter_features", _fake_stochastic)

    df = pd.DataFrame({"lc_path": ["/tmp/fake.dat2"], "Slope": [0.1], "max diff": [0.5]})
    out = ltv_pipeline.run_full_pipeline(
        df,
        run_filters=False,
        run_stochastic_postfilter=True,
        run_crossmatch=False,
        run_neowise=False,
        run_extinction=False,
        run_dust_flags=False,
        run_cmd=False,
        run_bailer_jones=False,
        run_gaia_epoch=False,
        verbose=False,
    )

    assert seen["called"] is True
    assert out.loc[0, "stoch_sf_ml_amplitude"] == 1.23
    assert out.loc[0, "stoch_iar_phi"] == 0.88
