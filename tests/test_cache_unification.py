from __future__ import annotations

from pathlib import Path

import pandas as pd

from malca import config
from malca import gaia_fetch
from malca import characterize
from malca.review import sed
from malca import vetting


def test_cache_defaults_are_under_output_cache() -> None:
    output_root = Path("output_migrated_camera_field_20260606")
    assert config.DEFAULT_OUTPUT_DIR == output_root
    assert config.MALCA_CACHE_ROOT == output_root / "cache"
    assert config.DEFAULT_CACHE_DIR == output_root / "cache" / "catalogs"
    assert config.GAIA_LOCAL_CATALOG == output_root / "cache" / "catalogs" / "gaia" / "gaia_dr3_crossmatched.parquet"
    assert config.SKYPATROL_CACHE_DIR == output_root / "cache" / "lightcurves" / "skypatrol"
    assert config.LTV_CACHE_DIR == output_root / "cache" / "joblib" / "ltv"
    assert config.REVIEW_IMPORTED_LC_CACHE_DIR == output_root / "cache" / "review" / "imported_lightcurves"
    assert vetting._vetting_cache_path(None, "simbad") == output_root / "cache" / "catalogs" / "vetting" / "vetting_simbad.parquet"


def test_gaia_fetch_reads_legacy_default_and_migrates_to_new_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    new_cache = tmp_path / "output" / "cache" / "catalogs" / "gaia" / "gaia_dr3_crossmatched.parquet"
    legacy_cache = tmp_path / "input" / "gaia" / "gaia_dr3_crossmatched.parquet"
    legacy_cache.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "source_id": ["123"],
            "ra": [1.0],
            "dec": [2.0],
            "phot_bp_mean_mag": [15.1],
            "phot_rp_mean_mag": [14.2],
            "w1": [11.0],
            "w1_err": [0.1],
            "w2": [10.8],
            "w2_err": [0.1],
            "w3": [10.0],
            "w3_err": [0.2],
            "w4": [9.5],
            "w4_err": [0.3],
        }
    ).to_parquet(legacy_cache, index=False)

    monkeypatch.setattr(gaia_fetch, "GAIA_LOCAL_CATALOG", new_cache)
    monkeypatch.setattr(gaia_fetch, "LEGACY_GAIA_LOCAL_CATALOG", legacy_cache)
    monkeypatch.setattr(gaia_fetch, "LEGACY_GAIA_CACHE_FILE", tmp_path / "output" / "gaia_cache.parquet")

    out = gaia_fetch.fetch_gaia_catalog(["123"], output_path=new_cache)

    assert len(out) == 1
    assert new_cache.exists()
    migrated = pd.read_parquet(new_cache)
    assert migrated["source_id"].astype(str).tolist() == ["123"]


def test_sed_source_cache_persists_hits_and_misses(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sed, "SED_CACHE_DIR", tmp_path / "sed")
    calls = {"ps1": 0}

    def ps1_fetch(df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        calls["ps1"] += 1
        row = {col: None for col in sed.SED_COLUMNS}
        row.update(
            {
                "candidate_id": str(df.iloc[0]["candidate_id"]),
                "source": "Pan-STARRS",
                "band": "g",
                "mag": 15.0,
                "mag_system": "AB",
            }
        )
        return pd.DataFrame([row], columns=sed.SED_COLUMNS)

    monkeypatch.setitem(sed.CATALOG_FETCHERS, "ps1", ps1_fetch)

    df = pd.DataFrame([{"candidate_id": "C1", "ra": 1.0, "dec": 2.0}])
    first = sed.fetch_sed_photometry(df, sources="ps1")
    second = sed.fetch_sed_photometry(df, sources="ps1")

    assert len(first) == 1
    assert len(second) == 1
    assert calls == {"ps1": 1}
    assert (tmp_path / "sed" / "ps1.parquet").exists()


def test_characterization_module_cache_persists_rows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(characterize, "CHARACTERIZE_CACHE_DIR", tmp_path / "characterize")
    calls = {"n": 0}
    df = pd.DataFrame(
        {
            "source_id": ["123"],
            "ra": [1.0],
            "dec": [2.0],
        }
    )

    def fake_apass(frame: pd.DataFrame) -> pd.DataFrame:
        calls["n"] += 1
        out = frame.copy()
        out["apass_v"] = 14.2
        out["apass_v_err"] = 0.03
        return out

    first = characterize._run_cached_characterization_module(
        df,
        module="apass",
        func=fake_apass,
        output_columns=["apass_v", "apass_v_err"],
    )
    second = characterize._run_cached_characterization_module(
        df,
        module="apass",
        func=lambda _frame: (_ for _ in ()).throw(AssertionError("cache miss")),
        output_columns=["apass_v", "apass_v_err"],
    )

    assert calls["n"] == 1
    assert float(first.loc[0, "apass_v"]) == 14.2
    assert float(second.loc[0, "apass_v"]) == 14.2
    assert (tmp_path / "characterize" / "apass.parquet").exists()
