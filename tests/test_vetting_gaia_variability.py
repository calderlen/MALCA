from __future__ import annotations

import pandas as pd

import malca.enrichment.vetting as vetting


def test_gaia_variability_surfaces_classifier_for_numeric_like_gaia_ids(monkeypatch, tmp_path) -> None:
    df = pd.DataFrame(
        {
            "candidate_id": ["GAIA-CLASSIFIED", "GAIA-UNCLASSIFIED"],
            "gaia_id": ["123.0", "4.56e2"],
        }
    )

    monkeypatch.setattr(
        vetting,
        "_connect_gaia_taps_until_available",
        lambda *args, **kwargs: ["fake-tap"],
    )

    def fake_run_query(_taps, query: str, *, label: str):
        if "gaiadr3.vari_summary" in query:
            return [
                {"source_id": 123, "in_vari_classification_result": False},
                {"source_id": 456, "in_vari_classification_result": False},
            ]
        if "gaiadr3.vari_classifier_result" in query:
            return [
                {"source_id": 123, "best_class_name": "LPV", "best_class_score": 0.91},
            ]
        raise AssertionError(f"unexpected Gaia query for {label}: {query}")

    monkeypatch.setattr(vetting, "_run_gaia_tap_query_until_success", fake_run_query)

    out = vetting.query_gaia_variability(df, chunk_size=100, cache_dir=tmp_path)

    assert bool(out.loc[0, "gaia_var_flag"]) is True
    assert out.loc[0, "gaia_var_class"] == "LPV"
    assert float(out.loc[0, "gaia_var_score"]) == 0.91
    assert bool(out.loc[1, "gaia_var_flag"]) is False
    assert out.loc[1, "gaia_var_class"] == ""


def test_gaia_variable_flag_alone_does_not_mark_candidate_likely_known() -> None:
    df = pd.DataFrame(
        {
            "candidate_id": ["GAIA-FLAG-ONLY", "GAIA-CLASSIFIED"],
            "gaia_var_flag": [True, False],
            "gaia_var_class": ["", "LPV"],
        }
    )

    out = vetting.vet_candidates(
        df,
        run_simbad=False,
        run_gaia_var=False,
        run_asassn_var=False,
        run_microlens=False,
        run_ztf_var=False,
        run_tns=False,
        run_gaia_eb=False,
        run_alerce=False,
        run_atlas=False,
        run_gaia_epoch=False,
        run_erosita=False,
        run_chandra_csc=False,
        run_pm_check=False,
    )

    assert bool(out.loc[0, "vetting_likely_known"]) is False
    assert bool(out.loc[1, "vetting_likely_known"]) is True


def test_null_catalog_labels_and_false_text_flags_do_not_mark_candidate_known() -> None:
    df = pd.DataFrame(
        {
            "candidate_id": ["UNKNOWN", "KNOWN"],
            "gaia_var_class": [None, None],
            "asassn_var_type": [None, None],
            "ztf_var_type": [None, None],
            "tns_name": [None, None],
            "alerce_lc_class": [None, None],
            "microlens_match": ["False", "yes"],
        }
    )

    out = vetting.vet_candidates(
        df,
        run_simbad=False,
        run_gaia_var=False,
        run_asassn_var=False,
        run_microlens=False,
        run_ztf_var=False,
        run_tns=False,
        run_gaia_eb=False,
        run_alerce=False,
        run_atlas=False,
        run_gaia_epoch=False,
        run_erosita=False,
        run_chandra_csc=False,
        run_pm_check=False,
    )

    assert bool(out.loc[0, "vetting_likely_known"]) is False
    assert bool(out.loc[1, "vetting_likely_known"]) is True
