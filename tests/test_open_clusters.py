from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from malca.enrichment.open_clusters import add_open_cluster_context
from malca.evaluation.open_cluster_enrichment import (
    evaluate_matched_outcomes,
    match_case_controls,
    matched_outcome_statistics,
    prepare_case_control_population,
)
from malca.io.table_io import read_feature_table, write_feature_table
from malca.products.feature_layers import expand_feature_layers, feature_layer_for_column


SOURCE_1 = "5953634325148751488"
SOURCE_2 = "3437905854827453184"
SOURCE_3 = "3131337377071123200"


def _write_ucc(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "README.txt").write_text(
        "These files correspond to the 260615 version of the UCC database.\n",
        encoding="utf-8",
    )
    clusters = pd.DataFrame(
        {
            "name": ["good_cluster", "bad_cluster", "near_cluster"],
            "N_membs": [100, 80, 60],
            "r_50": [10.0, 5.0, 20.0],
            "r_core": [1.0, 1.5, 2.0],
            "RA_ICRS": [10.0, 10.1, 30.0],
            "DE_ICRS": [20.0, 20.1, -10.0],
            "Plx": [2.0, 2.1, 1.0],
            "pmRA": [3.0, 3.1, 0.5],
            "pmDE": [-2.0, -2.1, 0.2],
            "Rv": [10.0, 11.0, np.nan],
            "Dist_[kpc]": [0.5, 0.48, 1.0],
            "Dist_STDDEV": [0.02, 0.02, 0.1],
            "Av_[mag]": [0.2, 0.3, 0.1],
            "Av_STDDEV": [0.05, 0.05, 0.02],
            "Age_[Myr]": [5.0, 7.0, 100.0],
            "Age_STDDEV": [1.0, 1.5, 10.0],
            "FeH_[dex]": [0.0, -0.1, 0.1],
            "FeH_STDDEV": [0.1, 0.1, 0.1],
            "Mass_[Msun]": [500.0, 300.0, 200.0],
            "Mass_STDDEV": [50.0, 30.0, 20.0],
            "C3": ["AA", "DD", "AB"],
            "P_dup": [0.0, 0.1, 0.0],
            "UTI": [0.9, 0.1, 0.8],
            "bad_oc": ["n", "y", "n"],
        }
    )
    clusters.to_csv(root / "UCC_cat.csv", index=False)
    pd.DataFrame(
        {
            "name": ["bad_cluster", "good_cluster", "good_cluster"],
            "Source": [int(SOURCE_1), int(SOURCE_1), int(SOURCE_2)],
            "probs": [0.99, 0.60, 0.40],
        }
    ).to_parquet(root / "UCC_members.parquet", index=False)
    return root


def _write_hr24(root: Path) -> Path:
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "Name": ["HR_bound", "HR_moving", "HR_weak"],
            "ID": [1, 2, 3],
            "Type": ["o", "m", "o"],
            "CST": [6.0, 9.0, 4.0],
            "CMDCl50": [0.8, 0.9, 0.9],
            "N": [100, 80, 40],
            "RAdeg": [10.0, 11.0, 30.0],
            "DEdeg": [20.0, 21.0, -10.0],
            "r50": [0.2, 0.3, 0.4],
            "r50pc": [2.0, 3.0, 4.0],
            "Plx": [2.0, 1.8, 1.0],
            "pmRA": [3.0, 3.2, 0.5],
            "pmDE": [-2.0, -2.2, 0.2],
            "dist50": [500.0, 550.0, 1000.0],
            "logAge50": [6.7, 7.0, 8.0],
            "probJ": [0.9, 0.1, 0.8],
            "MassJ": [400.0, 100.0, 150.0],
            "MassTot": [500.0, 200.0, 220.0],
        }
    ).to_csv(root / "clusters.csv", index=False)
    pd.DataFrame(
        {
            "Name": ["HR_bound", "HR_moving", "HR_weak"],
            "ID": [1, 2, 3],
            "GaiaDR3": [SOURCE_1, SOURCE_2, SOURCE_3],
            "Prob": [0.70, 0.90, 0.80],
            "inrj": [1, 0, 1],
            "inrt": [1, 1, 1],
        }
    ).to_csv(root / "members.csv", index=False)
    return root


def test_exact_id_membership_preserves_multiple_matches_and_quality(tmp_path: Path) -> None:
    ucc = _write_ucc(tmp_path / "ucc" / "260615")
    hr24 = _write_hr24(tmp_path / "hr24")
    sources = pd.DataFrame(
        {
            "candidate_id": ["c1", "c2", "c3", "c4"],
            "source_id": [SOURCE_1, SOURCE_2, SOURCE_3, pd.NA],
            "ra": [10.02, 10.03, 30.01, 0.0],
            "dec": [20.01, 20.02, -10.01, 0.0],
            "parallax": [2.02, 2.03, 1.01, np.nan],
            "pmra": [3.1, 3.2, 0.6, np.nan],
            "pmdec": [-2.1, -2.2, 0.3, np.nan],
        }
    )
    result = add_open_cluster_context(sources, ucc_dir=ucc, hr24_dir=hr24)
    out = result.sources.set_index("candidate_id")

    assert len(out) == len(sources)
    assert out.loc["c1", "open_cluster_gaia_id"] == SOURCE_1
    assert out.loc["c1", "ucc_cluster"] == "good_cluster"
    assert out.loc["c1", "ucc_n_matches"] == 2
    assert bool(out.loc["c1", "ucc_good_member"])
    assert out.loc["c1", "ucc_pmem"] == 0.60
    assert not bool(out.loc["c2", "ucc_p50_member"])
    assert out.loc["c2", "ucc_match_status"] == "listed_below_threshold"
    assert out.loc["c4", "ucc_match_status"] == "missing_gaia_id"
    assert out.loc["c1", "cluster_name"] == "good_cluster"
    assert out.loc["c1", "cluster_dist_pc"] == 500.0

    assert bool(out.loc["c1", "hr24_high_quality_member"])
    assert not bool(out.loc["c2", "hr24_bound_member"])
    assert bool(out.loc["c3", "hr24_bound_member"])
    assert not bool(out.loc["c3", "hr24_high_quality_member"])
    assert len(result.all_matches.query("candidate_id == 'c1'")) == 3
    assert set(result.all_matches["cluster_catalog"]) == {"UCC", "Hunt & Reffert 2024"}


def test_proximity_is_separate_and_excludes_bad_clusters(tmp_path: Path) -> None:
    ucc = _write_ucc(tmp_path / "ucc" / "260615")
    sources = pd.DataFrame(
        {
            "candidate_id": ["near_bad_but_not_member"],
            "source_id": [SOURCE_3],
            "ra": [10.1],
            "dec": [20.1],
            "parallax": [2.2],
            "pmra": [3.4],
            "pmdec": [-2.4],
        }
    )
    result = add_open_cluster_context(sources, ucc_dir=ucc, include_proximity=True)
    row = result.sources.iloc[0]
    assert not bool(row["ucc_listed_member"])
    assert row["ucc_nearest_cluster"] == "good_cluster"
    assert float(row["ucc_nearest_sep_arcmin"]) > 0
    assert np.isclose(float(row["ucc_nearest_dparallax_mas"]), 0.2)


def test_open_cluster_fields_roundtrip_through_feature_layers(tmp_path: Path) -> None:
    path = tmp_path / "membership.parquet"
    frame = pd.DataFrame(
        {
            "candidate_id": ["c1"],
            "timescale": ["stv"],
            "lc_path": ["one.dat3"],
            "ucc_cluster": ["good_cluster"],
            "ucc_pmem": [0.9],
            "ucc_good_member": [True],
            "hr24_high_quality_member": [True],
        }
    )
    write_feature_table(frame, path)
    expanded = expand_feature_layers(read_feature_table(path))
    assert expanded.loc[0, "ucc_cluster"] == "good_cluster"
    assert bool(expanded.loc[0, "ucc_good_member"])
    assert bool(expanded.loc[0, "hr24_high_quality_member"])
    assert feature_layer_for_column("ucc_cluster") == "external_stats"
    assert feature_layer_for_column("ucc_good_member") == "derived_stats"


def _matching_population() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    labels = []
    for index in range(3):
        candidate_id = f"case{index}"
        rows.append(
            {
                "candidate_id": candidate_id,
                "open_cluster_gaia_id": str(1000 + index),
                "gal_l": 10.0 + index,
                "gal_b": 1.0,
                "phot_g_mean_mag": 13.0,
                "distance_gspphot": 500.0,
                "ucc_good_member": index < 2,
                "hr24_bound_member": index == 0,
                "hr24_high_quality_member": index == 0,
                "hr24_match_status": "high_quality_bound_member" if index == 0 else "no_match",
            }
        )
        labels.append({"candidate_id": candidate_id, "event_class": "dipper", "workflow_status": "reviewed"})
        for control in range(3):
            control_id = f"control{index}_{control}"
            rows.append(
                {
                    "candidate_id": control_id,
                    "open_cluster_gaia_id": str(2000 + 10 * index + control),
                    "gal_l": 10.0 + index + 0.05 * control,
                    "gal_b": 1.1,
                    "phot_g_mean_mag": 13.1,
                    "distance_gspphot": 510.0,
                    "ucc_good_member": False,
                    "hr24_bound_member": False,
                    "hr24_high_quality_member": False,
                    "hr24_match_status": "no_match",
                }
            )
            labels.append({"candidate_id": control_id, "event_class": "periodic", "workflow_status": "reviewed"})
    return pd.DataFrame(rows), pd.DataFrame(labels)


def test_matched_case_control_analysis_is_deterministic() -> None:
    membership, labels = _matching_population()
    population = prepare_case_control_population(membership, labels)
    matched, audit = match_case_controls(
        population,
        controls_per_case=2,
        sky_caliper_deg=2.0,
        latitude_caliper_deg=1.0,
        g_caliper_mag=0.5,
        fractional_distance_caliper=0.1,
    )
    assert matched["match_role"].eq("case").sum() == 3
    assert matched["match_role"].eq("control").sum() == 6
    assert audit["status"].eq("matched").all()
    statistics = evaluate_matched_outcomes(
        matched,
        bootstrap_draws=100,
        permutation_draws=200,
        seed=12,
    )
    assert set(statistics["outcome"]) == {
        "ucc_good_member",
        "hr24_bound_member",
        "hr24_high_quality_member",
    }
    ucc = statistics.set_index("outcome").loc["ucc_good_member"]
    assert ucc["case_positive"] == 2
    assert ucc["control_positive"] == 0
    assert np.isinf(ucc["mantel_haenszel_odds_ratio"])


def test_matched_statistics_distinguish_adjusted_and_unadjusted_outputs() -> None:
    matched = pd.DataFrame(
        {
            "match_set_id": ["a", "a", "b", "b"],
            "match_role": ["case", "control", "case", "control"],
            "ucc_good_member": [True, False, False, False],
        }
    )
    result = matched_outcome_statistics(
        matched,
        "ucc_good_member",
        seed=1,
        bootstrap_draws=20,
        permutation_draws=100,
    )
    assert result["match_sets"] == 2
    assert result["case_positive"] == 1
    assert result["control_positive"] == 0
    assert "matched_permutation_p" in result
    assert "unadjusted_fisher_p" in result
