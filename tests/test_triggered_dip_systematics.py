from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malca.meta_analysis import triggered_dip_systematics as systematics


def test_prepare_best_dip_events_converts_time_and_audits_qc() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_id": ["good", "missing", "outside", "quiet"],
            "dip_significant": [True, True, True, False],
            "dip_best_t0": [100.0, np.nan, 120.0, 100.0],
            "dip_trigger_max": [9.0, 8.0, 7.0, 6.0],
            "jd_first": [99.0, 99.0, 99.0, 99.0],
            "jd_last": [101.0, 101.0, 101.0, 101.0],
            # This intentionally would reject the good row. The event-provenance
            # jd_first/jd_last bounds must take precedence when both are present.
            "stats_jd_start": [100.5, 99.0, 99.0, 99.0],
            "stats_jd_end": [101.0, 101.0, 101.0, 101.0],
            "ra": [10.0, 20.0, 30.0, 40.0],
            "dec": [-10.0, -20.0, -30.0, -40.0],
            "distance_gspphot": [250.0, 300.0, 350.0, 400.0],
        }
    )

    # Jupyter kernels can inherit strict NumPy error handling. Missing event
    # times must remain QC rows rather than crashing pandas' date conversion.
    with np.errstate(over="raise", invalid="raise"):
        events, qc = systematics.prepare_best_dip_events(candidates)

    assert events["source_key"].tolist() == ["good"]
    event = events.iloc[0]
    assert event["dip_jd"] == pytest.approx(2_450_100.0)
    assert event["dip_mjd"] == pytest.approx(50_099.5)
    assert event["dip_night_mjd"] == 50_099
    assert pd.notna(event["dip_night"])

    reasons = qc.set_index("source_key")["qc_reasons"].to_dict()
    assert reasons == {
        "good": "",
        "missing": "missing_t0",
        "outside": "outside_observed_span",
        "quiet": "not_triggered",
    }
    assert qc.attrs["summary"]["observation_span_source"] == "jd_first/jd_last"
    assert qc.attrs["summary"]["n_valid_events"] == 1

    fallback_events, fallback_qc = systematics.prepare_best_dip_events(
        candidates.drop(columns=["jd_first", "jd_last"])
    )
    assert "good" not in set(fallback_events["source_key"])
    assert fallback_qc.attrs["summary"]["observation_span_source"] == "stats_jd_start/stats_jd_end"


def test_summarize_nights_excludes_nonoverlapping_candidate_spans() -> None:
    events = pd.DataFrame({"dip_night_mjd": [50_010, 50_012]})
    candidates = pd.DataFrame(
        {
            "jd_first": [-100.0, 100.0, 10.5],
            "jd_last": [-90.0, 110.0, 12.5],
        }
    )

    nights = systematics.summarize_nights(events, all_candidates=candidates)

    assert nights["n_exposed_sources"].tolist() == [1, 1, 1]
    assert set(nights["exposure_source"]) == {"candidate_baseline_approximation"}


def test_group_trigger_rates_include_wilson_and_fisher_inference() -> None:
    candidates = pd.DataFrame(
        {
            "asassn_field_key": ["hot"] * 4 + ["cold"] * 4 + ["zero"] * 2,
            "camera_name_key": ["cam-hot"] * 4 + ["cam-cold"] * 6,
            "dip_significant": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        }
    )

    rates = systematics.candidate_group_trigger_rates(candidates)
    field = rates["field"].set_index("asassn_field_key")

    assert field.loc["hot", "n_candidates"] == 4
    assert field.loc["hot", "n_triggered"] == 4
    assert field.loc["hot", "trigger_rate"] == 1.0
    assert field.loc["hot", "rate_ci_high"] == 1.0
    assert field.loc["zero", "rate_ci_low"] == 0.0
    assert np.all(field["rate_ci_low"] <= field["trigger_rate"])
    assert np.all(field["trigger_rate"] <= field["rate_ci_high"])
    assert field["pvalue_overrepresented"].between(0.0, 1.0).all()
    assert field["qvalue_overrepresented"].between(0.0, 1.0).all()
    assert set(rates) == {"field", "camera", "camera_field"}


def test_prepare_triggered_runs_preserves_actual_run_labels_and_distance() -> None:
    runs = pd.DataFrame(
        {
            "event_id": ["a:1", "a:2", "b:1"],
            "candidate_id": ["a", "a", "b"],
            "source_key": ["a", "a", "b"],
            "run_number": [1, 2, 1],
            "run_start_jd": [2_450_100.0, 2_450_110.0, 2_450_101.0],
            "run_end_jd": [2_450_102.0, 2_450_111.0, 2_450_103.0],
            "dip_jd": [2_450_101.0, 2_450_110.5, 2_450_102.0],
            "dip_mjd": [50_100.5, 50_110.0, 50_101.5],
            "asassn_field_key": ["run-f1", "run-f2", "run-f1"],
            "camera_name_key": ["run-c1", "run-c2", "run-c1"],
            "n_trigger_points": [2, 3, 2],
        }
    )
    candidates = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "dip_run_count": [2, 1],
            "ra": [10.0, 10.1],
            "dec": [0.0, 0.0],
            "distance_gspphot": [100.0, 200.0],
            # Candidate-modal labels must not replace actual run labels.
            "asassn_field_key": ["modal-a", "modal-b"],
            "camera_name_key": ["modal-a", "modal-b"],
        }
    )

    events, qc = systematics.prepare_triggered_dip_runs(runs, candidates)

    assert events["event_id"].tolist() == ["a:1", "b:1", "a:2"]
    assert events["source_key"].value_counts().to_dict() == {"a": 2, "b": 1}
    assert set(events["asassn_field_key"]) == {"run-f1", "run-f2"}
    assert events.set_index("event_id").loc["a:1", "distance_pc"] == 100.0
    assert qc["qc_valid"].all()
    assert qc.attrs["summary"]["n_source_count_mismatches"] == 0


def test_prepare_triggered_runs_uses_a_joint_modal_trigger_group() -> None:
    runs = pd.DataFrame(
        {
            "event_id": ["a:1"], "candidate_id": ["a"], "source_key": ["a"],
            "run_number": [1], "run_start_jd": [2_450_100.0], "run_end_jd": [2_450_101.0],
            "dip_jd": [2_450_100.5], "dip_mjd": [50_100.0],
            # Independent lexical tie-breaking would incorrectly produce F1/C1.
            "asassn_field_key": ["F1"], "camera_name_key": ["C1"],
            "trigger_fields_json": ['["F1","F2"]'],
            "trigger_cameras_json": ['["C2","C1"]'],
        }
    )
    candidates = pd.DataFrame(
        {"candidate_id": ["a"], "dip_run_count": [1], "ra": [1.0], "dec": [2.0], "distance_pc": [100.0]}
    )

    events, _ = systematics.prepare_triggered_dip_runs(runs, candidates)

    assert events.loc[0, "asassn_field_key"] == "F1"
    assert events.loc[0, "camera_name_key"] == "C2"
    assert events.loc[0, "run_group_attribution"] == "joint_modal_trigger_pair"


def test_nearby_run_pairs_detect_overlapping_intervals() -> None:
    runs = pd.DataFrame(
        {
            "event_id": ["a:1", "a:2", "b:1", "c:1"],
            "source_key": ["a", "a", "b", "c"],
            "ra": [359.9, 359.9, 0.1, 5.0],
            "dec": [0.0, 0.0, 0.0, 0.0],
            "run_start_mjd": [10.0, 30.0, 12.0, 10.0],
            "run_end_mjd": [12.5, 31.0, 13.0, 11.0],
            "dip_mjd": [11.0, 30.5, 12.5, 10.5],
            "asassn_field_key": ["F", "F", "F", "G"],
            "camera_name_key": ["C", "C", "C", "D"],
        }
    )

    pairs = systematics.build_nearby_run_pairs(
        runs, max_sep_deg=0.5, max_interval_gap_days=1.0
    )

    assert len(pairs) == 1
    pair = pairs.iloc[0]
    assert {pair["event_id_i"], pair["event_id_j"]} == {"a:1", "b:1"}
    assert pair["time_lag_days"] == 0.0
    assert pair["peak_time_lag_days"] == pytest.approx(1.5)
    assert bool(pair["intervals_overlap"])
    assert bool(pair["same_field"])
    assert bool(pair["same_camera"])


def test_triggered_run_rates_include_trigger_point_rate() -> None:
    runs = pd.DataFrame(
        {
            "event_id": ["a:1", "a:2", "b:1"],
            "source_key": ["a", "a", "b"],
            "n_trigger_points": [2, 3, 4],
            "asassn_field_key": ["F", "F", "G"],
            "camera_name_key": ["C", "C", "D"],
        }
    )
    exposures = pd.DataFrame(
        {
            "source_key": ["a", "b"],
            "n_observations": [100, 200],
            "n_observed_nights": [50, 80],
            "asassn_field_key": ["F", "G"],
            "camera_name_key": ["C", "D"],
        }
    )

    rates = systematics.triggered_run_group_rates(runs, exposures)
    field = rates["field"].set_index("asassn_field_key")

    assert field.loc["F", "n_runs"] == 2
    assert field.loc["F", "runs_per_1000_observations"] == pytest.approx(20.0)
    assert field.loc["G", "trigger_points_per_1000_observations"] == pytest.approx(20.0)
    assert field["rate_error_per_1000_observations"].notna().all()
    assert field["qvalue_run_overdensity"].between(0.0, 1.0).all()


def test_prepare_triggered_runs_preserves_every_run_and_checks_source_counts() -> None:
    run_table = pd.DataFrame(
        {
            "event_id": ["a:dip_run_0001", "a:dip_run_0002", "b:dip_run_0001"],
            "candidate_id": ["a", "a", "b"],
            "source_key": ["a", "a", "b"],
            "run_number": [1, 2, 1],
            "run_start_jd": [2_450_100.0, 2_450_110.0, 2_450_105.0],
            "run_end_jd": [2_450_101.0, 2_450_112.0, 2_450_106.0],
            "dip_jd": [2_450_100.5, 2_450_111.0, 2_450_105.5],
            "dip_mjd": [50_100.0, 50_110.5, 50_105.0],
            "asassn_field_key": ["F1", "F2", "F1"],
            "camera_name_key": ["C1", "C2", "C1"],
        }
    )
    candidates = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "dip_run_count": [2, 1],
            "ra": [10.0, 10.1],
            "dec": [0.0, 0.0],
            "distance_pc": [100.0, 200.0],
            # Candidate modal labels must not overwrite actual run labels.
            "asassn_field_key": ["modal-a", "modal-b"],
            "camera_name_key": ["modal-a", "modal-b"],
        }
    )

    events, qc = systematics.prepare_triggered_dip_runs(run_table, candidates)

    assert len(events) == 3
    assert events["event_id"].is_unique
    assert events.loc[events["event_id"].eq("a:dip_run_0002"), "asassn_field_key"].item() == "F2"
    assert events.groupby("source_key").size().to_dict() == {"a": 2, "b": 1}
    assert qc["qc_valid"].all()
    assert qc.attrs["summary"]["n_source_count_mismatches"] == 0


def test_prepare_triggered_runs_retains_replay_count_mismatches_as_warning() -> None:
    run_table = pd.DataFrame(
        {
            "event_id": ["a:run1"], "candidate_id": ["a"], "source_key": ["a"],
            "run_number": [1], "run_start_jd": [2_450_100.0], "run_end_jd": [2_450_101.0],
            "dip_jd": [2_450_100.5], "dip_mjd": [50_100.0],
            "asassn_field_key": ["F"], "camera_name_key": ["C"],
        }
    )
    candidates = pd.DataFrame(
        {"candidate_id": ["a"], "dip_run_count": [2], "ra": [1.0], "dec": [2.0], "distance_pc": [100.0]}
    )

    events, qc = systematics.prepare_triggered_dip_runs(run_table, candidates)

    assert len(events) == 1
    assert bool(qc.loc[0, "qc_valid"])
    assert not bool(qc.loc[0, "qc_source_run_count_matches"])
    assert "source_run_count_mismatch" in qc.loc[0, "qc_warnings"]
    assert qc.attrs["summary"]["n_source_count_mismatches"] == 1


def test_run_pairs_use_interval_gap_and_exclude_same_source() -> None:
    runs = pd.DataFrame(
        {
            "event_id": ["a1", "a2", "b1", "c1"],
            "source_key": ["a", "a", "b", "c"],
            "ra": [359.9, 359.9, 0.1, 5.0],
            "dec": [0.0, 0.0, 0.0, 0.0],
            "dip_mjd": [10.5, 30.5, 12.5, 10.5],
            "run_start_mjd": [10.0, 30.0, 12.0, 10.0],
            "run_end_mjd": [11.0, 31.0, 13.0, 11.0],
            "asassn_field_key": ["F", "F", "F", "G"],
            "camera_name_key": ["C", "C", "C", "D"],
        }
    )

    pairs = systematics.build_nearby_run_pairs(
        runs, max_sep_deg=0.5, max_interval_gap_days=1.0
    )

    assert len(pairs) == 1
    pair = pairs.iloc[0]
    assert {pair["event_id_i"], pair["event_id_j"]} == {"a1", "b1"}
    assert pair["time_lag_days"] == pytest.approx(1.0)
    assert pair["peak_time_lag_days"] == pytest.approx(2.0)
    assert pair["source_i"] != pair["source_j"]
    assert bool(pair["same_field"])
    assert bool(pair["same_camera"])


def test_triggered_run_rates_use_observation_exposure() -> None:
    runs = pd.DataFrame(
        {
            "event_id": ["a1", "a2", "b1"],
            "source_key": ["a", "a", "b"],
            "n_trigger_points": [2, 3, 4],
            "asassn_field_key": ["F", "F", "G"],
            "camera_name_key": ["C", "C", "D"],
        }
    )
    exposures = pd.DataFrame(
        {
            "source_key": ["a", "b"],
            "n_observations": [100, 200],
            "n_observed_nights": [50, 80],
            "asassn_field_key": ["F", "G"],
            "camera_name_key": ["C", "D"],
        }
    )

    rates = systematics.triggered_run_group_rates(runs, exposures)
    field = rates["field"].set_index("asassn_field_key")

    assert field.loc["F", "n_runs"] == 2
    assert field.loc["F", "n_trigger_points"] == 5
    assert field.loc["F", "runs_per_1000_observations"] == pytest.approx(20.0)
    assert set(rates) == {"field", "camera", "camera_field"}


def test_run_level_source_block_permutations_are_deterministic() -> None:
    rows = []
    for source_index, source in enumerate("abcd"):
        for run_number, start in enumerate((10.0 + source_index, 30.0 + 2 * source_index), start=1):
            rows.append(
                {
                    "event_id": f"{source}:{run_number}", "source_key": source,
                    "ra": 0.1 * source_index, "dec": 0.0,
                    "run_start_mjd": start, "run_end_mjd": start + 0.2,
                    "dip_mjd": start + 0.1, "distance_pc": 100.0 * (source_index + 1),
                    "asassn_field_key": "F", "camera_name_key": "C",
                }
            )
    runs = pd.DataFrame(rows)

    pair_first = systematics.summarize_run_pair_excess(
        runs, angular_bins_deg=(0.5,), lag_thresholds_days=(1.0, 7.0),
        n_permutations=12, random_state=17,
    )
    pair_second = systematics.summarize_run_pair_excess(
        runs, angular_bins_deg=(0.5,), lag_thresholds_days=(1.0, 7.0),
        n_permutations=12, random_state=17,
    )
    pd.testing.assert_frame_equal(pair_first, pair_second)
    assert set(pair_first["permutation_unit"]) == {"source_run_schedule_block"}
    materialized = systematics.build_nearby_run_pairs(
        runs, max_sep_deg=0.5, max_interval_gap_days=7.0
    )
    expected_counts = {
        lag: int(materialized["time_lag_days"].le(lag).sum()) for lag in (1.0, 7.0)
    }
    assert pair_first.set_index("lag_max_days")["observed_pairs"].to_dict() == expected_counts
    explicit_pairs = systematics.build_nearby_run_pairs(
        runs, max_sep_deg=0.5, max_interval_gap_days=7.0
    )
    assert pair_first.loc[pair_first["lag_max_days"].eq(7.0), "observed_pairs"].iloc[0] == len(explicit_pairs)

    distance_first = systematics.distance_time_source_block_permutation_test(
        runs, strata=("asassn_field_key", "camera_name_key"),
        n_permutations=12, random_state=23,
    )
    distance_second = systematics.distance_time_source_block_permutation_test(
        runs, strata=("asassn_field_key", "camera_name_key"),
        n_permutations=12, random_state=23,
    )
    pd.testing.assert_frame_equal(distance_first, distance_second)
    assert distance_first.loc[0, "permutation_unit"] == "source_distance_block"
    expected_rho = runs["distance_pc"].rank().corr(runs["dip_mjd"].rank())
    assert distance_first.loc[0, "spearman_rho"] == pytest.approx(expected_rho)


def test_benjamini_hochberg_preserves_index_and_invalid_values() -> None:
    pvalues = pd.Series(
        [0.01, 0.04, 0.03, np.nan, 2.0, -0.1, 0.002],
        index=list("abcdefg"),
    )

    qvalues = systematics.benjamini_hochberg(pvalues)

    np.testing.assert_allclose(qvalues.loc[["a", "b", "c", "g"]], [0.02, 0.04, 0.04, 0.008])
    assert qvalues.loc[["d", "e", "f"]].isna().all()
    assert qvalues.index.equals(pvalues.index)


def test_nearby_pairs_handle_ra_wrap_and_preserve_group_labels() -> None:
    events = pd.DataFrame(
        {
            "source_key": ["west", "east", "far"],
            "ra": [359.9, 0.1, 5.0],
            "dec": [0.0, 0.0, 0.0],
            "dip_mjd": [100.0, 101.0, 110.0],
            "asassn_field_key": ["F", "F", "G"],
            "camera_name_key": ["C", "C", "D"],
        }
    )

    pairs = systematics.build_nearby_pairs(events, max_sep_deg=0.5)

    assert len(pairs) == 1
    pair = pairs.iloc[0]
    assert {pair["event_id_i"], pair["event_id_j"]} == {"west", "east"}
    assert pair["angular_sep_deg"] == pytest.approx(0.2)
    assert pair["time_lag_days"] == pytest.approx(1.0)
    assert bool(pair["same_field"])
    assert bool(pair["same_camera"])


def test_permutation_analyses_are_deterministic_and_degenerate_safe() -> None:
    events = pd.DataFrame(
        {
            "source_key": list("abcde"),
            "ra": [0.0, 0.1, 0.2, 0.3, 10.0],
            "dec": [0.0] * 5,
            "dip_mjd": [10.0, 10.2, 20.0, 20.2, 30.0],
            "distance_pc": [1.0, 2.0, 3.0, 5.0, 4.0],
            "asassn_field_key": ["F"] * 5,
            "camera_name_key": ["C"] * 5,
        }
    )
    pairs = systematics.build_nearby_pairs(events, max_sep_deg=0.5)

    pair_first = systematics.summarize_pair_excess(
        pairs,
        events,
        angular_bins_deg=(0.15, 0.5),
        lag_thresholds_days=(0.5, 2.0),
        n_permutations=24,
        random_state=123,
    )
    pair_second = systematics.summarize_pair_excess(
        pairs,
        events,
        angular_bins_deg=(0.15, 0.5),
        lag_thresholds_days=(0.5, 2.0),
        n_permutations=24,
        random_state=123,
    )
    pd.testing.assert_frame_equal(pair_first, pair_second)
    assert pair_first["pvalue"].between(1 / 25, 1.0).all()
    assert pair_first["qvalue"].between(0.0, 1.0).all()

    distance_first = systematics.distance_time_permutation_test(
        events, n_permutations=24, random_state=456
    )
    distance_second = systematics.distance_time_permutation_test(
        events, n_permutations=24, random_state=456
    )
    pd.testing.assert_frame_equal(distance_first, distance_second)
    assert np.isfinite(distance_first.loc[0, "spearman_rho"])
    assert 1 / 25 <= distance_first.loc[0, "permutation_pvalue"] <= 1.0

    constant_distance = events.assign(distance_pc=1.0)
    degenerate = systematics.distance_time_permutation_test(
        constant_distance, n_permutations=8, random_state=789
    )
    assert np.isnan(degenerate.loc[0, "spearman_rho"])
    assert np.isnan(degenerate.loc[0, "permutation_pvalue"])


def test_raw_attribution_has_stable_schema_and_distinct_source_exposure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    lightcurve_path = tmp_path / "source.dat3"
    lightcurve_path.write_text("synthetic")
    lightcurve = pd.DataFrame(
        {
            "jd": [2_450_099.9, 2_450_100.2, 2_450_102.0],
            "mjd": [50_099.4, 50_099.7, 50_101.5],
            "is_good": [True, True, True],
            "camera_name": ["cam-a", "cam-b", "cam-a"],
            "field": ["field-a", "field-b", "field-a"],
        }
    )
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_loader(path, **kwargs):
        calls.append((path, kwargs))
        return lightcurve.copy()

    monkeypatch.setattr(systematics, "load_lightcurve_df", fake_loader)
    events = pd.DataFrame(
        {
            "source_key": ["source"],
            "candidate_id": ["candidate"],
            "source_id": ["gaia"],
            "dip_jd": [2_450_100.0],
            "dip_mjd": [50_099.5],
            "dip_night_mjd": [50_099],
            "dip_night": ["1996-01-17"],
        }
    )

    attribution, exposures = systematics.scan_event_attribution_and_exposures(
        events,
        event_half_window_days=0.5,
        path_resolver=lambda _row: lightcurve_path,
    )

    assert calls == [(lightcurve_path, {"apply_quality": False})]
    assert {
        "source_key",
        "dip_jd",
        "camera_name",
        "field",
        "scan_status",
        "n_window_points",
        "n_good_window_points",
        "is_primary_event_group",
    } <= set(attribution.columns)
    assert len(attribution) == 2
    assert set(attribution["scan_status"]) == {"ok"}
    assert attribution["is_primary_event_group"].sum() == 1

    assert {
        "source_key",
        "night_mjd",
        "night",
        "camera_name",
        "field",
        "n_observations",
        "n_good_observations",
        "n_sources",
        "n_good_sources",
    } <= set(exposures.columns)
    assert set(exposures["source_key"]) == {"source"}
    assert set(exposures["n_sources"]) == {1}

    night_summary = systematics.summarize_nights(events, exposure_table=exposures)
    assert night_summary.loc[0, "n_exposed_sources"] == 1
    assert night_summary.loc[0, "exposure_source"] == "raw_lightcurve_scan"
