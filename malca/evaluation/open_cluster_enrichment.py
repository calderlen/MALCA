"""Matched case-control test of open-cluster membership among MALCA dippers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import tempfile

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from malca.io.table_io import is_layer_first_table, read_feature_table, read_parquet_table, write_parquet_table
from malca.products.feature_layers import expand_feature_layers


DEFAULT_OUTCOMES = (
    "ucc_good_member",
    "hr24_bound_member",
    "hr24_high_quality_member",
)


def _read_membership(path: Path) -> pd.DataFrame:
    if is_layer_first_table(path):
        return expand_feature_layers(read_feature_table(path))
    return read_parquet_table(path)


def load_review_labels(review_db: Path) -> pd.DataFrame:
    """Load one current Review label per candidate without modifying the DB."""
    query = """
        SELECT
            c.candidate_id,
            lower(trim(coalesce(r.event_class, ''))) AS event_class,
            lower(trim(coalesce(r.workflow_status, r.status, ''))) AS workflow_status
        FROM candidates AS c
        JOIN reviews AS r USING(candidate_id)
    """
    uri = f"file:{review_db.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        labels = pd.read_sql_query(query, connection)
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    if labels["candidate_id"].duplicated().any():
        raise ValueError("Review DB returned duplicate candidate labels")
    return labels


def _distance_pc(frame: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in ("distance_gspphot", "dist50", "distance_pc"):
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.gt(0) & np.isfinite(values)
        result.loc[result.isna() & valid] = values.loc[result.isna() & valid]
    if "parallax" in frame:
        parallax = pd.to_numeric(frame["parallax"], errors="coerce")
        valid = result.isna() & parallax.gt(0) & np.isfinite(parallax)
        result.loc[valid] = 1000.0 / parallax.loc[valid]
    return result


def _galactic_coordinates(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    longitude = pd.to_numeric(frame.get("gal_l"), errors="coerce") if "gal_l" in frame else pd.Series(np.nan, index=frame.index)
    latitude = pd.to_numeric(frame.get("gal_b"), errors="coerce") if "gal_b" in frame else pd.Series(np.nan, index=frame.index)
    missing = longitude.isna() | latitude.isna()
    if missing.any() and {"ra", "dec"}.issubset(frame.columns):
        ra = pd.to_numeric(frame["ra"], errors="coerce")
        dec = pd.to_numeric(frame["dec"], errors="coerce")
        usable = missing & ra.notna() & dec.notna()
        if usable.any():
            coords = SkyCoord(ra=ra.loc[usable].to_numpy() * u.deg, dec=dec.loc[usable].to_numpy() * u.deg)
            longitude.loc[usable] = coords.galactic.l.deg
            latitude.loc[usable] = coords.galactic.b.deg
    return longitude.astype(float), latitude.astype(float)


def prepare_case_control_population(
    membership: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    include_unclassified_controls: bool = False,
) -> pd.DataFrame:
    """Join membership to Review labels and construct matching covariates."""
    if "candidate_id" not in membership:
        raise ValueError("Membership table requires candidate_id")
    work = membership.copy()
    work["candidate_id"] = work["candidate_id"].astype(str)
    if work["candidate_id"].duplicated().any():
        raise ValueError("Membership table contains duplicate candidate_id values")
    work = work.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    work["is_case"] = work["event_class"].eq("dipper")
    excluded_controls = {"", "dipper"}
    if not include_unclassified_controls:
        excluded_controls.add("unclassified")
    work["is_control"] = ~work["event_class"].isin(excluded_controls)
    work["match_distance_pc"] = _distance_pc(work)
    work["match_g_mag"] = pd.to_numeric(work.get("phot_g_mean_mag"), errors="coerce")
    work["match_gal_l"], work["match_gal_b"] = _galactic_coordinates(work)
    gaia_id = work.get("open_cluster_gaia_id", pd.Series(pd.NA, index=work.index))
    work["match_eligible"] = (
        gaia_id.notna()
        & work["match_distance_pc"].gt(0)
        & work["match_g_mag"].notna()
        & work["match_gal_l"].notna()
        & work["match_gal_b"].notna()
    )
    return work


def _angular_separation_deg(
    case_l: float,
    case_b: float,
    control_l: np.ndarray,
    control_b: np.ndarray,
) -> np.ndarray:
    l1 = np.deg2rad(float(case_l))
    b1 = np.deg2rad(float(case_b))
    l2 = np.deg2rad(control_l.astype(float))
    b2 = np.deg2rad(control_b.astype(float))
    cosine = np.sin(b1) * np.sin(b2) + np.cos(b1) * np.cos(b2) * np.cos(l1 - l2)
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))


def match_case_controls(
    population: pd.DataFrame,
    *,
    controls_per_case: int = 4,
    min_controls: int = 1,
    sky_caliper_deg: float = 15.0,
    latitude_caliper_deg: float = 5.0,
    g_caliper_mag: float = 1.0,
    fractional_distance_caliper: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedily match the most constrained cases first without control reuse."""
    if controls_per_case < 1 or min_controls < 1 or min_controls > controls_per_case:
        raise ValueError("Require controls_per_case >= min_controls >= 1")
    cases = population.loc[population["is_case"] & population["match_eligible"]].copy()
    controls = population.loc[population["is_control"] & population["match_eligible"]].copy()
    if cases.empty:
        raise ValueError("No matching-eligible dipper cases")
    if controls.empty:
        raise ValueError("No matching-eligible controls")

    control_l = controls["match_gal_l"].to_numpy(dtype=float)
    control_b = controls["match_gal_b"].to_numpy(dtype=float)
    control_g = controls["match_g_mag"].to_numpy(dtype=float)
    control_logd = np.log(controls["match_distance_pc"].to_numpy(dtype=float))
    control_indices = controls.index.to_numpy()
    distance_log_caliper = np.log1p(float(fractional_distance_caliper))
    candidate_cache: dict[object, tuple[np.ndarray, np.ndarray]] = {}

    for case_index, case in cases.iterrows():
        sky_sep = _angular_separation_deg(
            float(case["match_gal_l"]),
            float(case["match_gal_b"]),
            control_l,
            control_b,
        )
        delta_b = np.abs(control_b - float(case["match_gal_b"]))
        delta_g = np.abs(control_g - float(case["match_g_mag"]))
        delta_logd = np.abs(control_logd - np.log(float(case["match_distance_pc"])))
        eligible = (
            (sky_sep <= float(sky_caliper_deg))
            & (delta_b <= float(latitude_caliper_deg))
            & (delta_g <= float(g_caliper_mag))
            & (delta_logd <= distance_log_caliper)
        )
        cost = (
            sky_sep / float(sky_caliper_deg)
            + delta_b / float(latitude_caliper_deg)
            + delta_g / float(g_caliper_mag)
            + delta_logd / distance_log_caliper
        )
        order = np.argsort(cost, kind="stable")
        candidate_cache[case_index] = (control_indices[order[eligible[order]]], cost[order[eligible[order]]])

    case_order = sorted(candidate_cache, key=lambda index: (len(candidate_cache[index][0]), str(index)))
    used_controls: set[object] = set()
    matched_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    match_set_number = 0
    for case_index in case_order:
        candidate_indices, candidate_costs = candidate_cache[case_index]
        available = [
            (index, float(cost))
            for index, cost in zip(candidate_indices, candidate_costs, strict=True)
            if index not in used_controls
        ]
        selected = available[:controls_per_case]
        case = population.loc[case_index]
        if len(selected) < min_controls:
            audit_rows.append(
                {
                    "candidate_id": str(case["candidate_id"]),
                    "status": "insufficient_controls",
                    "eligible_controls": int(len(candidate_indices)),
                    "available_controls": int(len(available)),
                }
            )
            continue
        match_set_number += 1
        match_set_id = f"match_{match_set_number:04d}"
        case_record = case.to_dict()
        case_record.update(match_set_id=match_set_id, match_role="case", match_rank=0, match_cost=0.0)
        matched_rows.append(case_record)
        for rank, (control_index, cost) in enumerate(selected, start=1):
            used_controls.add(control_index)
            control_record = population.loc[control_index].to_dict()
            control_record.update(
                match_set_id=match_set_id,
                match_role="control",
                match_rank=int(rank),
                match_cost=float(cost),
            )
            matched_rows.append(control_record)
        audit_rows.append(
            {
                "candidate_id": str(case["candidate_id"]),
                "status": "matched",
                "eligible_controls": int(len(candidate_indices)),
                "available_controls": int(len(available)),
                "selected_controls": int(len(selected)),
                "match_set_id": match_set_id,
            }
        )
    matched = pd.DataFrame(matched_rows)
    audit = pd.DataFrame(audit_rows)
    if matched.empty:
        raise RuntimeError("No case-control sets satisfied the matching constraints")
    return matched, audit


def _mh_odds_ratio(strata: pd.DataFrame, outcome: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for _, group in strata.groupby("match_set_id", sort=False):
        case = group.loc[group["match_role"].eq("case"), outcome].astype(bool)
        controls = group.loc[group["match_role"].eq("control"), outcome].astype(bool)
        if len(case) != 1 or controls.empty:
            continue
        a = float(case.iloc[0])
        b = 1.0 - a
        c = float(controls.sum())
        d = float(len(controls) - controls.sum())
        total = 1.0 + float(len(controls))
        numerator += a * d / total
        denominator += b * c / total
    if denominator == 0:
        return np.inf if numerator > 0 else np.nan
    return float(numerator / denominator)


def _matched_difference(strata: pd.DataFrame, outcome: str) -> float:
    differences: list[float] = []
    for _, group in strata.groupby("match_set_id", sort=False):
        case = group.loc[group["match_role"].eq("case"), outcome].astype(bool)
        controls = group.loc[group["match_role"].eq("control"), outcome].astype(bool)
        if len(case) == 1 and not controls.empty:
            differences.append(float(case.iloc[0]) - float(controls.mean()))
    return float(np.mean(differences)) if differences else np.nan


def matched_outcome_statistics(
    matched: pd.DataFrame,
    outcome: str,
    *,
    seed: int = 20260808,
    bootstrap_draws: int = 2000,
    permutation_draws: int = 10000,
) -> dict[str, object]:
    """Compute stratified OR, matched bootstrap CI, and within-set permutation p."""
    if outcome not in matched:
        raise KeyError(outcome)
    work = matched.copy()
    work[outcome] = work[outcome].fillna(False).astype(bool)
    sets = list(work["match_set_id"].drop_duplicates())
    odds_ratio = _mh_odds_ratio(work, outcome)
    observed_difference = _matched_difference(work, outcome)
    rng = np.random.default_rng(seed)
    groups = [group.copy() for _, group in work.groupby("match_set_id", sort=False)]

    bootstrap_values: list[float] = []
    if bootstrap_draws > 0 and sets:
        numerator_components = []
        denominator_components = []
        for group in groups:
            case = group.loc[group["match_role"].eq("case"), outcome].to_numpy(dtype=bool)
            controls = group.loc[group["match_role"].eq("control"), outcome].to_numpy(dtype=bool)
            if len(case) != 1 or controls.size == 0:
                numerator_components.append(0.0)
                denominator_components.append(0.0)
                continue
            a = float(case[0])
            c = float(controls.sum())
            total = 1.0 + float(controls.size)
            numerator_components.append(a * (float(controls.size) - c) / total)
            denominator_components.append((1.0 - a) * c / total)
        numerator_array = np.asarray(numerator_components, dtype=float)
        denominator_array = np.asarray(denominator_components, dtype=float)
        sampled_indices = rng.integers(
            0,
            len(groups),
            size=(int(bootstrap_draws), len(groups)),
        )
        sampled_numerator = numerator_array[sampled_indices].sum(axis=1)
        sampled_denominator = denominator_array[sampled_indices].sum(axis=1)
        finite = sampled_denominator > 0
        values = sampled_numerator[finite] / sampled_denominator[finite]
        bootstrap_values = values[np.isfinite(values) & (values > 0)].astype(float).tolist()
    if bootstrap_values:
        ci_low, ci_high = np.quantile(bootstrap_values, [0.025, 0.975])
    else:
        ci_low, ci_high = np.nan, np.nan

    extreme = 0
    valid_permutations = 0
    if permutation_draws > 0 and groups:
        permuted_difference_sum = np.zeros(int(permutation_draws), dtype=float)
        contributing_groups = 0
        for group in groups:
            values = group[outcome].to_numpy(dtype=bool).astype(float)
            if values.size < 2:
                continue
            selected_indices = rng.integers(0, values.size, size=int(permutation_draws))
            selected_values = values[selected_indices]
            control_means = (values.sum() - selected_values) / float(values.size - 1)
            permuted_difference_sum += selected_values - control_means
            contributing_groups += 1
        if contributing_groups:
            permuted = permuted_difference_sum / float(contributing_groups)
            extreme = int(
                np.count_nonzero(np.abs(permuted) >= abs(observed_difference) - 1e-15)
            )
            valid_permutations = int(permutation_draws)
    permutation_p = (
        float((extreme + 1) / (valid_permutations + 1)) if valid_permutations else np.nan
    )

    cases = work["match_role"].eq("case")
    controls = work["match_role"].eq("control")
    case_positive = int(work.loc[cases, outcome].sum())
    case_negative = int(cases.sum() - case_positive)
    control_positive = int(work.loc[controls, outcome].sum())
    control_negative = int(controls.sum() - control_positive)
    fisher = fisher_exact(
        [[case_positive, case_negative], [control_positive, control_negative]],
        alternative="two-sided",
    )
    return {
        "outcome": outcome,
        "match_sets": int(len(sets)),
        "cases": int(cases.sum()),
        "controls": int(controls.sum()),
        "case_positive": case_positive,
        "control_positive": control_positive,
        "case_rate": float(case_positive / cases.sum()) if cases.sum() else np.nan,
        "control_rate": float(control_positive / controls.sum()) if controls.sum() else np.nan,
        "matched_rate_difference": observed_difference,
        "mantel_haenszel_odds_ratio": odds_ratio,
        "bootstrap_ci95_low": float(ci_low),
        "bootstrap_ci95_high": float(ci_high),
        "bootstrap_finite_draws": int(len(bootstrap_values)),
        "matched_permutation_p": permutation_p,
        "matched_permutation_draws": int(valid_permutations),
        "unadjusted_fisher_odds_ratio": float(fisher.statistic),
        "unadjusted_fisher_p": float(fisher.pvalue),
    }


def evaluate_matched_outcomes(
    matched: pd.DataFrame,
    *,
    outcomes: tuple[str, ...] = DEFAULT_OUTCOMES,
    seed: int = 20260808,
    bootstrap_draws: int = 2000,
    permutation_draws: int = 10000,
) -> pd.DataFrame:
    rows = []
    for offset, outcome in enumerate(outcomes):
        if outcome not in matched:
            continue
        if outcome.startswith("hr24_") and "hr24_match_status" in matched:
            if matched["hr24_match_status"].eq("not_run").all():
                continue
        rows.append(
            matched_outcome_statistics(
                matched,
                outcome,
                seed=int(seed + offset),
                bootstrap_draws=bootstrap_draws,
                permutation_draws=permutation_draws,
            )
        )
    return pd.DataFrame(rows)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a matched case-control test of open-cluster membership among reviewed dippers."
    )
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--review-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controls-per-case", type=int, default=4)
    parser.add_argument("--min-controls", type=int, default=1)
    parser.add_argument("--sky-caliper-deg", type=float, default=15.0)
    parser.add_argument("--latitude-caliper-deg", type=float, default=5.0)
    parser.add_argument("--g-caliper-mag", type=float, default=1.0)
    parser.add_argument("--fractional-distance-caliper", type=float, default=0.30)
    parser.add_argument("--include-unclassified-controls", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--permutation-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args(argv)

    membership_path = args.membership.expanduser()
    review_db = args.review_db.expanduser()
    output_dir = args.output_dir.expanduser()
    if not membership_path.exists():
        raise FileNotFoundError(membership_path)
    if not review_db.exists():
        raise FileNotFoundError(review_db)

    membership = _read_membership(membership_path)
    labels = load_review_labels(review_db)
    population = prepare_case_control_population(
        membership,
        labels,
        include_unclassified_controls=bool(args.include_unclassified_controls),
    )
    matched, audit = match_case_controls(
        population,
        controls_per_case=args.controls_per_case,
        min_controls=args.min_controls,
        sky_caliper_deg=args.sky_caliper_deg,
        latitude_caliper_deg=args.latitude_caliper_deg,
        g_caliper_mag=args.g_caliper_mag,
        fractional_distance_caliper=args.fractional_distance_caliper,
    )
    statistics = evaluate_matched_outcomes(
        matched,
        seed=args.seed,
        bootstrap_draws=args.bootstrap_draws,
        permutation_draws=args.permutation_draws,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    matched_path = output_dir / "matched_control_sets.parquet"
    audit_path = output_dir / "matching_audit.parquet"
    statistics_path = output_dir / "membership_enrichment_statistics.csv"
    summary_path = output_dir / "membership_enrichment_summary.json"
    write_parquet_table(matched, matched_path)
    write_parquet_table(audit, audit_path)
    statistics.to_csv(statistics_path, index=False)
    summary = {
        "membership": str(membership_path.resolve()),
        "review_db": str(review_db.resolve()),
        "population_rows": int(len(population)),
        "reviewed_dipper_rows": int(population["is_case"].sum()),
        "eligible_dipper_rows": int((population["is_case"] & population["match_eligible"]).sum()),
        "eligible_control_rows": int((population["is_control"] & population["match_eligible"]).sum()),
        "matched_case_rows": int(matched["match_role"].eq("case").sum()),
        "matched_control_rows": int(matched["match_role"].eq("control").sum()),
        "matching": {
            "controls_per_case": args.controls_per_case,
            "min_controls": args.min_controls,
            "sky_caliper_deg": args.sky_caliper_deg,
            "latitude_caliper_deg": args.latitude_caliper_deg,
            "g_caliper_mag": args.g_caliper_mag,
            "fractional_distance_caliper": args.fractional_distance_caliper,
            "controls_reused": False,
        },
        "inference": {
            "primary": "Mantel-Haenszel odds ratio across matched sets",
            "ci": "matched-set nonparametric bootstrap",
            "p_value": "within-set case-label permutation",
            "unadjusted_secondary": "Fisher exact test",
            "bootstrap_draws": args.bootstrap_draws,
            "permutation_draws": args.permutation_draws,
            "seed": args.seed,
        },
        "statistics": statistics.replace({np.nan: None, np.inf: "inf", -np.inf: "-inf"}).to_dict("records"),
    }
    _write_json_atomic(summary_path, summary)
    print(f"Matched cases: {summary['matched_case_rows']}; controls: {summary['matched_control_rows']}")
    print(statistics.to_string(index=False))
    print(f"Matched sets: {matched_path}")
    print(f"Statistics: {statistics_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
