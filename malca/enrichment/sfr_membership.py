"""Association-membership evidence for distance-resolved SFR environments.

This module deliberately keeps two questions separate:

``sfr_environment_consistent``
    The source overlaps a molecular-cloud footprint and distance slab.  That
    value is supplied by the SFR footprint code; it is not stellar membership.

``sfr_membership_class``
    The source is an exact member-catalog match or is kinematically consistent
    with a stellar population explicitly mapped to the same SFR.

BANYAN's global young-association probability is never used as an SFR
probability.  Component probabilities are aggregated only through the curated
crosswalk in ``malca/data/sfr_association_crosswalk.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import chi2


SFR_MEMBERSHIP_VERSION = "1"
DEFAULT_BANYAN_SFR_THRESHOLD = 0.90
DEFAULT_KINEMATIC_CONFIDENCE = 0.9973
DEFAULT_MIN_KINEMATIC_MEMBERS = 8

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_CROSSWALK_PATH = DATA_DIR / "sfr_association_crosswalk.csv"
DEFAULT_MEMBER_CATALOG_PATH = DATA_DIR / "sfr_catalog_members.csv"

CROSSWALK_COLUMNS = (
    "sfr_name",
    "banyan_assoc",
    "relation",
    "include_in_sfr_probability",
    "source",
    "notes",
)
MEMBER_CATALOG_REQUIRED_COLUMNS = (
    "gaia_id",
    "association_name",
    "sfr_name",
    "catalog_name",
    "catalog_reference",
)
MEMBER_CATALOG_OPTIONAL_COLUMNS = (
    "subcluster",
    "catalog_membership_prob",
    "catalog_quality",
    "accepted_member",
    "parallax",
    "parallax_error",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "parallax_pmra_corr",
    "parallax_pmdec_corr",
    "pmra_pmdec_corr",
)

SFR_MEMBERSHIP_OUTPUT_COLUMNS = (
    "sfr_environment_matches",
    "sfr_environment_consistent",
    "banyan_sfr_name",
    "banyan_sfr_prob",
    "banyan_sfr_best_assoc",
    "banyan_sfr_best_assoc_prob",
    "banyan_sfr_agrees",
    "sfr_catalog_member",
    "sfr_catalog_match_status",
    "sfr_catalog_name",
    "sfr_catalog_reference",
    "sfr_catalog_membership_prob",
    "sfr_kinematic_name",
    "sfr_kinematic_method",
    "sfr_kinematic_consistent",
    "sfr_kinematic_mahalanobis_sq",
    "sfr_kinematic_p_value",
    "sfr_kinematic_n_members",
    "sfr_membership_class",
    "sfr_membership_name",
    "sfr_membership_evidence",
    "sfr_membership_status",
    "sfr_membership_threshold",
    "sfr_membership_version",
)


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def _normalize_gaia_id(value: object) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    try:
        numeric = Decimal(text)
    except InvalidOperation:
        return text
    if numeric.is_finite() and numeric == numeric.to_integral_value():
        return format(numeric.quantize(Decimal(1)), "f")
    return text


def _coerce_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        if bool(pd.isna(value)):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot interpret boolean value {value!r}")


def _finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def split_sfr_matches(value: object) -> tuple[str, ...]:
    """Return unique semicolon-delimited SFR names in their original order."""
    text = _clean_text(value)
    if not text:
        return ()
    return tuple(dict.fromkeys(part.strip() for part in text.split(";") if part.strip()))


def load_sfr_association_crosswalk(
    path: str | Path = DEFAULT_CROSSWALK_PATH,
) -> pd.DataFrame:
    """Load and validate the curated BANYAN-population to SFR crosswalk."""
    frame = pd.read_csv(Path(path))
    missing = [column for column in CROSSWALK_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"SFR association crosswalk is missing columns: {missing}")

    out = frame.loc[:, CROSSWALK_COLUMNS].copy()
    for column in ("sfr_name", "banyan_assoc", "relation", "source", "notes"):
        out[column] = out[column].fillna("").astype(str).str.strip()
    out["banyan_assoc"] = out["banyan_assoc"].str.upper()
    out["include_in_sfr_probability"] = out["include_in_sfr_probability"].map(
        _coerce_bool
    )

    required_nonempty = ("sfr_name", "banyan_assoc", "relation", "source")
    for column in required_nonempty:
        if out[column].eq("").any():
            rows = out.index[out[column].eq("")].tolist()
            raise ValueError(f"Crosswalk column {column!r} is blank at rows {rows}")

    included = out[out["include_in_sfr_probability"]]
    duplicate_pairs = included.duplicated(["sfr_name", "banyan_assoc"], keep=False)
    if duplicate_pairs.any():
        pairs = included.loc[duplicate_pairs, ["sfr_name", "banyan_assoc"]].to_dict(
            "records"
        )
        raise ValueError(f"Duplicate included SFR/BANYAN mappings: {pairs}")

    ambiguous = (
        included.groupby("banyan_assoc", sort=False)["sfr_name"].nunique().loc[lambda x: x > 1]
    )
    if not ambiguous.empty:
        raise ValueError(
            "Included BANYAN populations map to multiple SFRs: "
            + ", ".join(ambiguous.index.astype(str))
        )
    return out


def load_sfr_catalog_members(
    path: str | Path = DEFAULT_MEMBER_CATALOG_PATH,
) -> pd.DataFrame:
    """Load a normalized, provenance-bearing association member catalog."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported SFR member-catalog format: {path}")

    missing = [
        column for column in MEMBER_CATALOG_REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"SFR member catalog is missing columns: {missing}")

    out = frame.copy()
    for column in MEMBER_CATALOG_OPTIONAL_COLUMNS:
        if column not in out.columns:
            out[column] = True if column == "accepted_member" else np.nan
    out["gaia_id"] = out["gaia_id"].map(_normalize_gaia_id)
    for column in (
        "association_name",
        "sfr_name",
        "catalog_name",
        "catalog_reference",
        "subcluster",
        "catalog_quality",
    ):
        out[column] = out[column].fillna("").astype(str).str.strip()
    out["accepted_member"] = out["accepted_member"].map(
        lambda value: _coerce_bool(value, default=True)
    )
    for column in (
        "catalog_membership_prob",
        "parallax",
        "parallax_error",
        "pmra",
        "pmra_error",
        "pmdec",
        "pmdec_error",
        "parallax_pmra_corr",
        "parallax_pmdec_corr",
        "pmra_pmdec_corr",
    ):
        out[column] = pd.to_numeric(out[column], errors="coerce")

    invalid = out["gaia_id"].eq("") | out["association_name"].eq("") | out["sfr_name"].eq("")
    if invalid.any():
        rows = out.index[invalid].tolist()
        raise ValueError(f"SFR member catalog has incomplete identity fields at rows {rows}")
    probabilities = out["catalog_membership_prob"].dropna()
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("catalog_membership_prob values must lie in [0, 1]")
    if out.duplicated(
        ["gaia_id", "association_name", "sfr_name", "catalog_name"], keep=False
    ).any():
        raise ValueError("SFR member catalog contains duplicate membership rows")
    return out


def parse_banyan_probability_map(raw: object) -> dict[str, float]:
    """Return validated association probabilities from stored BANYAN JSON."""
    if isinstance(raw, Mapping):
        payload = dict(raw)
    else:
        text = _clean_text(raw)
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid BANYAN probability JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("BANYAN probability payload must be a JSON object")

    probabilities: dict[str, float] = {}
    for name, value in payload.items():
        probability = _finite_float(value)
        if not math.isfinite(probability):
            continue
        if probability < -1e-12 or probability > 1.0 + 1e-12:
            raise ValueError(f"Invalid BANYAN probability for {name!r}: {probability}")
        probabilities[str(name).strip().upper()] = float(np.clip(probability, 0.0, 1.0))
    return probabilities


def aggregate_banyan_sfr_probabilities(
    raw_probabilities: object,
    crosswalk: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Aggregate mutually exclusive BANYAN hypotheses into curated SFR groups."""
    probabilities = parse_banyan_probability_map(raw_probabilities)
    included = crosswalk[crosswalk["include_in_sfr_probability"]]
    aggregated: dict[str, dict[str, object]] = {}
    for sfr_name, mappings in included.groupby("sfr_name", sort=False):
        components = {
            association: probabilities.get(association, 0.0)
            for association in mappings["banyan_assoc"].astype(str)
        }
        total = float(sum(components.values()))
        if total > 1.0 + 1e-8:
            raise ValueError(
                f"Mapped BANYAN probabilities for {sfr_name!r} sum to {total:.6f}"
            )
        best_assoc = max(components, key=components.get) if components else ""
        best_prob = components.get(best_assoc, math.nan) if best_assoc else math.nan
        aggregated[str(sfr_name)] = {
            "probability": float(np.clip(total, 0.0, 1.0)),
            "best_assoc": best_assoc if best_prob > 0.0 else "",
            "best_assoc_probability": best_prob if best_prob > 0.0 else math.nan,
            "components": components,
        }
    return aggregated


@dataclass(frozen=True)
class KinematicModel:
    sfr_name: str
    association_name: str
    subcluster: str
    center: np.ndarray
    covariance: np.ndarray
    n_members: int


def fit_catalog_kinematic_models(
    catalog_members: pd.DataFrame,
    *,
    min_members: int = DEFAULT_MIN_KINEMATIC_MEMBERS,
) -> tuple[KinematicModel, ...]:
    """Fit robust empirical (parallax, pmra, pmdec) models to member groups."""
    if min_members < 4:
        raise ValueError("min_members must be at least 4 for a 3D covariance")
    if catalog_members.empty:
        return ()

    models: list[KinematicModel] = []
    accepted = catalog_members[catalog_members["accepted_member"]].copy()
    group_columns = ["sfr_name", "association_name", "subcluster"]
    for keys, group in accepted.groupby(group_columns, dropna=False, sort=False):
        values = group[["parallax", "pmra", "pmdec"]].to_numpy(dtype=float)
        values = values[np.isfinite(values).all(axis=1)]
        if len(values) < min_members:
            continue

        center = np.nanmedian(values, axis=0)
        absolute_deviation = np.abs(values - center)
        mad = np.nanmedian(absolute_deviation, axis=0)
        scale = np.where(mad > 0.0, 1.4826 * mad, np.nanstd(values, axis=0))
        scale = np.where(scale > 0.0, scale, 1e-6)
        retained = np.all(absolute_deviation <= 5.0 * scale, axis=1)
        values = values[retained]
        if len(values) < min_members:
            continue

        center = np.nanmedian(values, axis=0)
        covariance = np.cov(values - center, rowvar=False, ddof=1)
        if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
            continue
        diagonal_floor = np.maximum(np.diag(covariance) * 1e-6, 1e-10)
        covariance = covariance + np.diag(diagonal_floor)
        models.append(
            KinematicModel(
                sfr_name=str(keys[0]),
                association_name=str(keys[1]),
                subcluster=_clean_text(keys[2]),
                center=center,
                covariance=covariance,
                n_members=int(len(values)),
            )
        )
    return tuple(models)


def _candidate_astrometric_covariance(row: pd.Series) -> np.ndarray | None:
    errors = np.array(
        [
            _finite_float(row.get("parallax_error")),
            _finite_float(row.get("pmra_error")),
            _finite_float(row.get("pmdec_error")),
        ],
        dtype=float,
    )
    if not np.isfinite(errors).all() or np.any(errors <= 0.0):
        return None
    covariance = np.diag(errors**2)
    correlations = (
        (0, 1, "parallax_pmra_corr"),
        (0, 2, "parallax_pmdec_corr"),
        (1, 2, "pmra_pmdec_corr"),
    )
    for left, right, column in correlations:
        correlation = _finite_float(row.get(column))
        if not math.isfinite(correlation):
            correlation = 0.0
        if correlation < -1.0 or correlation > 1.0:
            return None
        covariance[left, right] = covariance[right, left] = (
            correlation * errors[left] * errors[right]
        )
    return covariance


def catalog_kinematic_consistency(
    row: pd.Series,
    models: tuple[KinematicModel, ...],
    *,
    confidence: float = DEFAULT_KINEMATIC_CONFIDENCE,
) -> dict[str, object] | None:
    """Return the most consistent member-catalog model for one candidate."""
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie between 0.5 and 1")
    values = np.array(
        [
            _finite_float(row.get("parallax")),
            _finite_float(row.get("pmra")),
            _finite_float(row.get("pmdec")),
        ],
        dtype=float,
    )
    measurement_covariance = _candidate_astrometric_covariance(row)
    if not np.isfinite(values).all() or measurement_covariance is None or not models:
        return None

    best: dict[str, object] | None = None
    threshold = float(chi2.ppf(confidence, df=3))
    for model in models:
        covariance = model.covariance + measurement_covariance
        try:
            precision = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            continue
        delta = values - model.center
        distance_sq = float(delta @ precision @ delta)
        if not math.isfinite(distance_sq):
            continue
        p_value = float(chi2.sf(distance_sq, df=3))
        candidate = {
            "sfr_name": model.sfr_name,
            "association_name": model.association_name,
            "subcluster": model.subcluster,
            "mahalanobis_sq": distance_sq,
            "p_value": p_value,
            "consistent": bool(distance_sq <= threshold),
            "n_members": model.n_members,
        }
        if best is None or distance_sq < float(best["mahalanobis_sq"]):
            best = candidate
    return best


def _catalog_index(catalog_members: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if catalog_members.empty:
        return {}
    return {
        str(gaia_id): group.copy()
        for gaia_id, group in catalog_members.groupby("gaia_id", sort=False)
    }


def _select_catalog_match(
    matches: pd.DataFrame | None,
    environment_matches: tuple[str, ...],
) -> pd.Series | None:
    if matches is None or matches.empty:
        return None
    accepted = matches[matches["accepted_member"]].copy()
    if accepted.empty:
        return None
    accepted["_environment_agreement"] = accepted["sfr_name"].isin(environment_matches)
    accepted["_probability_sort"] = accepted["catalog_membership_prob"].fillna(-1.0)
    accepted = accepted.sort_values(
        ["_environment_agreement", "_probability_sort"],
        ascending=[False, False],
        kind="stable",
    )
    return accepted.iloc[0]


def append_sfr_membership_evidence(
    candidates: pd.DataFrame,
    *,
    crosswalk: pd.DataFrame | None = None,
    catalog_members: pd.DataFrame | None = None,
    environment_column: str = "sfr_matches",
    banyan_threshold: float = DEFAULT_BANYAN_SFR_THRESHOLD,
    kinematic_confidence: float = DEFAULT_KINEMATIC_CONFIDENCE,
    min_kinematic_members: int = DEFAULT_MIN_KINEMATIC_MEMBERS,
    require_banyan_parallax: bool = True,
) -> pd.DataFrame:
    """Append separate environment, catalog, and kinematic membership evidence."""
    if not 0.0 < banyan_threshold < 1.0:
        raise ValueError("banyan_threshold must lie between 0 and 1")
    crosswalk = (
        load_sfr_association_crosswalk()
        if crosswalk is None
        else load_sfr_association_crosswalk_from_frame(crosswalk)
    )
    catalog_members = (
        load_sfr_catalog_members()
        if catalog_members is None
        else load_sfr_catalog_members_from_frame(catalog_members)
    )
    models = fit_catalog_kinematic_models(
        catalog_members,
        min_members=min_kinematic_members,
    )
    members_by_id = _catalog_index(catalog_members)
    mapped_sfr_names = set(
        crosswalk.loc[crosswalk["include_in_sfr_probability"], "sfr_name"].astype(str)
    )
    catalog_available = not catalog_members.empty

    output_rows: list[dict[str, object]] = []
    for _, row in candidates.iterrows():
        environment_matches = split_sfr_matches(row.get(environment_column))
        environment_text = ";".join(environment_matches)
        gaia_id = _normalize_gaia_id(row.get("gaia_id"))
        if not gaia_id:
            gaia_id = _normalize_gaia_id(row.get("source_id"))
        catalog_rows = members_by_id.get(gaia_id)
        catalog_match = _select_catalog_match(catalog_rows, environment_matches)

        if catalog_match is not None:
            catalog_member: object = True
            catalog_status = "exact_gaia_id_match"
            catalog_sfr = str(catalog_match["sfr_name"])
            catalog_name = str(catalog_match["catalog_name"])
            catalog_reference = str(catalog_match["catalog_reference"])
            catalog_probability = _finite_float(
                catalog_match.get("catalog_membership_prob")
            )
        else:
            catalog_member = None
            catalog_sfr = ""
            catalog_name = ""
            catalog_reference = ""
            catalog_probability = math.nan
            if not gaia_id:
                catalog_status = "missing_gaia_id"
            elif catalog_rows is not None and not catalog_rows.empty:
                catalog_status = "listed_not_accepted"
            elif catalog_available:
                catalog_status = "not_listed"
            else:
                catalog_status = "catalog_unavailable"

        banyan_status = _clean_text(row.get("banyan_status"))
        banyan_input_mode = _clean_text(row.get("banyan_input_mode"))
        banyan_error = ""
        try:
            aggregated = aggregate_banyan_sfr_probabilities(
                row.get("banyan_probabilities_json"),
                crosswalk,
            )
        except ValueError as exc:
            aggregated = {}
            banyan_error = str(exc)

        mapped_ranked = sorted(
            aggregated.items(),
            key=lambda item: float(item[1]["probability"]),
            reverse=True,
        )
        if mapped_ranked and float(mapped_ranked[0][1]["probability"]) > 0.0:
            banyan_sfr_name, banyan_details = mapped_ranked[0]
            banyan_sfr_prob = float(banyan_details["probability"])
            banyan_best_assoc = str(banyan_details["best_assoc"])
            banyan_best_assoc_prob = _finite_float(
                banyan_details["best_assoc_probability"]
            )
        else:
            banyan_sfr_name = ""
            banyan_sfr_prob = 0.0 if aggregated and banyan_status == "ok" else math.nan
            banyan_best_assoc = ""
            banyan_best_assoc_prob = math.nan

        banyan_eligible = banyan_status == "ok" and not banyan_error
        if require_banyan_parallax:
            banyan_eligible = banyan_eligible and "plx" in banyan_input_mode
        strong_banyan = bool(
            banyan_eligible
            and banyan_sfr_name
            and math.isfinite(banyan_sfr_prob)
            and banyan_sfr_prob >= banyan_threshold
        )
        if not banyan_eligible or not environment_matches:
            banyan_agrees: object = None
        else:
            banyan_agrees = bool(
                strong_banyan and banyan_sfr_name in environment_matches
            )

        local_kinematics = catalog_kinematic_consistency(
            row,
            models,
            confidence=kinematic_confidence,
        )
        local_consistent = bool(
            local_kinematics is not None and local_kinematics["consistent"]
        )

        if strong_banyan:
            kinematic_name = banyan_sfr_name
            kinematic_method = "banyan_mapped_sfr"
            kinematic_consistent: object = True
            kinematic_distance_sq = math.nan
            kinematic_p_value = math.nan
            kinematic_n_members = math.nan
        elif local_kinematics is not None:
            kinematic_name = str(local_kinematics["sfr_name"])
            kinematic_method = "catalog_mahalanobis"
            kinematic_consistent = bool(local_kinematics["consistent"])
            kinematic_distance_sq = float(local_kinematics["mahalanobis_sq"])
            kinematic_p_value = float(local_kinematics["p_value"])
            kinematic_n_members = int(local_kinematics["n_members"])
        else:
            kinematic_name = ""
            kinematic_method = ""
            kinematic_consistent = None
            kinematic_distance_sq = math.nan
            kinematic_p_value = math.nan
            kinematic_n_members = math.nan

        if catalog_match is not None:
            membership_name = catalog_sfr
            membership_evidence = "catalog"
            if catalog_sfr in environment_matches:
                membership_class = "catalog_confirmed_member"
            else:
                membership_class = "dispersed_association_member"
            membership_status = "ok"
        elif strong_banyan or local_consistent:
            membership_name = kinematic_name
            membership_evidence = kinematic_method
            if membership_name in environment_matches:
                membership_class = "kinematically_consistent_member"
            else:
                membership_class = "dispersed_association_member"
            membership_status = "ok"
        elif environment_matches:
            membership_name = environment_text
            membership_evidence = "dust_distance"
            membership_class = "environmental_candidate"
            membership_status = (
                "weak_kinematics"
                if any(name in mapped_sfr_names for name in environment_matches)
                else "unmapped_environment"
            )
        elif banyan_error:
            membership_name = ""
            membership_evidence = ""
            membership_class = "unknown"
            membership_status = "invalid_banyan_probabilities"
        elif banyan_status and banyan_status != "ok":
            membership_name = ""
            membership_evidence = ""
            membership_class = "unknown"
            membership_status = banyan_status
        else:
            membership_name = ""
            membership_evidence = ""
            membership_class = "neither"
            membership_status = "ok"

        output_rows.append(
            {
                "sfr_environment_matches": environment_text,
                "sfr_environment_consistent": bool(environment_matches),
                "banyan_sfr_name": banyan_sfr_name,
                "banyan_sfr_prob": banyan_sfr_prob,
                "banyan_sfr_best_assoc": banyan_best_assoc,
                "banyan_sfr_best_assoc_prob": banyan_best_assoc_prob,
                "banyan_sfr_agrees": banyan_agrees,
                "sfr_catalog_member": catalog_member,
                "sfr_catalog_match_status": catalog_status,
                "sfr_catalog_name": catalog_name,
                "sfr_catalog_reference": catalog_reference,
                "sfr_catalog_membership_prob": catalog_probability,
                "sfr_kinematic_name": kinematic_name,
                "sfr_kinematic_method": kinematic_method,
                "sfr_kinematic_consistent": kinematic_consistent,
                "sfr_kinematic_mahalanobis_sq": kinematic_distance_sq,
                "sfr_kinematic_p_value": kinematic_p_value,
                "sfr_kinematic_n_members": kinematic_n_members,
                "sfr_membership_class": membership_class,
                "sfr_membership_name": membership_name,
                "sfr_membership_evidence": membership_evidence,
                "sfr_membership_status": membership_status,
                "sfr_membership_threshold": float(banyan_threshold),
                "sfr_membership_version": SFR_MEMBERSHIP_VERSION,
            }
        )

    result = candidates.copy()
    evidence = pd.DataFrame(output_rows, index=result.index)
    for column in SFR_MEMBERSHIP_OUTPUT_COLUMNS:
        result[column] = evidence[column] if column in evidence else pd.Series(
            index=result.index, dtype=object
        )
    return result


def load_sfr_association_crosswalk_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an in-memory crosswalk with the same rules as the CSV loader."""
    temporary = frame.copy()
    missing = [column for column in CROSSWALK_COLUMNS if column not in temporary.columns]
    if missing:
        raise ValueError(f"SFR association crosswalk is missing columns: {missing}")
    # Reuse the complete validation logic without filesystem I/O.
    out = temporary.loc[:, CROSSWALK_COLUMNS].copy()
    for column in ("sfr_name", "banyan_assoc", "relation", "source", "notes"):
        out[column] = out[column].fillna("").astype(str).str.strip()
    out["banyan_assoc"] = out["banyan_assoc"].str.upper()
    out["include_in_sfr_probability"] = out["include_in_sfr_probability"].map(
        _coerce_bool
    )
    required_nonempty = ("sfr_name", "banyan_assoc", "relation", "source")
    for column in required_nonempty:
        if out[column].eq("").any():
            raise ValueError(f"Crosswalk column {column!r} contains blanks")
    included = out[out["include_in_sfr_probability"]]
    if included.duplicated(["sfr_name", "banyan_assoc"]).any():
        raise ValueError("Duplicate included SFR/BANYAN mappings")
    ambiguous = included.groupby("banyan_assoc")["sfr_name"].nunique()
    if (ambiguous > 1).any():
        raise ValueError("An included BANYAN population maps to multiple SFRs")
    return out


def load_sfr_catalog_members_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an in-memory normalized association member catalog."""
    missing = [
        column for column in MEMBER_CATALOG_REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"SFR member catalog is missing columns: {missing}")
    out = frame.copy()
    for column in MEMBER_CATALOG_OPTIONAL_COLUMNS:
        if column not in out:
            out[column] = True if column == "accepted_member" else np.nan
    out["gaia_id"] = out["gaia_id"].map(_normalize_gaia_id)
    for column in (
        "association_name",
        "sfr_name",
        "catalog_name",
        "catalog_reference",
        "subcluster",
        "catalog_quality",
    ):
        out[column] = out[column].fillna("").astype(str).str.strip()
    out["accepted_member"] = out["accepted_member"].map(
        lambda value: _coerce_bool(value, default=True)
    )
    for column in (
        "catalog_membership_prob",
        "parallax",
        "parallax_error",
        "pmra",
        "pmra_error",
        "pmdec",
        "pmdec_error",
        "parallax_pmra_corr",
        "parallax_pmdec_corr",
        "pmra_pmdec_corr",
    ):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    invalid = out["gaia_id"].eq("") | out["association_name"].eq("") | out["sfr_name"].eq("")
    if invalid.any():
        raise ValueError("SFR member catalog has incomplete identity fields")
    probabilities = out["catalog_membership_prob"].dropna()
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("catalog_membership_prob values must lie in [0, 1]")
    if out.duplicated(
        ["gaia_id", "association_name", "sfr_name", "catalog_name"]
    ).any():
        raise ValueError("SFR member catalog contains duplicate membership rows")
    return out
