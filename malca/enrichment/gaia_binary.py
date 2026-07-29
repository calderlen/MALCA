"""Build evidence-based Gaia DR3 binary and eclipsing-binary enrichments.

This module deliberately keeps Gaia's one-to-many NSS solutions in a long-form
artifact and derives one candidate-level evidence row separately.  Evidence is
grouped by physical/data family so a Gaia photometric EB period copied into the
NSS table is not counted twice.

Usage::

    malca gaia-binary \
      --input output/runs/<run>/results/lc_events_vetted.parquet \
      --review-db output/runs/<run>/review/review.db \
      --update-review-db
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd

from malca.catalogs.gaia_ids import parse_gaia_source_id
from malca.catalogs.periodic_catalogs import (
    PERIOD_HARMONIC_FACTORS,
    fetch_gaia_dr3_eb_periods,
)
from malca.config import POST_FILTER_REL_TOL
from malca.io.table_io import read_parquet_table


DEFAULT_GAIA_DIR = Path("output/cache/catalogs/gaia")
DEFAULT_GAIA_SOURCE = DEFAULT_GAIA_DIR / "gaia_dr3_crossmatched.parquet"
DEFAULT_NSS = DEFAULT_GAIA_DIR / "NssTwoBodyOrbit_1.csv.gz"
DEFAULT_NSS_OUTPUT = DEFAULT_GAIA_DIR / "gaia_nss_candidate_solutions.parquet"
DEFAULT_EVIDENCE_OUTPUT = DEFAULT_GAIA_DIR / "gaia_binary_evidence.parquet"

NSS_SPECTROSCOPIC_TYPES = {
    "SB1",
    "SB1C",
    "SB2",
    "SB2C",
    "AstroSpectroSB1",
    "EclipsingSpectro",
}
NSS_SB1_TYPES = {"SB1", "SB1C", "AstroSpectroSB1", "EclipsingSpectro"}
NSS_SB2_TYPES = {"SB2", "SB2C"}
NSS_PHOTOMETRIC_TYPES = {"EclipsingBinary", "EclipsingSpectro"}
NSS_ASTROMETRIC_TYPES = {
    "Orbital",
    "OrbitalTargetedSearch",
    "OrbitalTargetedSearchValidated",
    "AstroSpectroSB1",
}
NSS_TYPE_PRIORITY = {
    "EclipsingSpectro": 0,
    "SB2": 1,
    "SB2C": 2,
    "AstroSpectroSB1": 3,
    "SB1": 4,
    "SB1C": 5,
    "Orbital": 6,
    "OrbitalTargetedSearchValidated": 7,
    "OrbitalTargetedSearch": 8,
    "EclipsingBinary": 9,
}

CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "gaia_id": ("gaia_id", "source_id_gaia"),
    "ruwe": ("ruwe", "ruwe_gaia"),
    "parallax": ("parallax", "parallax_gaia"),
    "parallax_error": ("parallax_error", "parallax_error_gaia"),
    "pmra": ("pmra", "pmra_gaia"),
    "pmra_error": ("pmra_error", "pmra_error_gaia"),
    "pmdec": ("pmdec", "pmdec_gaia"),
    "pmdec_error": ("pmdec_error", "pmdec_error_gaia"),
    "gaia_eb_period": ("gaia_eb_period", "period_gaia_eb_days"),
    "gaia_eb_morph": ("gaia_eb_morph", "period_gaia_eb_class"),
    "gaia_eb_global_ranking": ("gaia_eb_global_ranking",),
}

PERIOD_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("asassn", ("period_asassn_var_days", "asassn_var_period")),
    ("vsx", ("period_vsx_days", "vsx_period")),
    ("ztf", ("period_ztf_periodic_days", "ztf_var_period")),
    ("ogle", ("period_ogle_days",)),
    ("asassn_lc", ("periodicity_period", "phase_period_days")),
)

NSS_BEST_VALUE_COLUMNS = (
    "solution_id",
    "nss_solution_type",
    "period",
    "period_error",
    "eccentricity",
    "eccentricity_error",
    "center_of_mass_velocity",
    "center_of_mass_velocity_error",
    "semi_amplitude_primary",
    "semi_amplitude_primary_error",
    "semi_amplitude_secondary",
    "semi_amplitude_secondary_error",
    "mass_ratio",
    "mass_ratio_error",
    "inclination",
    "inclination_error",
    "goodness_of_fit",
    "significance",
    "flags",
    "conf_spectro_period",
    "input_period_error",
    "g_rank",
)

REVIEW_EVIDENCE_COLUMNS = (
    "gaia_binary_evidence_version",
    "gaia_binary_evidence_score_kind",
    "gaia_nss_solution_count",
    "gaia_nss_solution_types",
    "gaia_nss_solution_type",
    "gaia_nss_period",
    "gaia_nss_period_error",
    "gaia_nss_has_sb1",
    "gaia_nss_has_sb2",
    "gaia_nss_has_spectroscopic",
    "gaia_nss_has_astrometric",
    "gaia_nss_has_eclipsing",
    "gaia_nss_has_eclipsing_spectro",
    "gaia_nss_photometric_duplicate_of_eb",
    "gaia_nss_semi_amplitude_primary",
    "gaia_nss_semi_amplitude_secondary",
    "gaia_nss_mass_ratio",
    "gaia_nss_inclination",
    "gaia_eb_period_error",
    "gaia_eb_reduced_chi2",
    "gaia_eb_primary_depth",
    "gaia_eb_secondary_depth",
    "gaia_eb_primary_duration",
    "gaia_eb_secondary_duration",
    "gaia_eb_depth_ratio",
    "gaia_eb_eclipse_phase_separation",
    "gaia_eb_two_eclipses",
    "gaia_rv_variable_flag",
    "gaia_rv_large_amplitude_flag",
    "gaia_astrometric_anomaly_flag",
    "gaia_blend_contamination_flag",
    "gaia_binary_reference_period",
    "gaia_binary_reference_period_source",
    "gaia_binary_period_n_independent",
    "gaia_binary_period_agreement",
    "gaia_binary_period_agreement_sources",
    "gaia_binary_period_conflict",
    "gaia_binary_period_conflict_sources",
    "gaia_binary_evidence_families",
    "gaia_binary_n_evidence_families",
    "gaia_binary_evidence_level",
    "gaia_binary_evidence_score",
    "gaia_eb_evidence_level",
    "gaia_eb_evidence_score",
    "gaia_binary_evidence_summary",
)

GAIA_SOURCE_CONTEXT_COLUMNS = {
    "ra",
    "dec",
    "ref_epoch",
    "parallax",
    "parallax_error",
    "parallax_over_error",
    "ruwe",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "astrometric_params_solved",
    "astrometric_excess_noise",
    "astrometric_excess_noise_sig",
    "astrometric_n_good_obs_al",
    "astrometric_sigma5d_max",
    "visibility_periods_used",
    "ipd_frac_multi_peak",
    "ipd_frac_odd_win",
    "ipd_gof_harmonic_amplitude",
    "duplicated_source",
    "radial_velocity",
    "radial_velocity_error",
    "rv_amplitude_robust",
    "rv_nb_transits",
    "rv_chisq_pvalue",
    "rv_renormalised_gof",
    "rv_time_duration",
    "rv_method_used",
    "grvs_mag",
    "phot_bp_rp_excess_factor",
    "phot_bp_n_obs",
    "phot_rp_n_obs",
    "phot_bp_n_blended_transits",
    "phot_rp_n_blended_transits",
    "phot_bp_n_contaminated_transits",
    "phot_rp_n_contaminated_transits",
    "non_single_star",
    "phot_variable_flag",
    "has_epoch_photometry",
    "has_epoch_rv",
    "has_rvs",
    "gaia_fetch_schema_version",
    "gaia_fetch_updated_at",
}


def _gaia_id(value: object) -> str | None:
    parsed = parse_gaia_source_id(value)
    return str(parsed) if parsed is not None else None


def _first_present(frame: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    out = pd.Series(pd.NA, index=frame.index, dtype="object")
    for name in names:
        if name not in frame.columns:
            continue
        values = frame[name]
        missing = out.isna() | out.astype(str).str.strip().eq("")
        out.loc[missing] = values.loc[missing]
    return out


def normalize_gaia_binary_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Add canonical Gaia/EB columns while preserving the input columns."""
    out = frame.copy()
    for canonical, aliases in CANONICAL_ALIASES.items():
        values = _first_present(out, aliases)
        if canonical == "gaia_id":
            values = values.map(_gaia_id)
        out[canonical] = values

    if "candidate_id" not in out.columns:
        for fallback in ("asas_sn_id", "source_id", "gaia_id"):
            if fallback in out.columns:
                out["candidate_id"] = out[fallback].astype(str)
                break
    if "candidate_id" not in out.columns:
        raise ValueError("Candidate table needs candidate_id, asas_sn_id, source_id, or gaia_id")
    out["candidate_id"] = out["candidate_id"].astype(str)
    return out


def _fill_from(frame: pd.DataFrame, supplemental: pd.DataFrame, *, key: str) -> pd.DataFrame:
    if supplemental.empty or key not in supplemental.columns:
        return frame
    right = supplemental.drop_duplicates(key, keep="last").copy()
    merged = frame.merge(right, how="left", on=key, suffixes=("", "__supp"))
    for column in right.columns:
        if column == key:
            continue
        supp = f"{column}__supp"
        if supp not in merged.columns:
            continue
        if column in frame.columns:
            base = merged[column]
            missing = base.isna()
            if base.dtype == object or pd.api.types.is_string_dtype(base):
                missing |= base.astype(str).str.strip().isin({"", "nan", "<NA>"})
            merged.loc[missing, column] = merged.loc[missing, supp]
            merged = merged.drop(columns=supp)
        else:
            merged = merged.rename(columns={supp: column})
    return merged


def read_review_context(db_path: str | Path) -> pd.DataFrame:
    """Read only Gaia/binary context columns that exist in a Review database."""
    desired = {
        "candidate_id",
        "gaia_id",
        "ruwe",
        "parallax",
        "parallax_error",
        "pmra",
        "pmra_error",
        "pmdec",
        "pmdec_error",
        "rv_amplitude_robust",
        "radial_velocity",
        "radial_velocity_error",
        "gaia_eb_period",
        "gaia_eb_morph",
        "gaia_eb_global_ranking",
        "periodicity_period",
        "phase_period_days",
        "asassn_var_period",
        "vsx_period",
        "ztf_var_period",
        "period_conflict_flag",
    }
    with sqlite3.connect(str(db_path)) as conn:
        available = {str(row[1]) for row in conn.execute("PRAGMA table_info(candidates)")}
        columns = [column for column in desired if column in available]
        if not columns:
            return pd.DataFrame()
        return pd.read_sql_query(f"SELECT {', '.join(sorted(columns))} FROM candidates", conn)


def load_candidate_nss_solutions(
    nss_paths: Iterable[str | Path],
    gaia_ids: Iterable[str],
    *,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Read and retain every NSS solution for the requested Gaia source IDs."""
    wanted = {str(value) for value in gaia_ids if value}
    pieces: list[pd.DataFrame] = []
    for raw_path in nss_paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Gaia NSS catalog not found: {path}")
        for chunk in pd.read_csv(
            path,
            comment="#",
            chunksize=max(1, int(chunksize)),
            dtype={"source_id": "string", "solution_id": "string"},
            na_values=["null", "NaN"],
            low_memory=False,
        ):
            source_ids = chunk["source_id"].astype(str).str.replace(r"\.0$", "", regex=True)
            keep = source_ids.isin(wanted)
            if keep.any():
                selected = chunk.loc[keep].copy()
                selected["source_id"] = source_ids.loc[keep].to_numpy()
                pieces.append(selected)
    if not pieces:
        return pd.DataFrame(columns=["solution_id", "source_id", "nss_solution_type"])
    return pd.concat(pieces, ignore_index=True)


def _nss_summary(nss: pd.DataFrame) -> pd.DataFrame:
    if nss.empty:
        return pd.DataFrame(columns=["gaia_id"])
    work = nss.copy()
    work["gaia_id"] = work["source_id"].map(_gaia_id)
    work["_priority"] = work["nss_solution_type"].map(NSS_TYPE_PRIORITY).fillna(99)
    significance = (
        work["significance"]
        if "significance" in work.columns
        else pd.Series(np.nan, index=work.index)
    )
    work["_significance"] = pd.to_numeric(significance, errors="coerce").fillna(-np.inf)
    best = (
        work.sort_values(["gaia_id", "_priority", "_significance"], ascending=[True, True, False])
        .drop_duplicates("gaia_id", keep="first")
        .set_index("gaia_id")
    )
    rows: list[dict[str, object]] = []
    for gaia_id, group in work.groupby("gaia_id", sort=False):
        types = sorted({str(value) for value in group["nss_solution_type"].dropna()})
        chosen = best.loc[gaia_id]
        row: dict[str, object] = {
            "gaia_id": gaia_id,
            "gaia_nss_solution_count": int(len(group)),
            "gaia_nss_solution_types": ",".join(types),
            "gaia_nss_has_sb1": bool(set(types) & NSS_SB1_TYPES),
            "gaia_nss_has_sb2": bool(set(types) & NSS_SB2_TYPES),
            "gaia_nss_has_spectroscopic": bool(set(types) & NSS_SPECTROSCOPIC_TYPES),
            "gaia_nss_has_astrometric": bool(set(types) & NSS_ASTROMETRIC_TYPES),
            "gaia_nss_has_eclipsing": bool(set(types) & NSS_PHOTOMETRIC_TYPES),
            "gaia_nss_has_eclipsing_spectro": "EclipsingSpectro" in types,
            # DR3 NSS EclipsingBinary periods are copied from vari_eclipsing_binary.
            "gaia_nss_photometric_duplicate_of_eb": "EclipsingBinary" in types,
        }
        for column in NSS_BEST_VALUE_COLUMNS:
            target = "gaia_nss_solution_type" if column == "nss_solution_type" else f"gaia_nss_{column}"
            row[target] = chosen.get(column, np.nan)
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _prepare_gaia_source(path: str | Path | None, wanted: set[str]) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=["gaia_id"])
    source = read_parquet_table(path).copy()
    source.columns = [str(column).lower() for column in source.columns]
    if "source_id" not in source.columns:
        return pd.DataFrame(columns=["gaia_id"])
    source["gaia_id"] = source["source_id"].map(_gaia_id)
    source = source[source["gaia_id"].isin(wanted)].drop(columns="source_id")
    return source.drop_duplicates("gaia_id", keep="last")


def prepare_gaia_eb_frame(eb: pd.DataFrame) -> pd.DataFrame:
    if eb.empty:
        return pd.DataFrame(columns=["gaia_id"])
    renames = {
        "source_id": "gaia_id",
        "period": "gaia_eb_period",
        "period_error": "gaia_eb_period_error",
        "var_type": "gaia_eb_morph",
        "global_ranking": "gaia_eb_global_ranking",
        "model_type": "gaia_eb_model_type",
        "reduced_chi2": "gaia_eb_reduced_chi2",
        "derived_primary_ecl_phase": "gaia_eb_primary_phase",
        "derived_primary_ecl_phase_error": "gaia_eb_primary_phase_error",
        "derived_primary_ecl_duration": "gaia_eb_primary_duration",
        "derived_primary_ecl_duration_error": "gaia_eb_primary_duration_error",
        "derived_primary_ecl_depth": "gaia_eb_primary_depth",
        "derived_primary_ecl_depth_error": "gaia_eb_primary_depth_error",
        "derived_secondary_ecl_phase": "gaia_eb_secondary_phase",
        "derived_secondary_ecl_phase_error": "gaia_eb_secondary_phase_error",
        "derived_secondary_ecl_duration": "gaia_eb_secondary_duration",
        "derived_secondary_ecl_duration_error": "gaia_eb_secondary_duration_error",
        "derived_secondary_ecl_depth": "gaia_eb_secondary_depth",
        "derived_secondary_ecl_depth_error": "gaia_eb_secondary_depth_error",
    }
    out = eb.rename(columns=renames).copy()
    out["gaia_id"] = out["gaia_id"].map(_gaia_id)
    for column in list(out.columns):
        if column == "gaia_id" or column.startswith("gaia_eb_"):
            continue
        out = out.rename(columns={column: f"gaia_eb_{column}"})
    return out.drop_duplicates("gaia_id", keep="last")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(values.to_numpy(dtype=float, na_value=np.nan), index=frame.index, dtype=float)


def _truth(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    text_truth = values.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "t", "yes", "y"}
    )
    return ((numeric.notna() & numeric.ne(0)) | text_truth).fillna(False).astype(bool)


def _period_match(reference: float, comparison: float) -> tuple[bool, float, float]:
    if not (np.isfinite(reference) and np.isfinite(comparison)) or reference <= 0 or comparison <= 0:
        return False, np.nan, np.nan
    ratio = comparison / reference
    residuals = [(abs(ratio - factor) / factor, factor) for factor in PERIOD_HARMONIC_FACTORS]
    residual, factor = min(residuals)
    return bool(residual <= POST_FILTER_REL_TOL), float(factor), float(residual)


def _derive_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    primary_depth = _numeric(out, "gaia_eb_primary_depth")
    secondary_depth = _numeric(out, "gaia_eb_secondary_depth")
    two_eclipses = primary_depth.gt(0) & secondary_depth.gt(0)
    out["gaia_eb_two_eclipses"] = two_eclipses
    out["gaia_eb_depth_ratio"] = np.where(
        two_eclipses,
        np.minimum(primary_depth, secondary_depth) / np.maximum(primary_depth, secondary_depth),
        np.nan,
    )
    primary_phase = _numeric(out, "gaia_eb_primary_phase")
    secondary_phase = _numeric(out, "gaia_eb_secondary_phase")
    separation = (secondary_phase - primary_phase).abs() % 1.0
    out["gaia_eb_eclipse_phase_separation"] = np.minimum(separation, 1.0 - separation)

    rv_n = _numeric(out, "rv_nb_transits")
    rv_p = _numeric(out, "rv_chisq_pvalue")
    rv_gof = _numeric(out, "rv_renormalised_gof")
    rv_amp = _numeric(out, "rv_amplitude_robust")
    out["gaia_rv_variable_flag"] = rv_n.ge(10) & rv_p.lt(0.01) & rv_gof.gt(4)
    out["gaia_rv_large_amplitude_flag"] = rv_n.ge(10) & rv_amp.ge(20)

    ruwe = _numeric(out, "ruwe")
    visibility = _numeric(out, "visibility_periods_used")
    excess_sig = _numeric(out, "astrometric_excess_noise_sig")
    out["gaia_astrometric_anomaly_flag"] = ruwe.gt(1.4) & (
        visibility.isna() | visibility.ge(9)
    ) & (excess_sig.isna() | excess_sig.gt(2))

    bp_obs = _numeric(out, "phot_bp_n_obs")
    rp_obs = _numeric(out, "phot_rp_n_obs")
    bp_blend = _numeric(out, "phot_bp_n_blended_transits")
    rp_blend = _numeric(out, "phot_rp_n_blended_transits")
    blend_fraction = pd.concat(
        [bp_blend.div(bp_obs.where(bp_obs.gt(0))), rp_blend.div(rp_obs.where(rp_obs.gt(0)))],
        axis=1,
    ).max(axis=1)
    out["gaia_blend_contamination_flag"] = (
        _truth(out, "duplicated_source")
        | blend_fraction.gt(0.1)
        | _numeric(out, "ipd_frac_multi_peak").gt(10)
    ).fillna(False).astype(bool)

    nss_period = _numeric(out, "gaia_nss_period")
    eb_period = _numeric(out, "gaia_eb_period")
    spectro = _truth(out, "gaia_nss_has_spectroscopic") | _truth(out, "gaia_rv_variable_flag")
    astrometric = _truth(out, "gaia_nss_has_astrometric")
    photometric = (
        _truth(out, "gaia_nss_has_eclipsing")
        | eb_period.gt(0)
        | _truth(out, "period_gaia_eb_match")
    )
    prefer_nss = nss_period.gt(0) & _truth(out, "gaia_nss_has_spectroscopic")
    reference_period = eb_period.where(eb_period.gt(0), nss_period)
    reference_period = reference_period.where(~prefer_nss, nss_period)
    out["gaia_binary_reference_period"] = reference_period
    out["gaia_binary_reference_period_source"] = np.where(
        prefer_nss,
        "gaia_nss_spectroscopic",
        np.where(eb_period.gt(0), "gaia_eb", np.where(nss_period.gt(0), "gaia_nss", "")),
    )

    agreement_flags: list[bool] = []
    agreement_sources: list[str] = []
    conflict_flags: list[bool] = []
    conflict_sources: list[str] = []
    independent_counts: list[int] = []
    for idx, reference in reference_period.items():
        agreements: list[str] = []
        conflicts: list[str] = []
        seen_sources: set[str] = set()
        reference_is_valid = bool(np.isfinite(reference) and float(reference) > 0)
        for source, columns in PERIOD_COLUMNS:
            value = np.nan
            for column in columns:
                if column in out.columns:
                    candidate = pd.to_numeric(pd.Series([out.at[idx, column]]), errors="coerce").iloc[0]
                    if np.isfinite(candidate) and candidate > 0:
                        value = float(candidate)
                        break
            if not np.isfinite(value) or source in seen_sources:
                continue
            seen_sources.add(source)
            if not reference_is_valid:
                continue
            agrees, _factor, _residual = _period_match(float(reference), value)
            (agreements if agrees else conflicts).append(source)
        independent_counts.append(len(seen_sources))
        agreement_flags.append(bool(agreements))
        conflict_flags.append(bool(conflicts))
        agreement_sources.append(",".join(agreements))
        conflict_sources.append(",".join(conflicts))
    out["gaia_binary_period_n_independent"] = independent_counts
    out["gaia_binary_period_agreement"] = agreement_flags
    out["gaia_binary_period_agreement_sources"] = agreement_sources
    out["gaia_binary_period_conflict"] = conflict_flags
    out["gaia_binary_period_conflict_sources"] = conflict_sources

    level_binary: list[str] = []
    level_eb: list[str] = []
    binary_score: list[float] = []
    eb_score: list[float] = []
    family_strings: list[str] = []
    family_counts: list[int] = []
    summaries: list[str] = []
    supporting = (
        _truth(out, "gaia_rv_large_amplitude_flag")
        | _truth(out, "gaia_astrometric_anomaly_flag")
        | _truth(out, "non_single_star")
    )
    for pos, idx in enumerate(out.index):
        families: list[str] = []
        if bool(photometric.loc[idx]):
            families.append("gaia_photometric_eb")
        if bool(spectro.loc[idx]):
            families.append("gaia_spectroscopy")
        if bool(astrometric.loc[idx]):
            families.append("gaia_astrometric_orbit")
        if bool(agreement_flags[pos]):
            families.append("independent_period")
        direct = sum(name != "independent_period" for name in families)
        n_family = len(families)
        conflicted = bool(conflict_flags[pos])

        if n_family >= 3:
            b_level, b_score = "very_strong", 0.95
        elif n_family >= 2:
            b_level, b_score = "strong", 0.80
        elif direct >= 1:
            b_level, b_score = "moderate", 0.60
        elif bool(supporting.loc[idx]):
            b_level, b_score = "supporting", 0.25
        else:
            b_level, b_score = "none", 0.0
        if conflicted and b_level in {"moderate", "supporting"}:
            b_level, b_score = "conflicted", min(b_score, 0.35)

        if bool(photometric.loc[idx]) and bool(spectro.loc[idx]) and bool(agreement_flags[pos]):
            e_level, e_score = "very_strong", 0.97
        elif bool(photometric.loc[idx]) and (bool(spectro.loc[idx]) or bool(agreement_flags[pos])):
            e_level, e_score = "strong", 0.85
        elif bool(photometric.loc[idx]):
            e_level, e_score = "moderate", 0.62
        elif bool(supporting.loc[idx]) or bool(spectro.loc[idx]) or bool(astrometric.loc[idx]):
            e_level, e_score = "supporting", 0.20
        else:
            e_level, e_score = "none", 0.0
        if conflicted and e_level in {"moderate", "supporting"}:
            e_level, e_score = "conflicted", min(e_score, 0.35)
        if bool(out.at[idx, "gaia_blend_contamination_flag"]):
            e_score = max(0.0, e_score - 0.08)

        level_binary.append(b_level)
        binary_score.append(b_score)
        level_eb.append(e_level)
        eb_score.append(e_score)
        family_strings.append(",".join(families))
        family_counts.append(n_family)
        detail = families or (["quality-only indicators"] if bool(supporting.loc[idx]) else ["no Gaia binary evidence"])
        if conflicted:
            detail = [*detail, "period conflict"]
        if bool(out.at[idx, "gaia_blend_contamination_flag"]):
            detail = [*detail, "blend warning"]
        summaries.append("; ".join(detail))

    out["gaia_binary_evidence_families"] = family_strings
    out["gaia_binary_n_evidence_families"] = family_counts
    out["gaia_binary_evidence_level"] = level_binary
    out["gaia_binary_evidence_score"] = binary_score
    out["gaia_eb_evidence_level"] = level_eb
    out["gaia_eb_evidence_score"] = eb_score
    out["gaia_binary_evidence_summary"] = summaries
    out["gaia_binary_evidence_version"] = "gaia_binary_evidence_v1"
    out["gaia_binary_evidence_score_kind"] = "rule_based_not_probability"
    return out


def build_gaia_binary_evidence(
    candidates: pd.DataFrame,
    *,
    nss_solutions: pd.DataFrame | None = None,
    gaia_source: pd.DataFrame | None = None,
    gaia_eb: pd.DataFrame | None = None,
    review_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one evidence row per candidate without double-counting Gaia products."""
    work = normalize_gaia_binary_aliases(candidates)
    if review_context is not None and not review_context.empty:
        context = normalize_gaia_binary_aliases(review_context)
        work = _fill_from(work, context, key="candidate_id")
    if gaia_source is not None and not gaia_source.empty:
        source = gaia_source.copy()
        if "gaia_id" not in source.columns and "source_id" in source.columns:
            source = source.rename(columns={"source_id": "gaia_id"})
        if "gaia_id" in source.columns:
            source["gaia_id"] = source["gaia_id"].map(_gaia_id)
        work = _fill_from(work, source, key="gaia_id")
    if gaia_eb is not None and not gaia_eb.empty:
        work = _fill_from(work, prepare_gaia_eb_frame(gaia_eb), key="gaia_id")
    if nss_solutions is not None and not nss_solutions.empty:
        work = _fill_from(work, _nss_summary(nss_solutions), key="gaia_id")
    work = _derive_evidence(work)

    identity = [column for column in ("candidate_id", "gaia_id") if column in work.columns]
    gaia_source_columns = [
        column
        for column in work.columns
        if column.startswith(("gaia_", "rv_", "astrometric_", "ipd_", "phot_bp_n_", "phot_rp_n_"))
        or column in GAIA_SOURCE_CONTEXT_COLUMNS
    ]
    columns = list(dict.fromkeys([*identity, *gaia_source_columns]))
    return work.loc[:, columns].drop_duplicates("candidate_id", keep="last").reset_index(drop=True)


def _atomic_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp.parquet", dir=output.parent)
    os.close(fd)
    try:
        frame.to_parquet(temporary, index=False, compression="snappy")
        check = pd.read_parquet(temporary)
        if len(check) != len(frame):
            raise RuntimeError(f"Parquet row-count validation failed for {output}")
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return output


def update_review_database(db_path: str | Path, evidence: pd.DataFrame) -> int:
    """Backfill only Gaia evidence columns in an existing Review database."""
    from malca.review.store import _COL_NAMES, ensure_review_db_schema

    ensure_review_db_schema(db_path)
    columns = [
        column
        for column in evidence.columns
        if column in _COL_NAMES
        and (column.startswith("gaia_") or column in GAIA_SOURCE_CONTEXT_COLUMNS)
    ]
    if not columns:
        return 0
    derived = set(REVIEW_EVIDENCE_COLUMNS)
    assignments = ", ".join(
        f"{column} = ?" if column in derived else f"{column} = COALESCE(?, {column})"
        for column in columns
    )
    rows: list[tuple[object, ...]] = []
    for _, row in evidence.iterrows():
        values: list[object] = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append(None)
            elif isinstance(value, (bool, np.bool_)):
                values.append(int(value))
            elif isinstance(value, np.generic):
                values.append(value.item())
            else:
                values.append(value)
        values.append(str(row["candidate_id"]))
        rows.append(tuple(values))
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.executemany(f"UPDATE candidates SET {assignments} WHERE candidate_id = ?", rows)
        conn.commit()
        return int(cursor.rowcount if cursor.rowcount >= 0 else len(rows))


def resolve_nss_paths(values: Iterable[str | Path] | None) -> list[Path]:
    paths = [Path(value) for value in values] if values else [DEFAULT_NSS]
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.glob("NssTwoBodyOrbit_*.csv.gz")))
        else:
            expanded.append(path)
    return expanded


def run_gaia_binary_enrichment(
    candidates: pd.DataFrame,
    *,
    gaia_source_path: str | Path | None = DEFAULT_GAIA_SOURCE,
    nss_paths: Iterable[str | Path] | None = None,
    eb_cache_dir: str | Path = DEFAULT_GAIA_DIR,
    nss_output: str | Path | None = None,
    evidence_output: str | Path | None = None,
    review_context: pd.DataFrame | None = None,
    fetch_eb: bool = True,
    query_all_eb: bool = False,
    chunk_size: int = 1000,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build candidate evidence and long-form NSS solutions for one cohort.

    The returned tuple is ``(evidence, nss_long)``.  Optional output paths are
    written atomically, making this callable suitable for both the STV stage
    and the standalone CLI.
    """
    normalized = normalize_gaia_binary_aliases(candidates)
    gaia_ids = sorted({value for value in normalized["gaia_id"].dropna().astype(str) if value})
    candidate_map = normalized[["candidate_id", "gaia_id"]].dropna().drop_duplicates()
    if show_progress:
        print(f"Gaia binary: {len(normalized):,} candidates; {len(gaia_ids):,} unique Gaia IDs")

    resolved_nss_paths = resolve_nss_paths(nss_paths)
    if gaia_ids:
        if not resolved_nss_paths:
            raise FileNotFoundError("No Gaia NSS NssTwoBodyOrbit CSV files were found")
        nss = load_candidate_nss_solutions(resolved_nss_paths, gaia_ids)
    else:
        # Do not scan the full NSS archive for an empty/no-Gaia cohort.  Still
        # emit valid empty sidecars so an empty STV run remains reproducible.
        nss = pd.DataFrame(columns=["solution_id", "source_id", "nss_solution_type"])
    nss_long = nss.merge(candidate_map, how="left", left_on="source_id", right_on="gaia_id")
    leading = [column for column in ("candidate_id", "gaia_id", "source_id") if column in nss_long.columns]
    nss_long = nss_long.loc[:, [*leading, *[column for column in nss_long.columns if column not in leading]]]
    if nss_output is not None:
        _atomic_parquet(nss_long, nss_output)
        if show_progress:
            print(
                f"Gaia binary: wrote {len(nss_long):,} NSS solutions for "
                f"{nss_long['source_id'].nunique():,} candidates to {nss_output}"
            )

    source = _prepare_gaia_source(gaia_source_path, set(gaia_ids))
    if query_all_eb:
        eb_gaia_ids = gaia_ids
    else:
        known_eb = _numeric(normalized, "gaia_eb_period").gt(0) | _truth(
            normalized, "period_gaia_eb_match"
        )
        eb_gaia_ids = normalized.loc[known_eb, "gaia_id"].dropna().astype(str).tolist()
        if not nss.empty:
            nss_eb = nss["nss_solution_type"].astype(str).isin(NSS_PHOTOMETRIC_TYPES)
            eb_gaia_ids.extend(nss.loc[nss_eb, "source_id"].dropna().astype(str).tolist())
        eb_gaia_ids = sorted(set(eb_gaia_ids))

    eb = pd.DataFrame()
    if eb_gaia_ids:
        if show_progress:
            print(
                f"Gaia binary: loading full EB models for {len(eb_gaia_ids):,} source(s)"
                + (" (all-candidate audit)" if query_all_eb else " with existing EB evidence")
                + (" (cache only)" if not fetch_eb else "")
            )
        eb = fetch_gaia_dr3_eb_periods(
            [int(value) for value in eb_gaia_ids],
            cache_dir=Path(eb_cache_dir),
            chunk_size=chunk_size,
            show_tqdm=show_progress,
            allow_network=fetch_eb,
        )

    evidence = build_gaia_binary_evidence(
        normalized,
        nss_solutions=nss,
        gaia_source=source,
        gaia_eb=eb,
        review_context=review_context,
    )
    if evidence_output is not None:
        _atomic_parquet(evidence, evidence_output)
        if show_progress:
            print(f"Gaia binary: wrote {len(evidence):,} candidate evidence rows to {evidence_output}")
    return evidence, nss_long


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gaia DR3 binary/EB evidence artifacts")
    parser.add_argument("--input", required=True, type=Path, help="Candidate Parquet table")
    parser.add_argument("--gaia-source", type=Path, default=DEFAULT_GAIA_SOURCE)
    parser.add_argument("--nss", action="append", help="NSS CSV.gz file or directory; repeatable")
    parser.add_argument("--nss-output", type=Path, default=DEFAULT_NSS_OUTPUT)
    parser.add_argument("--evidence-output", type=Path, default=DEFAULT_EVIDENCE_OUTPUT)
    parser.add_argument("--review-db", type=Path, help="Optional Review DB context")
    parser.add_argument("--update-review-db", action="store_true", help="Backfill evidence into --review-db")
    parser.add_argument("--offline", action="store_true", help="Do not query vari_eclipsing_binary")
    parser.add_argument(
        "--query-all-eb",
        action="store_true",
        help="Audit every candidate against vari_eclipsing_binary, including expected negative lookups",
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="Gaia EB TAP chunk size")
    args = parser.parse_args()

    review_context = read_review_context(args.review_db) if args.review_db else None
    evidence, _nss_long = run_gaia_binary_enrichment(
        read_parquet_table(args.input),
        gaia_source_path=args.gaia_source,
        nss_paths=args.nss,
        eb_cache_dir=args.gaia_source.parent,
        nss_output=args.nss_output,
        evidence_output=args.evidence_output,
        review_context=review_context,
        fetch_eb=not args.offline,
        query_all_eb=args.query_all_eb,
        chunk_size=args.chunk_size,
    )
    print("Gaia EB evidence levels:")
    print(evidence["gaia_eb_evidence_level"].value_counts(dropna=False).to_string())

    if args.update_review_db:
        if args.review_db is None:
            parser.error("--update-review-db requires --review-db")
        updated = update_review_database(args.review_db, evidence)
        print(f"Gaia binary: updated {updated:,} Review candidate rows")


if __name__ == "__main__":
    main()
