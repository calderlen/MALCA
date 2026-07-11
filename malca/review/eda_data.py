from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from malca.io.table_io import read_feature_table


BEST_FIELDS = [
    "vetting_likely_known",
    "vsx_class",
    "asassn_var_type",
    "gaia_var_class",
    "simbad_otype",
    "catalog_match",
    "period_n_sources",
    "period_consensus_days",
    "period_consensus_agree",
    "phase_quality_score",
    "periodicity_score",
    "periodicity_is_significant",
    "periodicity_bootstrap_sig",
    "periodicity_period",
    "periodicity_method",
    "lsp_is_significant",
    "lsp_bootstrap_sig",
    "lsp_period",
    "lsp_is_alias",
    "dip_is_single_event",
    "dip_run_count",
    "dipper_n_valid_dips",
    "dip_inter_event_spacing_median",
    "dip_inter_event_spacing_std",
    "dip_amplitude_consistency",
    "dip_duration_consistency",
    "dipper_score",
    "stats_photometry_robust_sigma_mag",
    "stats_amplitude",
    "stats_variability_stetson_J",
    "stats_percent_amplitude",
    "stats_skew",
    "stats_max_slope",
    "stats_variability_lag1_autocorr",
    "stats_autocor_length",
    "stats_harmonics_model_amplitude",
    "stats_harmonics_reduced_chi2",
    "final_class",
    "P_eb",
    "P_disk",
    "P_starspot",
    "P_cv",
]

DEFAULT_MAIN_X = "period_n_sources"
DEFAULT_MAIN_Y = "dip_run_count"
DEFAULT_COLOR = "periodic_evidence_bucket"
DEFAULT_SYMBOL = "oneoff_like"

TRUE_SET = {"1", "true", "t", "yes", "y"}
FALSE_SET = {"0", "false", "f", "no", "n"}


def _loads_payload(payload_json: object) -> dict[str, Any]:
    if payload_json in (None, "", b""):
        return {}
    try:
        obj = json.loads(payload_json)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def infer_source_kind(source_path: str | Path) -> str:
    path = Path(source_path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".db":
        return "db"
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".csv":
        return "csv"
    raise ValueError(f"Could not infer source kind from {path}")


def infer_plot_dir_from_source(source_path: str | Path, explicit_plot_dir: str | Path | None = None) -> Path | None:
    if explicit_plot_dir:
        plot_dir = Path(explicit_plot_dir).expanduser().resolve()
        return plot_dir if plot_dir.exists() else None

    path = Path(source_path).expanduser().resolve()
    if path.suffix.lower() == ".db" and path.parent.name == "review":
        candidate = path.parent.parent / "plots"
        if candidate.is_dir():
            return candidate
    if path.parent.name == "results":
        candidate = path.parent.parent / "plots"
        if candidate.is_dir():
            return candidate
    return None


def load_review_db(source_path: str | Path) -> pd.DataFrame:
    path = Path(source_path).expanduser().resolve()
    with sqlite3.connect(path) as conn:
        df = pd.read_sql_query("SELECT * FROM candidates", conn)
        try:
            reviews = pd.read_sql_query(
                "SELECT * FROM reviews",
                conn,
            )
        except Exception:
            reviews = pd.DataFrame()
    if "payload_json" in df.columns:
        payload_df = pd.json_normalize(df["payload_json"].map(_loads_payload))
        payload_df = payload_df.loc[:, ~payload_df.columns.duplicated()].reindex(df.index)
        base = df.drop(columns=["payload_json"])
        shared_cols = [col for col in payload_df.columns if col in base.columns]
        payload_only = payload_df.drop(columns=shared_cols, errors="ignore")
        if shared_cols:
            shared = base[shared_cols].combine_first(payload_df[shared_cols])
            base = base.drop(columns=shared_cols, errors="ignore")
            df = pd.concat([base, shared, payload_only], axis=1)
        else:
            df = pd.concat([base, payload_only], axis=1)
    if not reviews.empty and "candidate_id" in reviews.columns:
        df = df.merge(reviews, on="candidate_id", how="left")
    return df.loc[:, ~df.columns.duplicated()].copy()


def load_candidate_source(source_path: str | Path, source_kind: str | None = None) -> pd.DataFrame:
    path = Path(source_path).expanduser().resolve()
    kind = source_kind or infer_source_kind(path)
    if kind == "db":
        return load_review_db(path)
    if kind == "parquet":
        return read_feature_table(path)
    if kind == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported source kind: {kind}")


def bool_series(frame: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="bool")
    s = frame[col]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(default)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float) != 0
    text = s.astype(str).str.strip().str.lower()
    out = pd.Series(default, index=frame.index, dtype="bool")
    out.loc[text.isin(TRUE_SET)] = True
    out.loc[text.isin(FALSE_SET)] = False
    return out


def numeric_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def text_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[col].fillna("").astype(str)


def _coalesce_text_column(frame: pd.DataFrame, target: str, aliases: list[str]) -> None:
    values = text_series(frame, target).str.strip() if target in frame.columns else pd.Series("", index=frame.index, dtype="object")
    for alias in aliases:
        if alias not in frame.columns:
            continue
        alias_values = text_series(frame, alias).str.strip()
        mask = (values == "") & (alias_values != "")
        if bool(mask.any()):
            values.loc[mask] = alias_values.loc[mask]
    frame[target] = values


def contains_periodic_label(series: pd.Series) -> pd.Series:
    patterns = [
        r"\bEA\b",
        r"\bEB\b",
        r"\bEW\b",
        r"\bELL\b",
        r"\bECL\b",
        r"RR",
        r"CEP",
        r"DSCT",
        r"ROT",
        r"LPV",
        r"MIRA",
        r"ACV",
        r"BY",
        r"W UMA",
        r"ALGOL",
        r"EB\*",
        r"CV",
        r"PERIODIC",
    ]
    upper = series.fillna("").astype(str).str.upper()
    mask = pd.Series(False, index=series.index)
    for pattern in patterns:
        mask |= upper.str.contains(pattern, regex=True, na=False)
    return mask


def normalize_review_label(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"unknown dipper": "dipper"})
    )


def add_eda_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    _coalesce_text_column(out, "asassn_var_type", ["period_asassn_var_class"])
    _coalesce_text_column(out, "ztf_var_type", ["period_ztf_periodic_class"])

    numeric_cols = [
        "period_n_sources",
        "period_consensus_days",
        "phase_quality_score",
        "periodicity_score",
        "periodicity_bootstrap_sig",
        "periodicity_period",
        "lsp_bootstrap_sig",
        "lsp_power",
        "lsp_period",
        "dip_run_count",
        "dipper_n_valid_dips",
        "dip_inter_event_spacing_median",
        "dip_inter_event_spacing_std",
        "dip_amplitude_consistency",
        "dip_duration_consistency",
        "dipper_score",
        "P_eb",
        "P_disk",
        "P_starspot",
        "P_cv",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    periodic_text_match = (
        contains_periodic_label(text_series(out, "vsx_class"))
        | contains_periodic_label(text_series(out, "asassn_var_type"))
        | contains_periodic_label(text_series(out, "ztf_var_type"))
        | contains_periodic_label(text_series(out, "gaia_var_class"))
        | contains_periodic_label(text_series(out, "simbad_otype"))
        | contains_periodic_label(text_series(out, "alerce_lc_class"))
        | contains_periodic_label(text_series(out, "tns_type"))
    )

    out["known_periodic_catalog"] = bool_series(out, "vetting_likely_known") | periodic_text_match
    out["strong_catalog_period"] = (
        bool_series(out, "catalog_match")
        & bool_series(out, "period_consensus_agree")
        & (numeric_series(out, "period_n_sources").fillna(0) >= 2)
    )
    periodicity_significant = bool_series(out, "periodicity_is_significant") | bool_series(out, "lsp_is_significant")
    out["strong_native_period"] = (
        periodicity_significant
        & (~bool_series(out, "lsp_is_alias"))
        & (numeric_series(out, "phase_quality_score").fillna(-np.inf) >= 0.5)
    )
    out["recurrent_dips"] = (
        (numeric_series(out, "dip_run_count").fillna(0) >= 2)
        | (numeric_series(out, "dipper_n_valid_dips").fillna(0) >= 3)
    )
    out["oneoff_like"] = (
        bool_series(out, "dip_is_single_event")
        | (numeric_series(out, "dip_run_count").fillna(0) <= 1)
    )

    out["periodic_evidence_score"] = (
        out["known_periodic_catalog"].astype(int)
        + out["strong_catalog_period"].astype(int)
        + out["strong_native_period"].astype(int)
        + out["recurrent_dips"].astype(int)
    )
    out["periodic_evidence_bucket"] = pd.Categorical(
        np.select(
            [
                out["periodic_evidence_score"] >= 3,
                out["periodic_evidence_score"] == 2,
                out["periodic_evidence_score"] == 1,
            ],
            ["3+ signals", "2 signals", "1 signal"],
            default="0 signals",
        ),
        categories=["0 signals", "1 signal", "2 signals", "3+ signals"],
        ordered=True,
    )

    review_label = normalize_review_label(text_series(out, "event_class"))
    if "review_event_class" in out.columns:
        review_label = review_label.where(
            review_label.ne(""),
            normalize_review_label(text_series(out, "review_event_class")),
        )
    out["review_label"] = review_label
    out["is_reviewed"] = review_label.isin({
        "dipper",
        "yso",
        "microlensing",
        "flare",
        "instrumental",
        "unknown_interesting",
        "other",
    })
    out["is_reviewed_dipper"] = review_label.eq("dipper")
    out["is_reviewed_non_dipper"] = out["is_reviewed"] & (~out["is_reviewed_dipper"])

    out["proxy_periodic_contaminant"] = out["known_periodic_catalog"] | out["strong_catalog_period"]
    out["proxy_oneoff_dipper"] = (
        (~out["proxy_periodic_contaminant"])
        & out["oneoff_like"]
        & (numeric_series(out, "dipper_score").fillna(0) >= 5)
    )

    if out["is_reviewed_dipper"].any():
        out.attrs["default_target_col"] = "is_reviewed_dipper"
        out.attrs["default_reject_col"] = "is_reviewed_non_dipper"
    else:
        out.attrs["default_target_col"] = "proxy_oneoff_dipper"
        out.attrs["default_reject_col"] = "proxy_periodic_contaminant"

    out["final_class_label"] = text_series(out, "final_class").replace("", "unknown")
    return out


def available_metric_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [col for col in BEST_FIELDS if col in frame.columns]
    numeric_extra = [
        col
        for col in frame.columns
        if col not in preferred
        and pd.api.types.is_numeric_dtype(frame[col])
        and col not in {"candidate_key"}
    ]
    return preferred + sorted(numeric_extra)
