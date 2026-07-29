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
    "pdm_period",
    "pdm_corrected_period",
    "pdm_bootstrap_sig",
    "ce_period",
    "ce_corrected_period",
    "ce_bootstrap_sig",
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
    "prob_dipper_like",
    "dipper_score",
    "stats_photometry_robust_sigma_mag",
    "stats_amplitude",
    "stats_variability_stetson_J",
    "stats_percent_amplitude",
    "stats_skew",
    "stats_max_slope",
    "stats_variability_lag1_autocorr",
    "stats_autocor_length",
    "stats_variability_quasi_periodicity_q",
    "stats_variability_quasi_periodicity_bin_coverage",
    "stats_variability_quasi_periodicity_scatter_ratio",
    "stats_variability_quasi_periodicity_status",
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

EDA_REQUIRED_FIELDS = tuple(dict.fromkeys([
    "candidate_id",
    "candidate_key",
    "source_path",
    "asas_sn_id",
    "gaia_id",
    "status",
    "workflow_status",
    "event_class",
    "review_event_class",
    "classification_confidence",
    "review_pass",
    "period_asassn_var_class",
    "period_ztf_periodic_class",
    "ztf_var_type",
    "alerce_lc_class",
    "tns_type",
    "catalog_match",
    "period_consensus_agree",
    "periodicity_alias_flag",
    "lsp_is_significant",
    "lsp_is_alias",
    "dip_is_single_event",
    *BEST_FIELDS,
]))

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


def _quote_sql_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def load_review_db(
    source_path: str | Path,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load review candidate data, optionally restricted to requested fields.

    The EDA path supplies ``columns`` so it does not materialize all ~1,000
    candidate columns.  Missing requested SQL fields are recovered from the
    flattened payload JSON without expanding unrelated payload keys.
    """

    path = Path(source_path).expanduser().resolve()
    with sqlite3.connect(path) as conn:
        requested = list(dict.fromkeys(str(col) for col in (columns or []) if str(col)))
        if not requested:
            df = pd.read_sql_query("SELECT * FROM candidates", conn)
        else:
            candidate_columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(candidates)").fetchall()
            }
            review_columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(reviews)").fetchall()
            }
            selected_candidate_columns = [
                col for col in requested if col in candidate_columns
            ]
            if "candidate_id" in candidate_columns and "candidate_id" not in selected_candidate_columns:
                selected_candidate_columns.insert(0, "candidate_id")
            payload_fields = [
                col for col in requested
                if col not in candidate_columns
                and col not in review_columns
                and col != "payload_json"
            ]
            select_columns = list(selected_candidate_columns)
            if payload_fields and "payload_json" in candidate_columns:
                select_columns.append("payload_json")
            if not select_columns:
                select_columns = ["candidate_id"]
            select_sql = ", ".join(_quote_sql_identifier(col) for col in select_columns)
            df = pd.read_sql_query(f"SELECT {select_sql} FROM candidates", conn)
            if payload_fields and "payload_json" in df.columns:
                payload_rows = []
                for raw in df["payload_json"].tolist():
                    payload = _loads_payload(raw)
                    payload_rows.append({key: payload.get(key) for key in payload_fields})
                payload_df = pd.DataFrame(payload_rows, index=df.index)
                df = pd.concat([df.drop(columns=["payload_json"]), payload_df], axis=1)
        try:
            if not requested:
                reviews = pd.read_sql_query("SELECT * FROM reviews", conn)
            else:
                selected_review_columns = [
                    col for col in requested
                    if col in review_columns and col != "candidate_id"
                ]
                if "candidate_id" in review_columns:
                    selected_review_columns.insert(0, "candidate_id")
                if selected_review_columns:
                    review_sql = ", ".join(
                        _quote_sql_identifier(col) for col in selected_review_columns
                    )
                    reviews = pd.read_sql_query(f"SELECT {review_sql} FROM reviews", conn)
                else:
                    reviews = pd.DataFrame()
        except Exception:
            reviews = pd.DataFrame()
    if columns is None and "payload_json" in df.columns:
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


def bool_series(frame: pd.DataFrame, col: str, default: bool | None = None) -> pd.Series:
    """Parse a stored boolean without turning missing/malformed data into False."""
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="boolean")
    s = frame[col]
    if pd.api.types.is_bool_dtype(s):
        out = s.astype("boolean")
        return out if default is None else out.fillna(default)
    if pd.api.types.is_numeric_dtype(s):
        numeric = pd.to_numeric(s, errors="coerce")
        out = pd.Series(pd.NA, index=frame.index, dtype="boolean")
        out.loc[numeric.eq(1)] = True
        out.loc[numeric.eq(0)] = False
        return out if default is None else out.fillna(default)
    text = s.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    out.loc[text.isin(TRUE_SET)] = True
    out.loc[text.isin(FALSE_SET)] = False
    return out if default is None else out.fillna(default)


def numeric_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def text_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[col].fillna("").astype(str)


def numeric_predicate(frame: pd.DataFrame, col: str, op: str, threshold: float) -> pd.Series:
    values = numeric_series(frame, col)
    result = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    valid = values.notna() & np.isfinite(values)
    if op == "ge":
        result.loc[valid] = values.loc[valid].ge(threshold)
    elif op == "le":
        result.loc[valid] = values.loc[valid].le(threshold)
    else:
        raise ValueError(f"Unsupported comparison: {op}")
    return result


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
    # Catalog class codes must be token matches. Unbounded fragments such as
    # ``RR`` and ``ROT`` incorrectly classified IRREGULAR and PROTOSTAR.
    pattern = (
        r"(?:^|[^A-Z0-9])(?:EA|EB|EW|ELL|ECL|RR(?:AB|C|D|LYR)?|"
        r"CEP(?:H|I|II)?|DSCT|ROT|LPV|MIRA|ACV|BY|W\s*UMA|ALGOL|CV|PERIODIC)"
        r"(?:$|[^A-Z0-9])"
    )
    upper = series.fillna("").astype(str).str.upper()
    return upper.str.contains(pattern, regex=True, na=False)


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
        "pdm_period",
        "pdm_corrected_period",
        "pdm_bootstrap_sig",
        "ce_period",
        "ce_corrected_period",
        "ce_bootstrap_sig",
        "lsp_bootstrap_sig",
        "lsp_power",
        "lsp_period",
        "dip_run_count",
        "dipper_n_valid_dips",
        "dip_inter_event_spacing_median",
        "dip_inter_event_spacing_std",
        "dip_amplitude_consistency",
        "dip_duration_consistency",
        "prob_dipper_like",
        "dipper_score",
        "P_eb",
        "P_disk",
        "P_starspot",
        "P_cv",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    catalog_label_columns = [
        "vsx_class",
        "asassn_var_type",
        "ztf_var_type",
        "gaia_var_class",
        "simbad_otype",
        "alerce_lc_class",
        "tns_type",
    ]
    catalog_label_present = pd.Series(False, index=out.index, dtype="bool")
    periodic_text_match_raw = pd.Series(False, index=out.index, dtype="bool")
    for catalog_col in catalog_label_columns:
        if catalog_col not in out.columns:
            continue
        text = text_series(out, catalog_col).str.strip()
        catalog_label_present |= text.ne("")
        periodic_text_match_raw |= contains_periodic_label(text)
    periodic_text_match = pd.Series(pd.NA, index=out.index, dtype="boolean")
    periodic_text_match.loc[catalog_label_present] = periodic_text_match_raw.loc[catalog_label_present]

    out["known_periodic_catalog"] = periodic_text_match
    out["strong_catalog_period"] = (
        bool_series(out, "catalog_match")
        & bool_series(out, "period_consensus_agree")
        & numeric_predicate(out, "period_n_sources", "ge", 2)
    )
    periodicity_significant = bool_series(out, "periodicity_is_significant") | bool_series(out, "lsp_is_significant")
    alias_flag = bool_series(out, "periodicity_alias_flag")
    alias_flag = alias_flag.fillna(bool_series(out, "lsp_is_alias"))
    out["strong_native_period"] = (
        periodicity_significant
        & (~alias_flag)
        & numeric_predicate(out, "phase_quality_score", "ge", 0.5)
    )
    out["recurrent_dips"] = (
        numeric_predicate(out, "dip_run_count", "ge", 2)
        | numeric_predicate(out, "dipper_n_valid_dips", "ge", 3)
    )
    dip_run_count = numeric_series(out, "dip_run_count")
    exactly_one_run = pd.Series(pd.NA, index=out.index, dtype="boolean")
    valid_run_count = dip_run_count.notna() & np.isfinite(dip_run_count)
    exactly_one_run.loc[valid_run_count] = dip_run_count.loc[valid_run_count].eq(1)
    explicit_single = bool_series(out, "dip_is_single_event")
    oneoff_like = explicit_single.fillna(False) | exactly_one_run.fillna(False)
    out["oneoff_like"] = oneoff_like.where(explicit_single.notna() | exactly_one_run.notna(), pd.NA)

    evidence_columns = [
        "known_periodic_catalog",
        "strong_catalog_period",
        "strong_native_period",
        "recurrent_dips",
    ]
    evidence = out[evidence_columns].astype("boolean")
    evidence_complete = evidence.notna().all(axis=1)
    score = evidence.astype("Int64").sum(axis=1, min_count=len(evidence_columns)).astype("Int64")
    out["periodic_evidence_complete"] = evidence_complete
    out["periodic_evidence_score"] = score.where(evidence_complete, pd.NA)
    score_numeric = out["periodic_evidence_score"].astype("Float64").fillna(-1)
    out["periodic_evidence_bucket"] = pd.Categorical(
        np.select(
            [
                score_numeric >= 3,
                score_numeric == 2,
                score_numeric == 1,
                score_numeric == 0,
            ],
            ["3+ signals", "2 signals", "1 signal", "0 signals"],
            default="unknown",
        ),
        categories=["unknown", "0 signals", "1 signal", "2 signals", "3+ signals"],
        ordered=True,
    )

    review_label = normalize_review_label(text_series(out, "event_class"))
    if "review_event_class" in out.columns:
        review_label = review_label.where(
            review_label.ne(""),
            normalize_review_label(text_series(out, "review_event_class")),
        )
    out["review_label"] = review_label
    workflow = text_series(out, "workflow_status").str.strip().str.lower()
    status = text_series(out, "status").str.strip().str.lower()
    workflow = workflow.where(workflow.ne(""), status)
    has_workflow = workflow.ne("")
    out["is_reviewed"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out.loc[has_workflow, "is_reviewed"] = workflow.loc[has_workflow].eq("reviewed")
    out["review_status_known"] = has_workflow
    out["is_reviewed_dipper"] = out["is_reviewed"] & review_label.eq("dipper").astype("boolean")
    recognized_non_dipper = review_label.isin(
        {"ltv", "microlensing", "flare", "instrumental", "other", "periodic"}
    )
    reviewed_non_dipper = pd.Series(pd.NA, index=out.index, dtype="boolean")
    reviewed_non_dipper.loc[out["is_reviewed"].eq(False)] = False
    reviewed_non_dipper.loc[out["is_reviewed"].eq(True) & review_label.eq("dipper")] = False
    reviewed_non_dipper.loc[out["is_reviewed"].eq(True) & recognized_non_dipper] = True
    out["is_reviewed_non_dipper"] = reviewed_non_dipper

    out["proxy_periodic_contaminant"] = out["known_periodic_catalog"] | out["strong_catalog_period"]
    out["proxy_oneoff_dipper"] = (
        (~out["proxy_periodic_contaminant"])
        & out["oneoff_like"]
        & numeric_predicate(out, "dipper_score", "ge", 5)
    )

    if out["is_reviewed_dipper"].fillna(False).any():
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
