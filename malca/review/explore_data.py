from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
    if suffix in {".parquet", ".pq"}:
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


def infer_plot_dir_for_record(record: dict[str, Any], fallback_plot_dir: Path | None) -> Path | None:
    source_path = record.get("source_path")
    if source_path:
        candidate = Path(str(source_path)).expanduser()
        if candidate.exists() and candidate.is_dir():
            plot_dir = candidate / "plots"
            if plot_dir.is_dir():
                return plot_dir.resolve()
    plot_dir_value = record.get("plot_dir")
    if plot_dir_value:
        plot_dir = Path(str(plot_dir_value)).expanduser()
        if plot_dir.is_dir():
            return plot_dir.resolve()
    return fallback_plot_dir


def load_run_params(plot_dir: Path | None) -> dict[str, Any] | None:
    if plot_dir is None:
        return None
    run_dir = plot_dir.parent if plot_dir.name == "plots" else plot_dir
    candidates = [run_dir / "run_params.json"]
    candidates.extend(sorted(run_dir.glob("run_params_*.json")))
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return None


def load_review_db(source_path: str | Path) -> pd.DataFrame:
    path = Path(source_path).expanduser().resolve()
    with sqlite3.connect(path) as conn:
        df = pd.read_sql_query("SELECT * FROM candidates", conn)
        try:
            reviews = pd.read_sql_query(
                "SELECT candidate_id, interest_score, event_class, review_pass, notes, status, reviewer, updated_at FROM reviews",
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
        return pd.read_parquet(path)
    if kind == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported source kind: {kind}")


def _normalized_id(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except Exception:
            return text
    return text


def _build_lookup(df: pd.DataFrame) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for idx, row in df.iterrows():
        for col in ("candidate_id", "asas_sn_id", "gaia_id"):
            if col not in row:
                continue
            key = _normalized_id(row.get(col))
            if key and key not in lookup:
                lookup[key] = int(idx)
    return lookup


def _candidate_base_id(row: pd.Series, fallback: str) -> str:
    for col in ("candidate_id", "asas_sn_id", "gaia_id"):
        key = _normalized_id(row.get(col))
        if key:
            return key
    return fallback


def _source_label_from_path(source_path: Path) -> str:
    if source_path.suffix.lower() == ".db" and source_path.parent.name == "review":
        return source_path.parent.parent.name
    if source_path.parent.name == "results":
        return source_path.parent.parent.name
    return source_path.stem


@dataclass
class CandidateSourceData:
    source_path: Path
    source_kind: str
    source_label: str
    df: pd.DataFrame
    lookup: dict[str, int]
    default_plot_dir: Path | None

    @property
    def default_candidate_id(self) -> str:
        if self.df.empty:
            return ""
        row = self.df.iloc[0]
        for col in ("candidate_id", "asas_sn_id", "gaia_id"):
            key = _normalized_id(row.get(col))
            if key:
                return key
        return ""


@dataclass
class CombinedCandidateData:
    df: pd.DataFrame
    sources: list[CandidateSourceData]
    key_lookup: dict[str, int]
    id_lookup: dict[str, list[str]]

    @property
    def default_candidate_key(self) -> str:
        if self.df.empty or "candidate_key" not in self.df.columns:
            return ""
        return str(self.df.iloc[0]["candidate_key"])


def load_source_data(
    source_path: str | Path,
    source_kind: str | None = None,
    plot_dir: str | Path | None = None,
    source_label: str | None = None,
) -> CandidateSourceData:
    path = Path(source_path).expanduser().resolve()
    kind = source_kind or infer_source_kind(path)
    label = source_label or _source_label_from_path(path)
    df = load_candidate_source(path, kind).copy()
    df["source_file"] = str(path)
    df["source_label"] = label
    default_plot_dir = infer_plot_dir_from_source(path, plot_dir)
    if default_plot_dir is not None:
        df["plot_dir"] = str(default_plot_dir)

    used_keys: set[str] = set()
    candidate_keys: list[str] = []
    for idx, row in df.iterrows():
        base_id = _candidate_base_id(row, fallback=str(idx))
        key = f"{label}::{base_id}"
        if key in used_keys:
            key = f"{key}::{idx}"
        used_keys.add(key)
        candidate_keys.append(key)
    df["candidate_key"] = candidate_keys

    return CandidateSourceData(
        source_path=path,
        source_kind=kind,
        source_label=label,
        df=df,
        lookup=_build_lookup(df),
        default_plot_dir=default_plot_dir,
    )


def discover_default_sources(repo_root: str | Path | None = None) -> list[Path]:
    root = Path.cwd().resolve() if repo_root is None else Path(repo_root).expanduser().resolve()
    run_dbs = sorted(root.glob("output/runs/*/review/review.db"))
    populated = []
    for path in run_dbs:
        try:
            with sqlite3.connect(path) as conn:
                count = int(conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
        except Exception:
            count = -1
        if count > 0:
            populated.append(path)
    if populated:
        return populated

    for path in (root / "output" / "review" / "review.db", root / "output" / "review" / "standalone.db"):
        if path.exists():
            return [path]

    parquet_patterns = [
        "output/runs/*/results/lc_events_vetted.parquet",
        "output/runs/*/results/lc_events_spectra.parquet",
        "output/runs/*/results/lc_events_neighbors.parquet",
        "output/runs/*/results/lc_events_classified.parquet",
        "output/runs/*/results/lc_events_characterized.parquet",
        "output/runs/*/results/lc_events_filtered.parquet",
    ]
    parquet_matches: list[Path] = []
    for pattern in parquet_patterns:
        parquet_matches.extend(sorted(root.glob(pattern)))
    return parquet_matches


def load_combined_source_data(
    *,
    sources: list[str | Path] | None = None,
    source_kind: str | None = None,
    plot_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> CombinedCandidateData:
    source_paths = [Path(s).expanduser().resolve() for s in (sources or discover_default_sources(repo_root))]
    if not source_paths:
        raise FileNotFoundError("No candidate sources were found. Pass --source explicitly or create run review DBs.")

    loaded_sources = [load_source_data(path, source_kind=source_kind, plot_dir=plot_dir) for path in source_paths]
    frames = [src.df for src in loaded_sources if not src.df.empty]
    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
    else:
        combined = pd.DataFrame()

    key_lookup: dict[str, int] = {}
    id_lookup: dict[str, list[str]] = {}
    if not combined.empty:
        for idx, row in combined.iterrows():
            candidate_key = str(row.get("candidate_key") or "")
            if candidate_key:
                key_lookup[candidate_key] = int(idx)
            for col in ("candidate_id", "asas_sn_id", "gaia_id"):
                key = _normalized_id(row.get(col))
                if not key or not candidate_key:
                    continue
                id_lookup.setdefault(key, []).append(candidate_key)

    return CombinedCandidateData(df=combined, sources=loaded_sources, key_lookup=key_lookup, id_lookup=id_lookup)


def get_candidate_record(source_data: CandidateSourceData, candidate_id: str | None) -> dict[str, Any] | None:
    if source_data.df.empty:
        return None
    key = _normalized_id(candidate_id)
    if not key:
        key = source_data.default_candidate_id
    idx = source_data.lookup.get(key)
    if idx is None:
        return None
    row = source_data.df.loc[idx]
    return row.to_dict() if isinstance(row, pd.Series) else None


def get_candidate_record_by_key(combined: CombinedCandidateData, candidate_key: str | None) -> dict[str, Any] | None:
    if combined.df.empty:
        return None
    key = str(candidate_key or "").strip()
    if not key:
        key = combined.default_candidate_key
    idx = combined.key_lookup.get(key)
    if idx is None:
        return None
    row = combined.df.loc[idx]
    return row.to_dict() if isinstance(row, pd.Series) else None


def find_candidate_key(combined: CombinedCandidateData, search_value: str | None, subset: pd.DataFrame | None = None) -> str | None:
    key = _normalized_id(search_value)
    if not key:
        return None
    if subset is not None and not subset.empty:
        subset_lookup = _build_lookup(subset)
        idx = subset_lookup.get(key)
        if idx is not None:
            row = subset.loc[idx]
            if isinstance(row, pd.Series):
                candidate_key = str(row.get("candidate_key") or "")
                if candidate_key:
                    return candidate_key
    matches = combined.id_lookup.get(key)
    if matches:
        return matches[0]
    if key in combined.key_lookup:
        return key
    return None


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
    out["strong_native_period"] = (
        bool_series(out, "lsp_is_significant")
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


def query_mask(frame: pd.DataFrame, query: str | None = None) -> pd.Series:
    if not query:
        return pd.Series(True, index=frame.index, dtype="bool")
    idx = frame.query(query, engine="python").index
    return pd.Series(frame.index.isin(idx), index=frame.index, dtype="bool")


def cut_summary(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    target_col: str,
    reject_col: str,
    eligible_query: str | None = None,
    name: str = "cut",
) -> pd.Series:
    eligible = query_mask(frame, eligible_query)
    selected = eligible & mask.fillna(False).astype(bool)
    target = eligible & bool_series(frame, target_col)
    reject = eligible & bool_series(frame, reject_col)

    n_selected = int(selected.sum())
    n_target = int(target.sum())
    n_reject = int(reject.sum())
    n_selected_target = int((selected & target).sum())
    n_selected_reject = int((selected & reject).sum())

    purity = n_selected_target / n_selected if n_selected else np.nan
    target_recall = n_selected_target / n_target if n_target else np.nan
    reject_leakage = n_selected_reject / n_selected if n_selected else np.nan
    lift = purity / (n_target / int(eligible.sum())) if n_selected and int(eligible.sum()) and n_target else np.nan

    return pd.Series(
        {
            "name": name,
            "eligible": int(eligible.sum()),
            "selected": n_selected,
            "target_total": n_target,
            "reject_total": n_reject,
            "selected_target": n_selected_target,
            "selected_reject": n_selected_reject,
            "purity": purity,
            "target_recall": target_recall,
            "reject_leakage": reject_leakage,
            "lift_vs_base_rate": lift,
        }
    )


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
