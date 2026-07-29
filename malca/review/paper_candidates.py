"""Shared helpers for March 18 paper-focused candidate analysis notebooks."""

from __future__ import annotations

import json
import math
import sqlite3
import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from malca.config import DEFAULT_OUTPUT_DIR
from malca.products.feature_layers import FEATURE_LAYER_COLUMNS
from malca.plotting.lightcurve_publication import FIG_LC_SINGLE_COL, finalize_publication_figure, plot_lightcurve_panel
from malca.io.notebook_paths import find_repo_root, resolve_local_lightcurve_path
from malca.review.dustycult import load_canonical_cleaned_lightcurve
from malca.review.filter_schema import SIDEBAR_GROUPS


MARCH18_RUN = DEFAULT_OUTPUT_DIR / "runs" / "runs_march18_bundle_all"
REVIEW_DB = MARCH18_RUN / "review" / "review.taxonomy_filled.db"
LIGHTCURVE_DIR = MARCH18_RUN / "bundle_assets" / "lightcurves"
CLASSIFIED_PARQUET = MARCH18_RUN / "results" / "lc_events_classified.parquet"
EXPORT_DIR = DEFAULT_OUTPUT_DIR / "notebooks" / "march18_paper"

DEFAULT_PAPER_BUCKETS = ("Dipper", "LTV", "Microlensing")
PUBLICATION_COHORT_VERSION = "malca-publication-cohort-v1"
PUBLICATION_WORKFLOW_STATUS = frozenset({"reviewed"})
PUBLICATION_DISPOSITIONS = frozenset({"keep"})
PUBLICATION_CONFIDENCE = frozenset({3, 4})
BUCKET_ORDER = (
    "Dipper",
    "Interesting",
    "LTV",
    "Microlensing",
    "Eclipsing binary",
    "Unknown",
)

CAMERA_FIELD_COLUMNS = (
    "asassn_field_key",
    "asassn_fields",
    "asassn_field_count",
    "asassn_field_key_fraction",
    "camera_name_key",
    "camera_names",
    "camera_name_count",
    "camera_name_key_fraction",
)

PAPER_FEATURE_GROUP_NAMES: tuple[str, ...] = (
    "LC Cadence & Coverage",
    "LC Photometric Scatter",
    "LC Variability",
    "LC Period Search",
    "LC Structure Function",
    "LC Harmonics (folded)",
    "LC Stochastic Models",
    "LC Error / SNR / Trend",
    "LTV",
    "LTV Season Diagnostics",
    "LTV Trend Diagnostics",
    "LTV Long-Term Features",
    "LTV Multi-Survey",
    "Multi-Survey Event",
    "Multi-Survey ASAS-SN",
    "Multi-Survey Gaia",
    "Multi-Survey NEOWISE",
    "Multi-Survey TESS",
    "Multi-Survey ZTF",
    "External Coverage",
    "Nuclear Context",
    "Vetting",
    "Classification",
    "Camera / Field",
)


def _columns_from_sidebar_entries(entries: Sequence[tuple[str, str]]) -> list[str]:
    return [str(col) for _kind, col in entries]


def _build_paper_feature_groups() -> dict[str, list[str]]:
    sidebar = dict(SIDEBAR_GROUPS)
    groups: dict[str, list[str]] = {}
    for name in PAPER_FEATURE_GROUP_NAMES:
        if name == "Camera / Field":
            groups[name] = list(CAMERA_FIELD_COLUMNS)
            continue
        if name not in sidebar:
            continue
        groups[name] = _columns_from_sidebar_entries(sidebar[name])
    return groups


PAPER_FEATURE_GROUPS: dict[str, list[str]] = _build_paper_feature_groups()


def paper_feature_columns(group: str | None = None) -> list[str]:
    """Return feature column names for one paper group or all groups."""
    if group is None:
        cols: list[str] = []
        seen: set[str] = set()
        for values in PAPER_FEATURE_GROUPS.values():
            for col in values:
                if col not in seen:
                    seen.add(col)
                    cols.append(col)
        return cols
    return list(PAPER_FEATURE_GROUPS.get(group, ()))


def assign_review_bucket(row: Mapping[str, object]) -> str | None:
    """Map review taxonomy to CMD-style bucket labels."""
    event_class = str(row.get("event_class") or "").strip().lower()
    morph_secondary = str(row.get("morphology_secondary") or "").strip().lower()
    physical_primary = str(row.get("physical_primary") or "").strip().lower()

    if event_class == "dipper":
        return "Dipper"
    if event_class == "ltv":
        return "LTV"
    if event_class == "microlensing":
        return "Microlensing"
    if morph_secondary in {"detached_binary_like", "eclipsing_like"}:
        return "Eclipsing binary"
    if physical_primary == "unknown" or event_class == "unknown":
        return "Unknown"
    return None


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _parse_layer_object(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def flatten_layer_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Merge layer JSON blobs into a flat mapping (review import semantics)."""
    flat = dict(payload)
    for layer in FEATURE_LAYER_COLUMNS:
        layer_values = _parse_layer_object(flat.get(layer))
        for key, value in layer_values.items():
            if key not in flat or _is_missing(flat.get(key)):
                flat[str(key)] = value
    return flat


def load_reviewed_cohort(
    review_db: Path | str | None = None,
    *,
    buckets: Sequence[str] | None = None,
    only_reviewed: bool = True,
    publication_only: bool = False,
    require_confident_classification: bool = False,
) -> pd.DataFrame:
    """Load review rows and attach an explicit, versioned cohort decision.

    ``only_reviewed`` retains the historical notebook behavior.  Publication
    exports should additionally pass ``publication_only=True``; that applies
    the disposition, duplicate, bucket, and optional confidence rules in
    :func:`build_publication_cohort` and keeps an auditable exclusion reason.
    """
    db_path = Path(review_db or REVIEW_DB)
    where = ""
    if only_reviewed:
        where = " WHERE r.workflow_status = 'reviewed' OR r.status = 'reviewed'"
    query = f"""
        SELECT
            c.*,
            r.status AS review_status,
            r.workflow_status,
            r.event_class,
            r.review_pass,
            r.notes,
            r.reviewer,
            r.updated_at,
            r.morphology_primary,
            r.morphology_secondary,
            r.morphology_polarity,
            r.morphology_recurrence,
            r.baseline_behavior,
            r.physical_primary,
            r.physical_secondary,
            r.classification_confidence,
            r.disposition,
            r.duplicate_of,
            r.taxonomy_version
        FROM candidates c
        JOIN reviews r ON r.candidate_id = c.candidate_id
        {where}
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    cohort_buckets = DEFAULT_PAPER_BUCKETS if publication_only and buckets is None else buckets
    df = build_publication_cohort(
        df,
        buckets=cohort_buckets,
        require_confident_classification=require_confident_classification,
    )
    if publication_only:
        df = df.loc[df["publication_selected"]].copy()
    elif buckets:
        bucket_set = {str(b) for b in buckets}
        df = df.loc[df["review_bucket"].isin(bucket_set)].copy()
    return df.reset_index(drop=True)


def _normalized_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def build_publication_cohort(
    frame: pd.DataFrame,
    *,
    buckets: Sequence[str] | None = DEFAULT_PAPER_BUCKETS,
    require_confident_classification: bool = False,
) -> pd.DataFrame:
    """Apply the paper-cohort rules without silently dropping any row.

    The returned table always contains ``publication_selected`` and
    ``publication_exclusion_reason``.  This makes changes in review state or
    taxonomy visible in a diff instead of changing a paper sample invisibly.
    Duplicate or blank candidate identifiers are never publication eligible.
    """
    if "candidate_id" not in frame.columns:
        raise ValueError("Publication cohort requires candidate_id")

    out = frame.reset_index(drop=True).copy()
    out["candidate_id"] = out["candidate_id"].astype("string").str.strip()
    out["review_bucket"] = out.apply(assign_review_bucket, axis=1)
    out["publication_cohort_version"] = PUBLICATION_COHORT_VERSION

    workflow = _normalized_text(
        out.get("workflow_status", pd.Series(pd.NA, index=out.index, dtype="string"))
    )
    if "status" in out.columns:
        fallback = _normalized_text(out["status"])
        workflow = workflow.where(workflow.notna() & workflow.ne(""), fallback)
    disposition = _normalized_text(
        out.get("disposition", pd.Series(pd.NA, index=out.index, dtype="string"))
    )
    confidence = pd.to_numeric(
        out.get("classification_confidence", pd.Series(pd.NA, index=out.index)),
        errors="coerce",
    ).astype("Int64")
    duplicate_of = _normalized_text(
        out.get("duplicate_of", pd.Series(pd.NA, index=out.index, dtype="string"))
    )
    bucket_set = {str(bucket) for bucket in (buckets or ())}

    reasons: list[str] = []
    duplicated = out["candidate_id"].duplicated(keep=False)
    for idx, candidate_id in out["candidate_id"].items():
        row_reasons: list[str] = []
        if pd.isna(candidate_id) or not str(candidate_id).strip():
            row_reasons.append("missing_candidate_id")
        elif bool(duplicated.loc[idx]):
            row_reasons.append("duplicate_candidate_id")
        if workflow.loc[idx] not in PUBLICATION_WORKFLOW_STATUS:
            row_reasons.append("workflow_not_reviewed")
        if disposition.loc[idx] not in PUBLICATION_DISPOSITIONS:
            row_reasons.append("disposition_not_keep")
        if pd.notna(duplicate_of.loc[idx]) and str(duplicate_of.loc[idx]).strip():
            row_reasons.append("marked_duplicate")
        bucket = out.at[idx, "review_bucket"]
        if bucket is None or (bucket_set and bucket not in bucket_set):
            row_reasons.append("class_outside_publication_cohort")
        if require_confident_classification and confidence.loc[idx] not in PUBLICATION_CONFIDENCE:
            row_reasons.append("classification_confidence_below_3")
        reasons.append(";".join(row_reasons))

    out["publication_exclusion_reason"] = pd.Series(reasons, index=out.index, dtype="string")
    out["publication_selected"] = out["publication_exclusion_reason"].eq("")
    return out


def audit_flattened_vs_layers(df: pd.DataFrame) -> pd.DataFrame:
    """Report JSON-layer keys missing from flattened SQL columns."""
    work = df.reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for layer in FEATURE_LAYER_COLUMNS:
        for idx, raw in work[layer].items() if layer in work.columns else []:
            layer_obj = _parse_layer_object(raw)
            if not layer_obj:
                continue
            candidate_id = str(work.at[idx, "candidate_id"]) if "candidate_id" in work.columns else str(idx)
            flat_row = work.loc[idx]
            for key, json_value in layer_obj.items():
                if _is_missing(json_value):
                    continue
                flat_value = flat_row.get(key) if key in flat_row.index else None
                if key not in flat_row.index:
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "layer": layer,
                            "key": key,
                            "issue": "json_only",
                            "json_value": json_value,
                            "flat_value": None,
                        }
                    )
                elif _is_missing(flat_value) and not _is_missing(json_value):
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "layer": layer,
                            "key": key,
                            "issue": "flat_missing",
                            "json_value": json_value,
                            "flat_value": flat_value,
                        }
                    )
                elif not _values_equivalent(flat_value, json_value):
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "layer": layer,
                            "key": key,
                            "issue": "value_mismatch",
                            "json_value": json_value,
                            "flat_value": flat_value,
                        }
                    )
    return pd.DataFrame(rows)


def _values_equivalent(left: object, right: object) -> bool:
    """Compare audit values with tolerance for equivalent numeric encodings."""
    try:
        left_numeric = float(left)
        right_numeric = float(right)
        if math.isfinite(left_numeric) and math.isfinite(right_numeric):
            return bool(np.isclose(left_numeric, right_numeric, rtol=1e-9, atol=1e-12))
    except (TypeError, ValueError):
        pass
    return str(left).strip() == str(right).strip()


def feature_missingness_by_bucket(
    df: pd.DataFrame,
    *,
    bucket_col: str = "review_bucket",
    groups: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Compute per-bucket non-null fractions for each feature group."""
    groups = dict(groups or PAPER_FEATURE_GROUPS)
    rows: list[dict[str, object]] = []
    for group_name, columns in groups.items():
        present_cols = [col for col in columns if col in df.columns]
        if not present_cols:
            continue
        for bucket, sub in df.groupby(bucket_col, dropna=False):
            n = len(sub)
            for col in present_cols:
                present = sub[col].notna()
                if pd.api.types.is_object_dtype(sub[col]) or pd.api.types.is_string_dtype(sub[col]):
                    present &= sub[col].astype("string").str.strip().ne("").fillna(False)
                non_null = int(present.sum())
                rows.append(
                    {
                        "feature_group": group_name,
                        "column": col,
                        bucket_col: bucket,
                        "n": n,
                        "non_null": non_null,
                        "non_null_fraction": non_null / max(n, 1),
                    }
                )
    return pd.DataFrame(rows)


def discover_numeric_columns(
    df: pd.DataFrame,
    *,
    exclude: Iterable[str] = (),
    min_non_null: int = 3,
) -> list[str]:
    """Auto-discover numeric analysis columns."""
    excluded = set(exclude)
    numeric_cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() >= min_non_null:
            numeric_cols.append(col)
    return numeric_cols


def grouped_numeric_summary(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    group_col: str = "review_bucket",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in features:
        if feature not in df.columns:
            continue
        for label, group in df.groupby(group_col, dropna=False):
            x = pd.to_numeric(group[feature], errors="coerce").dropna()
            rows.append(
                {
                    "feature": feature,
                    group_col: label,
                    "n": len(x),
                    "missing_frac": 1 - len(x) / max(len(group), 1),
                    "mean": x.mean() if len(x) else np.nan,
                    "std": x.std(ddof=1) if len(x) > 1 else np.nan,
                    "median": x.median() if len(x) else np.nan,
                    "q25": x.quantile(0.25) if len(x) else np.nan,
                    "q75": x.quantile(0.75) if len(x) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def mwu_separability_table(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    group_col: str = "review_bucket",
    reference_group: str,
    compare_groups: Sequence[str] | None = None,
) -> pd.DataFrame:
    from scipy import stats

    compare_groups = list(compare_groups or sorted(df[group_col].dropna().unique()))
    rows: list[dict[str, object]] = []
    ref = df.loc[df[group_col].eq(reference_group)]
    for feature in features:
        if feature not in df.columns:
            continue
        ref_x = pd.to_numeric(ref[feature], errors="coerce").dropna()
        if len(ref_x) < 2:
            continue
        for label in compare_groups:
            if label == reference_group:
                continue
            other = df.loc[df[group_col].eq(label)]
            other_x = pd.to_numeric(other[feature], errors="coerce").dropna()
            if len(other_x) < 2:
                continue
            try:
                result = stats.mannwhitneyu(ref_x, other_x, alternative="two-sided")
                pvalue = float(result.pvalue)
                rank_biserial = 2.0 * float(result.statistic) / (len(ref_x) * len(other_x)) - 1.0
            except Exception:
                pvalue = np.nan
                rank_biserial = np.nan
            rows.append(
                {
                    "feature": feature,
                    "reference_group": reference_group,
                    "compare_group": label,
                    "reference_n": int(len(ref_x)),
                    "compare_n": int(len(other_x)),
                    "reference_median": float(ref_x.median()),
                    "compare_median": float(other_x.median()),
                    "median_difference": float(ref_x.median() - other_x.median()),
                    "rank_biserial": rank_biserial,
                    "mwu_pvalue": pvalue,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["mwu_fdr_qvalue"] = _benjamini_hochberg(result["mwu_pvalue"])
    result["inference_scope"] = "exploratory_unpaired_feature_screen"
    return result


def _benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """Return Benjamini-Hochberg adjusted p-values, preserving missing rows."""
    numeric = pd.to_numeric(pvalues, errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric) & numeric.between(0.0, 1.0)
    adjusted = pd.Series(np.nan, index=pvalues.index, dtype="float64")
    if not bool(valid.any()):
        return adjusted
    ordered = numeric.loc[valid].sort_values()
    n_tests = len(ordered)
    ranks = np.arange(1, n_tests + 1, dtype=float)
    raw = ordered.to_numpy(dtype=float) * n_tests / ranks
    monotonic = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted.loc[ordered.index] = np.clip(monotonic, 0.0, 1.0)
    return adjusted


def zscore_median_profile(
    summary_long: pd.DataFrame,
    *,
    group_col: str = "review_bucket",
    value_col: str = "median",
) -> pd.DataFrame:
    """Z-score feature medians across all reviewed rows."""
    pivot = summary_long.pivot_table(index="feature", columns=group_col, values=value_col, aggfunc="first")
    global_median = summary_long.groupby("feature")[value_col].median()
    global_std = summary_long.groupby("feature")[value_col].std(ddof=0).replace(0, np.nan)
    z = pivot.sub(global_median, axis=0).div(global_std, axis=0)
    return z


def _candidate_payload(row: pd.Series) -> dict[str, object]:
    payload: dict[str, object] = row.to_dict()
    if "payload_json" in row.index and isinstance(row.get("payload_json"), str):
        try:
            parsed = json.loads(row["payload_json"])
            if isinstance(parsed, dict):
                payload.update(parsed)
        except json.JSONDecodeError:
            pass
    return payload


def resolve_bucket_label(analysis_df: pd.DataFrame, label: str) -> str:
    """Map an event_class or review_bucket label to a review bucket name."""
    label = str(label)
    if "review_bucket" in analysis_df.columns:
        buckets = set(analysis_df["review_bucket"].dropna().astype(str))
        if label in buckets:
            return label
    mapped = assign_review_bucket({"event_class": label})
    if mapped:
        return mapped
    if "review_bucket" in analysis_df.columns:
        buckets = set(analysis_df["review_bucket"].dropna().astype(str))
        for bucket in buckets:
            if bucket.lower() == label.lower():
                return bucket
    return label


def plot_candidate_lightcurve(
    analysis_df: pd.DataFrame,
    candidate_id: str,
    *,
    run_dir: Path | str | None = None,
    color_by_camera: bool = True,
    max_legend: int = 12,
    show: bool = True,
    ax=None,
    figsize: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Plot a cleaned ASAS-SN light curve for one reviewed candidate."""
    rows = analysis_df.loc[analysis_df["candidate_id"].astype(str).eq(str(candidate_id))]
    if rows.empty:
        raise KeyError(f"candidate_id not found: {candidate_id}")
    row = rows.iloc[0]
    run_path = Path(run_dir or MARCH18_RUN)
    payload = _candidate_payload(row)
    lc_path = resolve_local_lightcurve_path(row.get("lc_path"), run_dir=run_path)
    df, _resolved = load_canonical_cleaned_lightcurve(payload, lc_path=lc_path, plot_dir=run_path)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize or FIG_LC_SINGLE_COL)
    camera_col = "camera#" if "camera#" in df.columns else ("camera" if "camera" in df.columns else None)
    camera_count = df[camera_col].nunique(dropna=True) if camera_col else 0
    plot_lightcurve_panel(
        ax,
        df,
        time_col="JD" if "JD" in df.columns else "time",
        value_col="mag",
        error_col="mag_err" if "mag_err" in df.columns else ("error" if "error" in df.columns else None),
        camera_col=camera_col if color_by_camera and camera_col else None,
        group_by="camera" if color_by_camera and camera_col else "none",
        legend="auto" if color_by_camera and camera_count <= max_legend else "none",
        marker_size=3.0,
        time_offset="none",
        xlabel="JD",
        ylabel="mag",
        title=(
            f"{candidate_id} | bucket={row.get('review_bucket')} | class={row.get('event_class')} | "
            f"confidence={row.get('classification_confidence')}"
        ),
    )
    if show:
        finalize_publication_figure(plt.gcf())
        plt.show()
    return df


def plot_class_examples(
    analysis_df: pd.DataFrame,
    label: str,
    n: int = 4,
    *,
    sort_by: str = "classification_confidence",
    run_dir: Path | str | None = None,
) -> list[pd.DataFrame]:
    """Plot example light curves for a review bucket or event_class label."""
    bucket = resolve_bucket_label(analysis_df, label)
    data = analysis_df.loc[analysis_df["review_bucket"].eq(bucket)].copy()
    if data.empty:
        print(f"No rows for bucket={bucket}")
        return []
    ascending = sort_by not in {"classification_confidence", "updated_at", "ltv_dispersion", "ltv_max_diff"}
    data = data.sort_values([sort_by, "candidate_id"], ascending=[not ascending, True]).head(n)
    outputs: list[pd.DataFrame] = []
    for cid in data["candidate_id"].astype(str):
        try:
            outputs.append(
                plot_candidate_lightcurve(analysis_df, cid, run_dir=run_dir, show=True)
            )
        except Exception as exc:
            print(f"{cid}: {exc}")
    return outputs


def export_paper_tables(
    df: pd.DataFrame,
    missingness: pd.DataFrame,
    *,
    export_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Write candidate and missingness tables for paper workflows."""
    out_dir = Path(export_dir or EXPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / "candidate_table.parquet"
    missingness_path = out_dir / "feature_missingness.csv"
    inventory_path = out_dir / "feature_group_inventory.csv"
    cohort = (
        df
        if "publication_selected" in df.columns
        else build_publication_cohort(df, buckets=None)
    )
    selected = cohort.loc[cohort["publication_selected"]].copy()
    selected.to_parquet(candidate_path, index=False)
    missingness.to_csv(missingness_path, index=False)
    inventory = pd.DataFrame(
        [{"feature_group": name, "n_columns": len(cols)} for name, cols in PAPER_FEATURE_GROUPS.items()]
    )
    inventory.to_csv(inventory_path, index=False)
    cohort_path = out_dir / "publication_cohort_decisions.csv"
    exclusion_path = out_dir / "publication_cohort_exclusions.csv"
    manifest_path = out_dir / "publication_cohort_manifest.json"
    cohort.to_csv(cohort_path, index=False)
    cohort.loc[~cohort["publication_selected"]].to_csv(exclusion_path, index=False)
    selected_ids = sorted(selected["candidate_id"].astype(str).tolist())
    manifest = {
        "publication_cohort_version": PUBLICATION_COHORT_VERSION,
        "n_input_rows": int(len(cohort)),
        "n_selected_rows": int(len(selected)),
        "n_excluded_rows": int((~cohort["publication_selected"]).sum()),
        "selected_candidate_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")
        ).hexdigest(),
        "selection_rules": {
            "workflow_status": sorted(PUBLICATION_WORKFLOW_STATUS),
            "disposition": sorted(PUBLICATION_DISPOSITIONS),
            "duplicate_rows_excluded": True,
            "marked_duplicates_excluded": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "candidate_table": candidate_path,
        "feature_missingness": missingness_path,
        "feature_group_inventory": inventory_path,
        "publication_cohort_decisions": cohort_path,
        "publication_cohort_exclusions": exclusion_path,
        "publication_cohort_manifest": manifest_path,
    }


def resolve_paper_paths(repo_root: Path | str | None = None) -> dict[str, Path]:
    """Return canonical March 18 paper paths relative to repo root."""
    root = find_repo_root(repo_root)
    return {
        "repo_root": root,
        "march18_run": root / MARCH18_RUN,
        "review_db": root / REVIEW_DB,
        "lightcurve_dir": root / LIGHTCURVE_DIR,
        "classified_parquet": root / CLASSIFIED_PARQUET,
        "export_dir": root / EXPORT_DIR,
    }
