"""LightGBM training helpers for MALCA review databases.

The current-schema loader is intentionally strict: it requires the taxonomy
review columns and does not fall back to old review fields. The March 18 legacy
loader is the only place where old review-schema compatibility is applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import re
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from malca.lightcurve_io import load_lightcurve_df
from malca.review.taxonomy import legacy_review_to_taxonomy


CURRENT_SCHEMA_REQUIRED_REVIEW_COLUMNS = (
    "candidate_id",
    "workflow_status",
    "disposition",
    "morphology_primary",
    "morphology_secondary",
    "physical_primary",
    "physical_secondary",
    "classification_confidence",
)

LEGACY_SCHEMA_REQUIRED_REVIEW_COLUMNS = (
    "candidate_id",
    "interest_score",
    "event_class",
    "review_pass",
    "status",
)

CURRENT_TAXONOMY_TARGET_COLUMNS = (
    "morphology_primary",
    "physical_primary",
)

LEGACY_MARCH18_TARGET_COLUMNS = ("event_class",)

RECOMPUTE_SURVIVAL_TARGET_COLUMNS = ("recompute_survived",)

DEFAULT_RECOMPUTE_SURVIVAL_OLD_VETTED_PATHS: tuple[Path, ...] = (
    Path("output/runs/output_bundle_12_12.5_home_bundle_12_12.5/results/lc_events_vetted.parquet"),
    Path("output/runs/output_bundle_12.5_13_home_bundle_12.5_13/results/lc_events_vetted.parquet"),
    Path("output/runs/output_bundle_13_13.5_bundle_13_13.5/results/lc_events_vetted.parquet"),
    Path("output/runs/output_bundle_13.5_14_bundle_13.5_14/results/lc_events_vetted.parquet"),
)

DEFAULT_RECOMPUTE_SURVIVAL_RECOMPUTED_PATH = Path(
    "output/runs/runs_march18_bundle_all/results/lc_events_vetted.parquet"
)

DEFAULT_RECOMPUTE_SURVIVAL_MAG_BINS = ("12_12.5", "12.5_13", "13_13.5", "13.5_14")

REVIEW_METADATA_COLUMNS = {
    "interest_score",
    "review_pass",
    "notes",
    "status",
    "reviewer",
    "updated_at",
    "workflow_status",
    "disposition",
    "classification_confidence",
    "priority_tags_json",
    "evidence_flags_json",
    "model_tags_json",
    "duplicate_of",
    "known_object_id",
    "known_object_source",
    "taxonomy_version",
    "legacy_review_json",
    "legacy_schema_source",
}

TAXONOMY_LABEL_COLUMNS = {
    "event_class",
    "morphology_primary",
    "morphology_secondary",
    "morphology_polarity",
    "morphology_recurrence",
    "baseline_behavior",
    "physical_primary",
    "physical_secondary",
    "physical_family",
    "physical_subclass",
}

IDENTIFIER_AND_PATH_COLUMNS = {
    "candidate_id",
    "source_id",
    "asas_sn_id",
    "gaia_id",
    "tmass_id",
    "allwise_id",
    "path",
    "lc_path",
    "source_path",
    "resolved_lc_path",
    "index_csv",
    "lc_dir",
    "dat_path",
}

JSON_PAYLOAD_COLUMNS = {
    "payload_json",
    "camera_ids",
    "period_sources",
    "period_source_periods",
    "model_tags_json",
    "priority_tags_json",
    "evidence_flags_json",
    "legacy_review_json",
}

NON_FEATURE_UTILITY_COLUMNS = {
    "schema_mode",
    "timescale",
    "recompute_status",
}

DEFAULT_DROP_COLUMNS = (
    REVIEW_METADATA_COLUMNS
    | TAXONOMY_LABEL_COLUMNS
    | IDENTIFIER_AND_PATH_COLUMNS
    | JSON_PAYLOAD_COLUMNS
    | NON_FEATURE_UTILITY_COLUMNS
)

INTEGER_FLOAT_RE = re.compile(r"^([+-]?\d+)\.0+$")
MAG_BIN_RE = re.compile(r"/(\d+(?:\.\d+)?_\d+(?:\.\d+)?)(?:/|$)")


@dataclass(frozen=True)
class TrainingConfig:
    """Small set of LightGBM knobs used by both notebooks."""

    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 63
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 20
    max_categorical_cardinality: int = 50
    min_class_count: int = 2
    class_weight: str | None = "balanced"
    n_jobs: int = 1


@dataclass
class TargetTrainingResult:
    target_column: str
    n_rows: int
    n_features: int
    class_counts: dict[str, int]
    feature_columns: list[str]
    categorical_features: list[str]
    categorical_maps: dict[str, list[str]]
    label_classes: list[str]
    cv_metrics: pd.DataFrame
    holdout_metrics: dict[str, Any]
    model: lgb.LGBMClassifier | None = field(repr=False, default=None)

    def metadata(self) -> dict[str, Any]:
        return {
            "target_column": self.target_column,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "class_counts": self.class_counts,
            "feature_columns": self.feature_columns,
            "categorical_features": self.categorical_features,
            "categorical_maps": self.categorical_maps,
            "label_classes": self.label_classes,
            "cv_metrics": self.cv_metrics.to_dict(orient="records"),
            "holdout_metrics": self.holdout_metrics,
        }


def _sqlite_ro(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path).expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"Review DB not found: {db_path}")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _read_candidates_and_reviews(db_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with _sqlite_ro(db_path) as conn:
        if not _table_exists(conn, "candidates"):
            raise ValueError(f"{db_path} does not contain a candidates table")
        candidates = pd.read_sql_query("SELECT * FROM candidates", conn)
        if _table_exists(conn, "reviews"):
            reviews = pd.read_sql_query("SELECT * FROM reviews", conn)
        else:
            reviews = pd.DataFrame()
    return candidates, reviews


def _merge_candidates_reviews(candidates: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    left = candidates.copy()
    left["candidate_id"] = left["candidate_id"].astype(str)
    if reviews.empty:
        return left
    right = reviews.copy()
    right["candidate_id"] = right["candidate_id"].astype(str)
    merged = left.merge(right, on="candidate_id", how="inner", suffixes=("", "_review"))
    return merged.loc[:, ~merged.columns.duplicated()].copy()


def _missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col not in df.columns]


def _nonempty_text(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def _empty_text_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def _reviewed_current_mask(df: pd.DataFrame) -> pd.Series:
    status = df["workflow_status"].fillna("").astype(str).str.strip()
    return status.ne("") & status.ne("unreviewed")


def _reviewed_legacy_mask(df: pd.DataFrame) -> pd.Series:
    status = df["status"].fillna("").astype(str).str.strip()
    return status.ne("") & status.ne("unreviewed")


def _read_candidate_product(path: str | Path) -> pd.DataFrame:
    product_path = Path(path).expanduser()
    if not product_path.exists():
        raise FileNotFoundError(f"Candidate product not found: {product_path}")
    suffix = product_path.suffix.lower()
    if suffix == ".parquet" or product_path.is_dir():
        table = pd.read_parquet(product_path)
    elif suffix == ".csv":
        table = pd.read_csv(product_path, dtype=str, keep_default_na=False)
    else:
        raise ValueError(f"Unsupported candidate product type: {product_path}")
    return table.to_frame() if isinstance(table, pd.Series) else table


def _normalize_recompute_id(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "null"}:
        return ""
    match = INTEGER_FLOAT_RE.match(text)
    if match:
        text = match.group(1)
    return text.casefold()


def _path_stem(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "null"}:
        return ""
    return Path(text).stem.strip()


def _normalize_mag_bin(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "null"}:
        return ""
    text = text.replace("-", "_")
    parts = [part.strip() for part in text.split("_")]
    normalized_parts = []
    for part in parts:
        match = INTEGER_FLOAT_RE.match(part)
        normalized_parts.append(match.group(1) if match else part)
    return "_".join(normalized_parts)


def _mag_bin_from_path(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    match = MAG_BIN_RE.search(text)
    return _normalize_mag_bin(match.group(1)) if match else ""


def _id_series_from_asas_or_path(df: pd.DataFrame) -> pd.Series:
    if "asas_sn_id" in df.columns:
        ids = df["asas_sn_id"].map(_normalize_recompute_id)
    else:
        ids = pd.Series("", index=df.index, dtype="object")
    missing = ids.eq("")
    if "path" in df.columns and bool(missing.any()):
        path_ids = df.loc[missing, "path"].map(_path_stem).map(_normalize_recompute_id)
        ids = ids.where(~missing, path_ids)
    return ids.astype(str)


def _ensure_recompute_identity_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    out = df.copy()
    ids = _id_series_from_asas_or_path(out)
    if bool(ids.eq("").any()):
        missing_count = int(ids.eq("").sum())
        raise ValueError(
            f"{missing_count} candidate rows are missing both asas_sn_id and path-derived IDs"
        )

    if "asas_sn_id" not in out.columns:
        out["asas_sn_id"] = ids
    else:
        asas = out["asas_sn_id"].astype("object")
        missing_asas = asas.isna() | asas.astype(str).str.strip().eq("")
        out["asas_sn_id"] = asas.where(~missing_asas, ids)

    if "candidate_id" not in out.columns:
        out["candidate_id"] = out["asas_sn_id"].astype(str)
    else:
        candidate_id = out["candidate_id"].astype("object")
        missing_candidate_id = candidate_id.isna() | candidate_id.astype(str).str.strip().eq("")
        out["candidate_id"] = candidate_id.where(
            ~missing_candidate_id,
            out["asas_sn_id"].astype(str),
        )

    if "mag_bin" in out.columns:
        out["mag_bin"] = out["mag_bin"].map(_normalize_mag_bin)
    elif "path" in out.columns:
        out["mag_bin"] = out["path"].map(_mag_bin_from_path)
    else:
        out["mag_bin"] = ""

    return out, ids


def _normalized_mag_bin_set(mag_bins: Sequence[str] | None) -> set[str] | None:
    if mag_bins is None:
        return None
    normalized = [_normalize_mag_bin(value) for value in mag_bins]
    return {value for value in normalized if value}


def load_recompute_survival_training_table(
    old_vetted_paths: Sequence[str | Path] | None = None,
    recomputed_candidates_path: str | Path = DEFAULT_RECOMPUTE_SURVIVAL_RECOMPUTED_PATH,
    *,
    mag_bins: Sequence[str] | None = DEFAULT_RECOMPUTE_SURVIVAL_MAG_BINS,
) -> pd.DataFrame:
    """Load old candidate features labeled by March 18 recompute survival.

    The March 18 recomputed table is used only to derive the survived ID set.
    Its feature columns are intentionally not merged into the returned table.
    """

    feature_paths = tuple(old_vetted_paths or DEFAULT_RECOMPUTE_SURVIVAL_OLD_VETTED_PATHS)
    if not feature_paths:
        raise ValueError("At least one old vetted feature path is required")

    old_frames = [_read_candidate_product(path) for path in feature_paths]
    old = pd.concat(old_frames, ignore_index=True, sort=False)
    old, old_ids = _ensure_recompute_identity_columns(old)

    mag_bin_set = _normalized_mag_bin_set(mag_bins)
    if mag_bin_set is not None:
        old = old.loc[old["mag_bin"].isin(mag_bin_set)].copy()
        old_ids = old_ids.loc[old.index]

    duplicate_ids = old_ids[old_ids.duplicated(keep=False)]
    if not duplicate_ids.empty:
        examples = ", ".join(sorted(duplicate_ids.unique())[:5])
        raise ValueError(f"Old vetted candidate IDs are not unique; examples: {examples}")

    recomputed = _read_candidate_product(recomputed_candidates_path)
    recomputed, recomputed_ids = _ensure_recompute_identity_columns(recomputed)
    if mag_bin_set is not None:
        recomputed = recomputed.loc[recomputed["mag_bin"].isin(mag_bin_set)].copy()
        recomputed_ids = recomputed_ids.loc[recomputed.index]

    survived_ids = {value for value in recomputed_ids.tolist() if value}
    out = old.copy()
    survived = old_ids.isin(survived_ids)
    out["recompute_survived"] = survived.astype("int8")
    out["recompute_status"] = np.where(survived, "survived_recompute", "fell_away")
    out["timescale"] = "stv"
    out["schema_mode"] = "recompute_survival"
    return out.reset_index(drop=True)


def load_current_schema_training_table(
    db_path: str | Path,
    *,
    flat_lightcurve_dir: str | Path | None = None,
    include_lightcurve_features: bool = True,
    lightcurve_feature_cache: str | Path | None = None,
    only_reviewed: bool = True,
    max_lightcurves: int | None = None,
) -> pd.DataFrame:
    """Load reviewed rows from a current taxonomy review DB.

    This function deliberately refuses old-schema-only review tables. It does
    not derive taxonomy labels from ``event_class`` or rename old physical
    taxonomy columns.
    """

    candidates, reviews = _read_candidates_and_reviews(db_path)
    missing = _missing_columns(reviews, CURRENT_SCHEMA_REQUIRED_REVIEW_COLUMNS)
    if missing:
        raise ValueError(
            "Current-schema notebook requires review taxonomy columns; "
            f"missing: {missing}"
        )

    table = _merge_candidates_reviews(candidates, reviews)
    if only_reviewed and not table.empty:
        table = table.loc[_reviewed_current_mask(table)].copy()

    table["schema_mode"] = "current"
    if include_lightcurve_features:
        table = append_lightcurve_features(
            table,
            flat_lightcurve_dir=flat_lightcurve_dir,
            cache_path=lightcurve_feature_cache,
            max_lightcurves=max_lightcurves,
        )
    return table


def _legacy_mapping_frame(reviews: pd.DataFrame) -> pd.DataFrame:
    mapped_rows: list[dict[str, Any]] = []
    for row in reviews.to_dict(orient="records"):
        mapped = legacy_review_to_taxonomy(row)
        mapped_rows.append(mapped)
    mapped_df = pd.DataFrame(mapped_rows, index=reviews.index)
    return mapped_df


def load_legacy_march18_training_table(
    db_path: str | Path,
    *,
    flat_lightcurve_dir: str | Path | None = None,
    include_lightcurve_features: bool = True,
    lightcurve_feature_cache: str | Path | None = None,
    only_reviewed: bool = True,
    derive_taxonomy_from_event_class: bool = True,
    max_lightcurves: int | None = None,
) -> pd.DataFrame:
    """Load the March 18 review DB format and preserve its old labels."""

    candidates, reviews = _read_candidates_and_reviews(db_path)
    missing = _missing_columns(reviews, LEGACY_SCHEMA_REQUIRED_REVIEW_COLUMNS)
    if missing:
        raise ValueError(f"Legacy review table is missing required old-schema columns: {missing}")

    reviews = reviews.copy()
    if "workflow_status" not in reviews.columns:
        reviews["workflow_status"] = pd.NA
    if "disposition" not in reviews.columns:
        reviews["disposition"] = pd.NA
    if "physical_primary" not in reviews.columns:
        reviews["physical_primary"] = pd.NA
    if "physical_secondary" not in reviews.columns:
        reviews["physical_secondary"] = pd.NA
    if "morphology_primary" not in reviews.columns:
        reviews["morphology_primary"] = pd.NA
    if "morphology_secondary" not in reviews.columns:
        reviews["morphology_secondary"] = pd.NA

    reviews["workflow_status"] = reviews["workflow_status"].where(
        _nonempty_text(reviews["workflow_status"]),
        reviews["status"],
    )

    if "physical_family" in reviews.columns:
        reviews["physical_primary"] = reviews["physical_primary"].where(
            _nonempty_text(reviews["physical_primary"]),
            reviews["physical_family"],
        )
    if "physical_subclass" in reviews.columns:
        reviews["physical_secondary"] = reviews["physical_secondary"].where(
            _nonempty_text(reviews["physical_secondary"]),
            reviews["physical_subclass"],
        )

    if derive_taxonomy_from_event_class:
        mapped = _legacy_mapping_frame(reviews)
        for col in (
            "disposition",
            "morphology_primary",
            "morphology_secondary",
            "physical_primary",
            "physical_secondary",
        ):
            if col in mapped.columns:
                reviews[col] = reviews[col].where(_nonempty_text(reviews[col]), mapped[col])
        reviews["legacy_review_json"] = reviews.get("legacy_review_json", pd.Series(index=reviews.index, dtype=object))
        reviews["legacy_review_json"] = reviews["legacy_review_json"].where(
            _nonempty_text(reviews["legacy_review_json"]),
            mapped.get("legacy_review_json", pd.Series(index=reviews.index, dtype=object)),
        )

    table = _merge_candidates_reviews(candidates, reviews)
    if only_reviewed and not table.empty:
        table = table.loc[_reviewed_legacy_mask(table)].copy()

    table["legacy_schema_source"] = "march18"
    table["schema_mode"] = "legacy_march18"
    if include_lightcurve_features:
        table = append_lightcurve_features(
            table,
            flat_lightcurve_dir=flat_lightcurve_dir,
            cache_path=lightcurve_feature_cache,
            max_lightcurves=max_lightcurves,
        )
    return table


def _candidate_id_values(row: pd.Series) -> list[str]:
    out: list[str] = []
    for col in ("candidate_id", "asas_sn_id", "source_id"):
        if col in row and pd.notna(row[col]):
            value = str(row[col]).strip()
            if value and value not in out:
                out.append(value)
    return out


def resolve_lightcurve_path(
    row: pd.Series,
    *,
    flat_lightcurve_dir: str | Path | None = None,
    ext_order: Sequence[str] = (".dat3", ".dat2", ".dat", ".csv"),
) -> Path | None:
    """Resolve a candidate light-curve path, preferring local flat exports."""

    for col in ("lc_path", "path", "dat_path"):
        if col in row and pd.notna(row[col]):
            candidate = Path(str(row[col])).expanduser()
            if candidate.exists() and candidate.suffix.lower() in ext_order:
                return candidate

    if flat_lightcurve_dir is None:
        return None
    flat_dir = Path(flat_lightcurve_dir).expanduser()
    if not flat_dir.exists():
        return None
    for stem in _candidate_id_values(row):
        for ext in ext_order:
            candidate = flat_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def _nan_feature_row(candidate_id: str, path: Path | None) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "resolved_lc_path": str(path) if path is not None else None,
        "lc_file_exists": bool(path is not None and path.exists()),
        "lc_n_points": np.nan,
        "lc_n_good": np.nan,
        "lc_n_cameras": np.nan,
        "lc_n_bands": np.nan,
        "lc_time_span_days": np.nan,
        "lc_mag_median": np.nan,
        "lc_mag_std": np.nan,
        "lc_mag_mad": np.nan,
        "lc_mag_iqr": np.nan,
        "lc_mag_p05": np.nan,
        "lc_mag_p95": np.nan,
        "lc_mag_p95_minus_p05": np.nan,
        "lc_error_median": np.nan,
        "lc_g_n_points": np.nan,
        "lc_v_n_points": np.nan,
        "lc_g_mag_median": np.nan,
        "lc_v_mag_median": np.nan,
        "lc_v_minus_g_median": np.nan,
    }


def summarize_lightcurve(path: str | Path) -> dict[str, Any]:
    """Compute compact numeric features from one light curve file."""

    lc_path = Path(path).expanduser()
    df = load_lightcurve_df(lc_path, apply_quality=False)
    if df.empty or "mag" not in df.columns:
        return _nan_feature_row(lc_path.stem, lc_path)

    mag = pd.to_numeric(df["mag"], errors="coerce")
    jd = pd.to_numeric(df.get("jd"), errors="coerce")
    err = pd.to_numeric(df.get("mag_err"), errors="coerce")
    good = df.get("is_good", pd.Series(False, index=df.index)).astype(bool)
    finite_mag = mag[np.isfinite(mag)]
    median = float(finite_mag.median()) if not finite_mag.empty else np.nan
    mad = float((finite_mag - median).abs().median()) if not finite_mag.empty else np.nan
    q05 = float(finite_mag.quantile(0.05)) if not finite_mag.empty else np.nan
    q25 = float(finite_mag.quantile(0.25)) if not finite_mag.empty else np.nan
    q75 = float(finite_mag.quantile(0.75)) if not finite_mag.empty else np.nan
    q95 = float(finite_mag.quantile(0.95)) if not finite_mag.empty else np.nan

    out = _nan_feature_row(lc_path.stem, lc_path)
    out.update(
        {
            "lc_n_points": int(len(df)),
            "lc_n_good": int(good.sum()) if len(good) else np.nan,
            "lc_n_cameras": int(df["camera"].nunique(dropna=True)) if "camera" in df.columns else np.nan,
            "lc_n_bands": int(df["band"].nunique(dropna=True)) if "band" in df.columns else np.nan,
            "lc_time_span_days": float(jd.max() - jd.min()) if jd.notna().any() else np.nan,
            "lc_mag_median": median,
            "lc_mag_std": float(finite_mag.std(ddof=0)) if not finite_mag.empty else np.nan,
            "lc_mag_mad": mad,
            "lc_mag_iqr": float(q75 - q25) if np.isfinite(q75) and np.isfinite(q25) else np.nan,
            "lc_mag_p05": q05,
            "lc_mag_p95": q95,
            "lc_mag_p95_minus_p05": float(q95 - q05) if np.isfinite(q95) and np.isfinite(q05) else np.nan,
            "lc_error_median": float(err.median()) if err.notna().any() else np.nan,
        }
    )

    if "band" in df.columns:
        bands = df["band"].astype(str).str.strip().str.lower()
        for value, name in (("g", "g"), ("v", "v")):
            band_mag = mag.loc[bands == value].dropna()
            out[f"lc_{name}_n_points"] = int(len(band_mag))
            out[f"lc_{name}_mag_median"] = float(band_mag.median()) if not band_mag.empty else np.nan
        if np.isfinite(out["lc_v_mag_median"]) and np.isfinite(out["lc_g_mag_median"]):
            out["lc_v_minus_g_median"] = float(out["lc_v_mag_median"] - out["lc_g_mag_median"])

    return out


def compute_lightcurve_feature_frame(
    table: pd.DataFrame,
    *,
    flat_lightcurve_dir: str | Path | None = None,
    max_lightcurves: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source = table
    if max_lightcurves is not None:
        source = source.head(max_lightcurves)
    for _, row in source.iterrows():
        candidate_id = str(row.get("candidate_id", "")).strip()
        path = resolve_lightcurve_path(row, flat_lightcurve_dir=flat_lightcurve_dir)
        if path is None:
            rows.append(_nan_feature_row(candidate_id, None))
            continue
        try:
            features = summarize_lightcurve(path)
            features["candidate_id"] = candidate_id
            rows.append(features)
        except Exception:
            rows.append(_nan_feature_row(candidate_id, path))
    return pd.DataFrame(rows)


def append_lightcurve_features(
    table: pd.DataFrame,
    *,
    flat_lightcurve_dir: str | Path | None = None,
    cache_path: str | Path | None = None,
    max_lightcurves: int | None = None,
) -> pd.DataFrame:
    """Attach compact light-curve features to a candidate/review table."""

    if table.empty:
        return table.copy()

    cache = Path(cache_path).expanduser() if cache_path else None
    if cache is not None and cache.exists():
        lc_features = pd.read_parquet(cache)
    else:
        lc_features = compute_lightcurve_feature_frame(
            table,
            flat_lightcurve_dir=flat_lightcurve_dir,
            max_lightcurves=max_lightcurves,
        )
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            lc_features.to_parquet(cache, index=False)

    out = table.copy()
    if "candidate_id" not in lc_features.columns:
        return out
    out["candidate_id"] = out["candidate_id"].astype(str)
    lc_features = lc_features.copy()
    lc_features["candidate_id"] = lc_features["candidate_id"].astype(str)
    return out.merge(lc_features, on="candidate_id", how="left", suffixes=("", "_lc"))


def label_audit(df: pd.DataFrame, target_columns: Sequence[str]) -> dict[str, pd.Series]:
    """Return value counts for the requested labels."""

    audits: dict[str, pd.Series] = {}
    for target in target_columns:
        if target not in df.columns:
            audits[target] = pd.Series(dtype="int64")
            continue
        y = df[target].fillna("").astype(str).str.strip()
        audits[target] = y[y.ne("")].value_counts()
    return audits


def _valid_target_mask(df: pd.DataFrame, target_col: str, min_class_count: int) -> pd.Series:
    y = df[target_col].fillna("").astype(str).str.strip()
    mask = y.ne("") & y.ne("unclassified")
    counts = y[mask].value_counts()
    keep_classes = set(counts[counts >= min_class_count].index)
    return mask & y.isin(keep_classes)


def _feature_columns_for_target(
    df: pd.DataFrame,
    *,
    target_col: str,
    drop_columns: Iterable[str],
    max_categorical_cardinality: int,
) -> tuple[list[str], list[str]]:
    drop = set(drop_columns) | {target_col}
    feature_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in df.columns:
        if col in drop:
            continue
        series = df[col]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            feature_cols.append(col)
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        nonempty = series.dropna().astype(str).str.strip()
        if nonempty.empty:
            continue
        if nonempty.nunique() <= max_categorical_cardinality:
            feature_cols.append(col)
            categorical_cols.append(col)
    return feature_cols, categorical_cols


def _fit_feature_encoder(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    encoded_cols: dict[str, pd.Series] = {}
    categorical_maps: dict[str, list[str]] = {}
    categorical_set = set(categorical_columns)
    for col in feature_columns:
        if col in categorical_set:
            values = df[col].fillna("").astype(str).str.strip()
            categories = sorted(v for v in values.unique() if v)
            categorical_maps[col] = categories
            mapping = {value: idx for idx, value in enumerate(categories)}
            encoded_cols[col] = values.map(mapping).fillna(-1).astype("int32")
        else:
            numeric = pd.to_numeric(df[col], errors="coerce")
            encoded_cols[col] = numeric.replace([np.inf, -np.inf], np.nan).astype("float64")
    encoded = pd.DataFrame(encoded_cols, index=df.index)
    return encoded, categorical_maps


def transform_features(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    categorical_maps: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    encoded_cols: dict[str, pd.Series] = {}
    for col in feature_columns:
        if col in categorical_maps:
            values = df[col].fillna("").astype(str).str.strip() if col in df.columns else pd.Series("", index=df.index)
            mapping = {value: idx for idx, value in enumerate(categorical_maps[col])}
            encoded_cols[col] = values.map(mapping).fillna(-1).astype("int32")
        else:
            values = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
            encoded_cols[col] = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")
    return pd.DataFrame(encoded_cols, index=df.index)


def _encode_labels(y: pd.Series) -> tuple[np.ndarray, list[str]]:
    classes = sorted(y.astype(str).unique())
    mapping = {value: idx for idx, value in enumerate(classes)}
    return y.map(mapping).to_numpy(dtype=int), classes


def _build_classifier(config: TrainingConfig, n_classes: int) -> lgb.LGBMClassifier:
    objective = "binary" if n_classes == 2 else "multiclass"
    return lgb.LGBMClassifier(
        objective=objective,
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_samples=config.min_child_samples,
        random_state=config.random_state,
        class_weight=config.class_weight,
        n_jobs=config.n_jobs,
        verbosity=-1,
    )


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray, *, fold: int | str) -> dict[str, Any]:
    return {
        "fold": fold,
        "n_eval": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _cv_metrics(
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    *,
    config: TrainingConfig,
    n_classes: int,
) -> pd.DataFrame:
    counts = pd.Series(y_encoded).value_counts()
    n_splits = min(config.cv_folds, int(counts.min()))
    if n_splits < 2:
        return pd.DataFrame()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    rows: list[dict[str, Any]] = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y_encoded), start=1):
        model = _build_classifier(config, n_classes)
        model.fit(X.iloc[train_idx], y_encoded[train_idx])
        pred = model.predict(X.iloc[val_idx])
        rows.append(_metric_row(y_encoded[val_idx], pred, fold=fold))
    return pd.DataFrame(rows)


def _effective_test_size(n_rows: int, n_classes: int, requested: float) -> float:
    if n_rows <= n_classes:
        return requested
    min_fraction = n_classes / n_rows
    max_fraction = (n_rows - n_classes) / n_rows
    adjusted = max(requested, min_fraction)
    return min(adjusted, max_fraction)


def train_target_model(
    df: pd.DataFrame,
    target_col: str,
    *,
    config: TrainingConfig | None = None,
    drop_columns: Iterable[str] = DEFAULT_DROP_COLUMNS,
) -> TargetTrainingResult:
    """Train one LightGBM classifier for one target column."""

    if config is None:
        config = TrainingConfig()
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")

    mask = _valid_target_mask(df, target_col, config.min_class_count)
    work = df.loc[mask].copy()
    if work.empty:
        raise ValueError(f"No trainable rows for target {target_col!r}")

    y = work[target_col].fillna("").astype(str).str.strip()
    class_counts = y.value_counts().sort_index()
    feature_columns, categorical_columns = _feature_columns_for_target(
        work,
        target_col=target_col,
        drop_columns=drop_columns,
        max_categorical_cardinality=config.max_categorical_cardinality,
    )
    if not feature_columns:
        raise ValueError(f"No usable feature columns for target {target_col!r}")
    X, categorical_maps = _fit_feature_encoder(work, feature_columns, categorical_columns)
    y_encoded, label_classes = _encode_labels(y)
    n_classes = len(label_classes)
    if n_classes < 2:
        raise ValueError(f"Target {target_col!r} has fewer than two trainable classes")

    cv = _cv_metrics(X, y_encoded, config=config, n_classes=n_classes)

    stratify = y_encoded if pd.Series(y_encoded).value_counts().min() >= 2 else None
    test_size = _effective_test_size(len(X), n_classes, config.test_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=config.random_state,
        stratify=stratify,
    )
    model = _build_classifier(config, n_classes)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    holdout = _metric_row(y_test, pred, fold="holdout")
    holdout["classification_report"] = classification_report(
        y_test,
        pred,
        labels=list(range(n_classes)),
        target_names=label_classes,
        output_dict=True,
        zero_division=0,
    )

    return TargetTrainingResult(
        target_column=target_col,
        n_rows=int(len(work)),
        n_features=int(len(feature_columns)),
        class_counts={str(k): int(v) for k, v in class_counts.items()},
        feature_columns=list(feature_columns),
        categorical_features=list(categorical_columns),
        categorical_maps={k: list(v) for k, v in categorical_maps.items()},
        label_classes=label_classes,
        cv_metrics=cv,
        holdout_metrics=holdout,
        model=model,
    )


def train_review_models(
    df: pd.DataFrame,
    target_columns: Sequence[str],
    *,
    output_dir: str | Path | None = None,
    config: TrainingConfig | None = None,
    drop_columns: Iterable[str] = DEFAULT_DROP_COLUMNS,
) -> dict[str, TargetTrainingResult]:
    """Train and optionally save one LightGBM classifier per target column."""

    if config is None:
        config = TrainingConfig()
    results: dict[str, TargetTrainingResult] = {}
    for target in target_columns:
        result = train_target_model(df, target, config=config, drop_columns=drop_columns)
        results[target] = result
        if output_dir is not None:
            save_target_model(result, Path(output_dir) / target)
    return results


def save_target_model(result: TargetTrainingResult, output_dir: str | Path) -> None:
    """Save model booster and metadata for one target."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if result.model is None:
        raise ValueError("Cannot save a result without a trained model")
    result.model.booster_.save_model(str(out_dir / "lightgbm_model.txt"))
    metadata = result.metadata()
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    if not result.cv_metrics.empty:
        result.cv_metrics.to_csv(out_dir / "cv_metrics.csv", index=False)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return str(value)
