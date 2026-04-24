from __future__ import annotations

from pathlib import Path
import re
import sqlite3
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from malca.lightcurve_io import load_lightcurve_df
from malca.notebook_paths import find_repo_root, localize_lightcurve_frame_paths
from malca.review.explore_data import infer_source_kind, load_candidate_source


DEFAULT_CANDIDATES_SOURCE = Path("output/candidates.parquet")
DEFAULT_EXPORT_DIR = Path("output/triage/march18_other_eb")
DEFAULT_EVENT_CLASS_FILTER = "other"
DEFAULT_REVIEW_DB_GLOBS = (
    "review.db",
    "output/review/review.db",
    "output/review/standalone.db",
    "output/runs/*/review/review.db",
    "*review*.db",
)
LABEL_COLUMN_CANDIDATES = (
    "event_class",
    "review_label",
    "label",
    "class",
)
STATUS_COLUMN_CANDIDATES = (
    "status",
    "review_status",
)
ID_COLUMN_CANDIDATES = (
    "candidate_id",
    "asas_sn_id",
    "gaia_id",
)
PATH_COLUMN_CANDIDATES = (
    "local_lightcurve_path",
    "path",
    "dat_path",
    "lc_path",
)
EB_CLASS_PATTERNS = (
    r"\bEA\b",
    r"\bEB\b",
    r"\bEW\b",
    r"\bELL\b",
    r"\bECL\b",
    r"ALGOL",
    r"W UMA",
    r"CONTACT",
)
PERIODIC_CLASS_PATTERNS = EB_CLASS_PATTERNS + (
    r"\bRR",
    r"CEP",
    r"DSCT",
    r"SXPHE",
    r"GDOR",
    r"\bROT\b",
    r"\bBY\b",
    r"LPV",
    r"MIRA",
    r"PERIODIC",
)
EB_REGEX = re.compile("|".join(EB_CLASS_PATTERNS), flags=re.IGNORECASE)
PERIODIC_REGEX = re.compile("|".join(PERIODIC_CLASS_PATTERNS), flags=re.IGNORECASE)
DISPLAY_COLUMNS = (
    "candidate_id",
    "asas_sn_id",
    "event_class",
    "status",
    "eb_likely_label",
    "eb_likely_flag",
    "eb_bin",
    "eb_score",
    "eb_score_notes",
    "stats_variability_lomb_scargle_best_period_days",
    "stats_variability_lomb_scargle_peak_power",
    "stats_variability_lomb_scargle_fap",
    "dip_run_count",
    "dipper_n_valid_dips",
    "spacing_cv",
    "dip_amplitude_consistency",
    "dip_duration_consistency",
    "symmetry_abs",
    "stats_variability_von_neumann_ratio",
    "stats_variability_stetson_J",
    "stats_variability_lag1_autocorr",
    "dipper_score",
    "known_eb_hint",
    "known_periodic_hint",
    "gaia_var_class",
    "gaia_eb_period",
    "gaia_eb_morph",
    "vsx_class",
    "catalog_match",
    "local_lightcurve_path",
)
BIN_ORDER = (
    "strong_eb_candidate",
    "possible_eb",
    "maybe_periodic_not_eb",
    "unlikely_eb",
)
EB_TRIAGE_COLOR_MAP = {
    "strong_eb_candidate": "#d62728",
    "possible_eb": "#ff7f0e",
    "maybe_periodic_not_eb": "#1f77b4",
    "unlikely_eb": "#7f7f7f",
}
LIKELY_EB_BINS = (
    "strong_eb_candidate",
    "possible_eb",
)
DEFAULT_TOP_EXPORT_COLUMNS = (
    "candidate_id",
    "asas_sn_id",
    "event_class",
    "status",
    "eb_likely_label",
    "eb_likely_flag",
    "eb_bin",
    "eb_score",
    "eb_score_notes",
    "stats_variability_lomb_scargle_best_period_days",
    "stats_variability_lomb_scargle_peak_power",
    "stats_variability_lomb_scargle_fap",
    "ls_fap_score",
    "dip_run_count",
    "dipper_n_valid_dips",
    "spacing_cv",
    "dip_amplitude_consistency",
    "dip_duration_consistency",
    "symmetry_abs",
    "stats_variability_von_neumann_ratio",
    "stats_variability_stetson_J",
    "dipper_score",
    "known_eb_hint",
    "known_periodic_hint",
    "gaia_var_class",
    "gaia_eb_period",
    "gaia_eb_morph",
    "vsx_class",
    "catalog_match",
)


def _text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    series = df[column]
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype("object")
    return series.fillna("").astype(str).str.strip()


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    series = df[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0
    lowered = series.fillna("").astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def _normalize_id(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if not text:
        return ""
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except Exception:
            return text
    return text


def _series_contains_pattern(df: pd.DataFrame, column: str, pattern: re.Pattern[str]) -> pd.Series:
    return _text_series(df, column).str.contains(pattern, na=False)


def _detect_label_column(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(f"Label column {explicit!r} not found in review source")
        return explicit
    for column in LABEL_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(
        "Could not find a review label column. Expected one of "
        f"{', '.join(LABEL_COLUMN_CANDIDATES)}."
    )


def _detect_status_column(df: pd.DataFrame) -> str | None:
    for column in STATUS_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    return None


def _ensure_candidate_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "candidate_id" not in out.columns:
        out["candidate_id"] = ""
    out["candidate_id"] = out["candidate_id"].map(_normalize_id)
    missing = out["candidate_id"] == ""

    if bool(missing.any()):
        for column in ("asas_sn_id", "gaia_id"):
            if column not in out.columns:
                continue
            fill_values = out.loc[missing, column].map(_normalize_id)
            out.loc[missing, "candidate_id"] = fill_values
            missing = out["candidate_id"] == ""
            if not bool(missing.any()):
                break

    if bool(missing.any()):
        for column in ("path", "dat_path", "lc_path"):
            if column not in out.columns:
                continue
            stems = out.loc[missing, column].map(
                lambda value: Path(str(value)).stem if value is not None and str(value).strip() else ""
            )
            out.loc[missing, "candidate_id"] = stems.map(_normalize_id)
            missing = out["candidate_id"] == ""
            if not bool(missing.any()):
                break

    if bool((out["candidate_id"] == "").all()):
        raise ValueError(
            "Could not derive candidate IDs from the input table. "
            "Expected candidate_id, asas_sn_id, gaia_id, or a light-curve path column."
        )

    return out


def _standard_review_db_paths(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in DEFAULT_REVIEW_DB_GLOBS:
        for candidate in repo_root.glob(pattern):
            if not candidate.is_file():
                continue
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)
            paths.append(candidate.resolve())
    return sorted(paths)


def discover_review_sources(
    *,
    event_class_filter: str = DEFAULT_EVENT_CLASS_FILTER,
    search_paths: Sequence[str | Path] | None = None,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    """Scan likely review DBs and report how many reviewed rows match the target label."""
    root = find_repo_root(repo_root)
    if search_paths is None:
        paths = _standard_review_db_paths(root)
    else:
        paths = []
        for item in search_paths:
            path = Path(item).expanduser()
            path = path if path.is_absolute() else (root / path)
            if path.is_file():
                paths.append(path.resolve())
        paths = sorted({str(path): path for path in paths}.values(), key=lambda path: str(path))

    rows: list[dict[str, object]] = []
    label_value = str(event_class_filter or "").strip().lower()
    for path in paths:
        record: dict[str, object] = {
            "source_path": str(path),
            "has_candidates_table": False,
            "has_reviews_table": False,
            "n_candidates": 0,
            "n_reviews": 0,
            "n_reviewed": 0,
            "n_matching_reviews": 0,
        }
        try:
            with sqlite3.connect(path) as conn:
                table_names = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                record["has_candidates_table"] = "candidates" in table_names
                record["has_reviews_table"] = "reviews" in table_names
                if "candidates" in table_names:
                    record["n_candidates"] = int(
                        conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
                    )
                if "reviews" in table_names:
                    record["n_reviews"] = int(
                        conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
                    )
                    record["n_reviewed"] = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM reviews WHERE status IS NOT NULL AND status != 'unreviewed'"
                        ).fetchone()[0]
                    )
                    record["n_matching_reviews"] = int(
                        conn.execute(
                            """
                            SELECT COUNT(*)
                            FROM reviews
                            WHERE lower(coalesce(event_class, '')) = ?
                              AND status IS NOT NULL
                              AND status != 'unreviewed'
                            """,
                            (label_value,),
                        ).fetchone()[0]
                    )
        except Exception as exc:
            record["error"] = str(exc)
        rows.append(record)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "source_path",
                "has_candidates_table",
                "has_reviews_table",
                "n_candidates",
                "n_reviews",
                "n_reviewed",
                "n_matching_reviews",
            ]
        )
    return out.sort_values(
        ["n_matching_reviews", "n_reviewed", "source_path"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def load_reviewed_other_subset(
    review_source: str | Path,
    candidates_source: str | Path = DEFAULT_CANDIDATES_SOURCE,
    *,
    event_class_filter: str = DEFAULT_EVENT_CLASS_FILTER,
    review_label_col: str | None = None,
) -> pd.DataFrame:
    """Load the reviewed `other` subset from a review DB or exported review table."""
    review_path = Path(review_source).expanduser().resolve()
    review_kind = infer_source_kind(review_path)

    if review_kind == "db":
        merged = load_candidate_source(review_path, review_kind)
        merged = _ensure_candidate_id(merged)
    else:
        review_df = load_candidate_source(review_path, review_kind)
        review_df = _ensure_candidate_id(review_df)
        label_column = _detect_label_column(review_df, explicit=review_label_col)
        if label_column != "event_class":
            review_df = review_df.rename(columns={label_column: "event_class"})
        status_column = _detect_status_column(review_df)
        if status_column is not None and status_column != "status":
            review_df = review_df.rename(columns={status_column: "status"})
        if "status" not in review_df.columns:
            review_df["status"] = "reviewed"

        candidates_path = Path(candidates_source).expanduser().resolve()
        candidates_kind = infer_source_kind(candidates_path)
        candidates_df = load_candidate_source(candidates_path, candidates_kind)
        candidates_df = _ensure_candidate_id(candidates_df)

        review_cols = [
            column
            for column in (
                "candidate_id",
                "event_class",
                "interest_score",
                "review_pass",
                "notes",
                "status",
                "reviewer",
                "updated_at",
            )
            if column in review_df.columns
        ]
        reviews_only = review_df[review_cols].copy()
        reviews_only = reviews_only.drop_duplicates(subset=["candidate_id"], keep="last")
        merged = candidates_df.merge(reviews_only, on="candidate_id", how="left")

    label_series = _text_series(merged, "event_class").str.lower()
    if "status" in merged.columns:
        status_series = _text_series(merged, "status").str.lower()
        reviewed_mask = status_series.ne("") & status_series.ne("unreviewed")
    else:
        reviewed_mask = pd.Series(True, index=merged.index, dtype=bool)

    filtered = merged.loc[
        reviewed_mask & label_series.eq(str(event_class_filter or "").strip().lower())
    ].copy()
    filtered = filtered.reset_index(drop=True)
    filtered.attrs["review_source"] = str(review_path)
    filtered.attrs["candidates_source"] = str(candidates_source)
    filtered.attrs["event_class_filter"] = str(event_class_filter)
    return filtered


def _clip_log10(values: pd.Series, floor: float = 1e-300) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        clipped = np.clip(arr.to_numpy(dtype=float), floor, None)
        result = -np.log10(clipped)
    result[~np.isfinite(arr.to_numpy(dtype=float))] = np.nan
    return pd.Series(result, index=values.index, dtype=float)


def _pick_local_lightcurve_path(row: pd.Series) -> str | None:
    for column in PATH_COLUMN_CANDIDATES:
        if column not in row.index:
            continue
        value = row.get(column)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        if column == "local_lightcurve_path":
            return text
        path = Path(text)
        if path.exists():
            return str(path)
    return None


def resolve_local_paths(
    df: pd.DataFrame,
    run_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Resolve stored path columns against a local run or bundle directory."""
    out, localized_counts = localize_lightcurve_frame_paths(df, run_dir=run_dir)
    local_paths: list[str | None] = []
    local_exists: list[bool] = []
    for _, row in out.iterrows():
        local_path = _pick_local_lightcurve_path(row)
        local_paths.append(local_path)
        local_exists.append(bool(local_path and Path(local_path).exists()))
    out["local_lightcurve_path"] = local_paths
    out["local_lightcurve_exists"] = local_exists
    out.attrs["localized_counts"] = localized_counts
    out.attrs["run_dir"] = str(run_dir) if run_dir is not None else None
    return out


def _coefficient_of_variation(numer: pd.Series, denom: pd.Series) -> pd.Series:
    numer_arr = pd.to_numeric(numer, errors="coerce").to_numpy(dtype=float)
    denom_arr = pd.to_numeric(denom, errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numer_arr / denom_arr
    invalid = ~np.isfinite(numer_arr) | ~np.isfinite(denom_arr) | (denom_arr <= 0)
    out[invalid] = np.nan
    return pd.Series(out, index=numer.index, dtype=float)


def _score_notes(row: pd.Series) -> str:
    labels = []
    for column, label in (
        ("eb_score_periodicity", "period"),
        ("eb_score_run_count", "repeat_runs"),
        ("eb_score_spacing_regularity", "spacing_regular"),
        ("eb_score_amplitude_consistency", "amp_consistent"),
        ("eb_score_duration_consistency", "dur_consistent"),
        ("eb_score_symmetry", "symmetric"),
        ("eb_score_von_neumann", "von_neumann"),
        ("eb_score_stetson", "stetson"),
    ):
        value = row.get(column, 0)
        try:
            if int(value) > 0:
                labels.append(f"{label}:{int(value)}")
        except Exception:
            continue
    return ", ".join(labels)


def compute_eb_triage(df: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic EB-triage scores and bins from existing candidate stats."""
    out = _ensure_candidate_id(df)

    ls_period = _numeric_series(out, "stats_variability_lomb_scargle_best_period_days")
    ls_power = _numeric_series(out, "stats_variability_lomb_scargle_peak_power")
    ls_fap = _numeric_series(out, "stats_variability_lomb_scargle_fap")
    dip_run_count = _numeric_series(out, "dip_run_count").fillna(0)
    dip_spacing_median = _numeric_series(out, "dip_inter_event_spacing_median")
    dip_spacing_std = _numeric_series(out, "dip_inter_event_spacing_std")
    dip_amp_consistency = _numeric_series(out, "dip_amplitude_consistency")
    dip_dur_consistency = _numeric_series(out, "dip_duration_consistency")
    dip_symmetry = _numeric_series(out, "dip_symmetry_score")
    vn_ratio = _numeric_series(out, "stats_variability_von_neumann_ratio")
    stetson_j = _numeric_series(out, "stats_variability_stetson_J")
    dipper_score = _numeric_series(out, "dipper_score")

    out["stats_variability_lomb_scargle_best_period_days"] = ls_period
    out["stats_variability_lomb_scargle_peak_power"] = ls_power
    out["stats_variability_lomb_scargle_fap"] = ls_fap
    out["dip_run_count"] = dip_run_count
    out["dip_inter_event_spacing_median"] = dip_spacing_median
    out["dip_inter_event_spacing_std"] = dip_spacing_std
    out["dip_amplitude_consistency"] = dip_amp_consistency
    out["dip_duration_consistency"] = dip_dur_consistency
    out["dip_symmetry_score"] = dip_symmetry
    out["stats_variability_von_neumann_ratio"] = vn_ratio
    out["stats_variability_stetson_J"] = stetson_j
    out["dipper_score"] = dipper_score

    out["ls_fap_score"] = _clip_log10(ls_fap)
    out["spacing_cv"] = _coefficient_of_variation(
        dip_spacing_std,
        dip_spacing_median,
    )
    out["symmetry_abs"] = dip_symmetry.abs()

    known_eb_hint = (
        _series_contains_pattern(out, "gaia_var_class", EB_REGEX)
        | _series_contains_pattern(out, "vsx_class", EB_REGEX)
        | _numeric_series(out, "gaia_eb_period").notna()
        | _text_series(out, "gaia_eb_morph").ne("")
    )
    known_periodic_hint = (
        _bool_series(out, "catalog_match")
        | _bool_series(out, "periodic_flag")
        | _series_contains_pattern(out, "gaia_var_class", PERIODIC_REGEX)
        | _series_contains_pattern(out, "vsx_class", PERIODIC_REGEX)
    )
    out["known_eb_hint"] = known_eb_hint.astype(bool)
    out["known_periodic_hint"] = known_periodic_hint.astype(bool)

    finite_ls_period = ls_period.notna() & np.isfinite(ls_period) & (ls_period > 0)
    out["finite_ls_period"] = finite_ls_period.astype(bool)

    score_periodicity = pd.Series(0, index=out.index, dtype=int)
    strong_period_mask = finite_ls_period & ls_fap.le(1e-6) & ls_power.ge(0.20)
    medium_period_mask = finite_ls_period & ~strong_period_mask & ls_fap.gt(1e-6) & ls_fap.le(1e-3)
    weak_period_mask = finite_ls_period & ~(strong_period_mask | medium_period_mask)
    score_periodicity.loc[strong_period_mask] = 3
    score_periodicity.loc[medium_period_mask] = 2
    score_periodicity.loc[weak_period_mask] = 1

    score_run_count = pd.Series(0, index=out.index, dtype=int)
    score_run_count.loc[dip_run_count >= 2] += 2
    score_run_count.loc[dip_run_count >= 3] += 1

    score_spacing = pd.Series(0, index=out.index, dtype=int)
    score_spacing.loc[_numeric_series(out, "spacing_cv").le(0.25)] = 1

    score_amp = pd.Series(0, index=out.index, dtype=int)
    score_amp.loc[dip_amp_consistency.le(0.25)] = 1

    score_dur = pd.Series(0, index=out.index, dtype=int)
    score_dur.loc[dip_dur_consistency.le(0.35)] = 1

    score_sym = pd.Series(0, index=out.index, dtype=int)
    score_sym.loc[_numeric_series(out, "symmetry_abs").le(0.30)] = 1

    score_vn = pd.Series(0, index=out.index, dtype=int)
    score_vn.loc[vn_ratio.ge(1.5)] = 1

    score_stetson = pd.Series(0, index=out.index, dtype=int)
    score_stetson.loc[stetson_j.ge(0.5)] = 1

    out["eb_score_periodicity"] = score_periodicity
    out["eb_score_run_count"] = score_run_count
    out["eb_score_spacing_regularity"] = score_spacing
    out["eb_score_amplitude_consistency"] = score_amp
    out["eb_score_duration_consistency"] = score_dur
    out["eb_score_symmetry"] = score_sym
    out["eb_score_von_neumann"] = score_vn
    out["eb_score_stetson"] = score_stetson

    out["eb_score"] = (
        score_periodicity
        + score_run_count
        + score_spacing
        + score_amp
        + score_dur
        + score_sym
        + score_vn
        + score_stetson
    ).astype(int)
    out["eb_score_notes"] = out.apply(_score_notes, axis=1)

    strong_mask = out["eb_score"].ge(7) & finite_ls_period & dip_run_count.ge(2)
    possible_mask = out["eb_score"].between(5, 6, inclusive="both") & finite_ls_period & ~strong_mask
    maybe_mask = ((out["eb_score"].between(3, 4, inclusive="both")) | finite_ls_period) & ~(strong_mask | possible_mask)

    out["eb_bin"] = "unlikely_eb"
    out.loc[maybe_mask, "eb_bin"] = "maybe_periodic_not_eb"
    out.loc[possible_mask, "eb_bin"] = "possible_eb"
    out.loc[strong_mask, "eb_bin"] = "strong_eb_candidate"
    out["eb_bin"] = pd.Categorical(out["eb_bin"], categories=BIN_ORDER, ordered=True)
    eb_bin_text = _text_series(out, "eb_bin")
    out["eb_likely_flag"] = eb_bin_text.isin(LIKELY_EB_BINS)
    out["eb_likely_label"] = np.where(out["eb_likely_flag"], "likely_eb", "not_likely_eb")

    out = out.sort_values(
        ["eb_bin", "eb_score", "ls_fap_score", "dip_run_count", "dipper_score", "candidate_id"],
        ascending=[True, False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return out


def top_candidate_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact export table for strong and possible EB candidates."""
    keep_mask = _text_series(df, "eb_bin").isin(LIKELY_EB_BINS)
    export_cols = [column for column in DEFAULT_TOP_EXPORT_COLUMNS if column in df.columns]
    return df.loc[keep_mask, export_cols].copy().reset_index(drop=True)


def build_eb_triage_summary_figure(df: pd.DataFrame) -> plt.Figure:
    """Build the standard March 18 EB-triage summary figure."""
    plot_df = df.copy()
    colors = (
        _text_series(plot_df, "eb_bin")
        .map(EB_TRIAGE_COLOR_MAP)
        .fillna(EB_TRIAGE_COLOR_MAP["unlikely_eb"])
    )
    plot_specs = (
        (
            "ls_fap_score",
            "dip_run_count",
            "ls_fap_score = -log10(FAP)",
            "dip_run_count",
            "Period significance vs repeat runs",
        ),
        (
            "spacing_cv",
            "dip_amplitude_consistency",
            "spacing_cv",
            "dip_amplitude_consistency",
            "Recurrence regularity vs amplitude repeatability",
        ),
        (
            "symmetry_abs",
            "dip_duration_consistency",
            "symmetry_abs",
            "dip_duration_consistency",
            "Symmetry vs duration repeatability",
        ),
    )

    fig, axes = plt.subplots(1, len(plot_specs), figsize=(20, 5), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, (x_col, y_col, x_label, y_label, title) in zip(axes, plot_specs):
        x = _numeric_series(plot_df, x_col)
        y = _numeric_series(plot_df, y_col)
        valid = x.notna() & y.notna()
        if bool(valid.any()):
            ax.scatter(
                x.loc[valid],
                y.loc[valid],
                c=colors.loc[valid],
                s=16,
                alpha=0.75,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No finite data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#6e6e6e",
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(alpha=0.2)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=color,
            markersize=8,
        )
        for label, color in EB_TRIAGE_COLOR_MAP.items()
    ]
    axes[0].legend(handles=legend_handles, title="eb_bin", loc="best")
    return fig


def export_eb_triage_products(
    subset_df: pd.DataFrame,
    triage_df: pd.DataFrame,
    *,
    export_dir: str | Path = DEFAULT_EXPORT_DIR,
) -> dict[str, Path]:
    """Write the standard triage artifacts to disk."""
    out_dir = Path(export_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    subset_path = out_dir / "other_subset.parquet"
    triage_path = out_dir / "other_eb_triage.parquet"
    top_path = out_dir / "other_eb_top_candidates.csv"
    summary_plot_path = out_dir / "other_eb_triage_summary.png"

    subset_df.to_parquet(subset_path, index=False)
    triage_df.to_parquet(triage_path, index=False)
    top_candidate_export_frame(triage_df).to_csv(top_path, index=False)
    fig = build_eb_triage_summary_figure(triage_df)
    fig.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "subset_path": subset_path,
        "triage_path": triage_path,
        "top_candidates_path": top_path,
        "summary_plot_path": summary_plot_path,
    }


def _find_candidate_row(df: pd.DataFrame, candidate_id: object) -> pd.Series:
    candidate_key = _normalize_id(candidate_id)
    if not candidate_key:
        raise KeyError("Candidate ID is empty")

    frame = _ensure_candidate_id(df)
    masks = [frame["candidate_id"].map(_normalize_id).eq(candidate_key)]
    for column in ("asas_sn_id", "gaia_id"):
        if column in frame.columns:
            masks.append(frame[column].map(_normalize_id).eq(candidate_key))
    mask = masks[0]
    for extra in masks[1:]:
        mask |= extra
    if not bool(mask.any()):
        raise KeyError(f"Candidate {candidate_key!r} not found")
    return frame.loc[mask].iloc[0]


def _plot_lightcurve_axes(
    lc_df: pd.DataFrame,
    *,
    candidate_id: str,
    period_days: float | None,
) -> plt.Figure:
    has_period = period_days is not None and np.isfinite(period_days) and period_days > 0
    ncols = 2 if has_period else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    jd = pd.to_numeric(lc_df.get("JD"), errors="coerce")
    mag = pd.to_numeric(lc_df.get("mag"), errors="coerce")
    valid = jd.notna() & mag.notna()
    jd = jd.loc[valid].to_numpy(dtype=float)
    mag = mag.loc[valid].to_numpy(dtype=float)

    axes[0].scatter(jd, mag, s=8, alpha=0.8, color="#1f77b4")
    axes[0].set_title(f"{candidate_id} raw light curve")
    axes[0].set_xlabel("JD")
    axes[0].set_ylabel("mag")
    axes[0].invert_yaxis()

    if has_period:
        phase = np.mod((jd - np.nanmin(jd)) / float(period_days), 1.0)
        phase_twice = np.concatenate([phase, phase + 1.0])
        mag_twice = np.concatenate([mag, mag])
        axes[1].scatter(phase_twice, mag_twice, s=8, alpha=0.8, color="#d62728")
        axes[1].set_title(f"Phase folded (P={float(period_days):.4f} d)")
        axes[1].set_xlabel("phase")
        axes[1].set_ylabel("mag")
        axes[1].invert_yaxis()

    return fig


def _pretty_bin_label(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "unlabeled"
    return text.replace("_", " ")


def _safe_filename_token(value: object) -> str:
    text = _normalize_id(value) or str(value).strip()
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return token or "candidate"


def select_example_candidates(
    df: pd.DataFrame,
    *,
    examples_per_bin: int = 2,
    bins: Sequence[str] = BIN_ORDER,
    require_local_lightcurve: bool = True,
    run_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Select representative candidates for individual light-curve examples."""
    if examples_per_bin <= 0:
        return df.iloc[0:0].copy()

    candidates = df.copy()
    if run_dir is not None or "local_lightcurve_exists" not in candidates.columns:
        candidates = resolve_local_paths(candidates, run_dir=run_dir)

    frames: list[pd.DataFrame] = []
    bin_text = _text_series(candidates, "eb_bin")
    for eb_bin in bins:
        subset = candidates.loc[bin_text.eq(str(eb_bin))].copy()
        if require_local_lightcurve:
            if "local_lightcurve_exists" in subset.columns:
                subset = subset.loc[_bool_series(subset, "local_lightcurve_exists")].copy()
            else:
                local_paths = _text_series(subset, "local_lightcurve_path")
                subset = subset.loc[local_paths.ne("")].copy()
        if subset.empty:
            continue
        frames.append(subset.head(int(examples_per_bin)))

    if not frames:
        return candidates.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


def inspect_candidate(
    df: pd.DataFrame,
    candidate_id: object,
    run_dir: str | Path | None = None,
    *,
    display_metadata: bool = True,
    show_figure: bool = True,
) -> dict[str, Any]:
    """Display metadata and, when possible, raw + phase-folded light curves."""
    row = _find_candidate_row(df, candidate_id)
    row_frame = pd.DataFrame([row]).copy()
    localized = resolve_local_paths(row_frame, run_dir=run_dir)
    row_local = localized.iloc[0]
    metadata_cols = [column for column in DISPLAY_COLUMNS if column in localized.columns]
    metadata = localized.loc[:, metadata_cols].copy()

    result: dict[str, Any] = {
        "candidate_id": _normalize_id(row_local.get("candidate_id")),
        "record": row_local.copy(),
        "metadata": metadata,
        "resolved_path": row_local.get("local_lightcurve_path"),
        "lightcurve_df": None,
        "figure": None,
        "status": "metadata_only",
        "messages": [],
    }

    try:
        from IPython.display import display
    except Exception:
        display = None

    if display is not None and display_metadata:
        display(metadata.T.rename(columns={metadata.index[0]: "value"}))

    lightcurve_path = row_local.get("local_lightcurve_path")
    if not lightcurve_path:
        message = (
            "No local light curve could be resolved. "
            "Use `resolve_local_paths(..., run_dir=...)` or provide a bundle/run directory."
        )
        result["messages"].append(message)
        if display is None:
            print(message)
        return result

    path_obj = Path(str(lightcurve_path))
    if not path_obj.exists():
        message = f"Resolved light-curve path does not exist: {path_obj}"
        result["messages"].append(message)
        if display is None:
            print(message)
        return result

    lc_df = load_lightcurve_df(path_obj)
    result["lightcurve_df"] = lc_df
    if not isinstance(lc_df, pd.DataFrame) or lc_df.empty or "JD" not in lc_df.columns or "mag" not in lc_df.columns:
        message = f"Light-curve file could not be loaded for plotting: {path_obj}"
        result["messages"].append(message)
        if display is None:
            print(message)
        result["status"] = "lightcurve_unavailable"
        return result

    ls_period = pd.to_numeric(
        pd.Series([row_local.get("stats_variability_lomb_scargle_best_period_days")]),
        errors="coerce",
    ).iloc[0]
    period_days = float(ls_period) if pd.notna(ls_period) and np.isfinite(ls_period) and ls_period > 0 else None
    fig = _plot_lightcurve_axes(
        lc_df,
        candidate_id=str(result["candidate_id"]),
        period_days=period_days,
    )
    if show_figure:
        plt.show()
    result["figure"] = fig
    result["status"] = "plotted"
    return result


def plot_example_lightcurves(
    df: pd.DataFrame,
    *,
    run_dir: str | Path | None = None,
    examples_per_bin: int = 2,
    bins: Sequence[str] = BIN_ORDER,
    require_local_lightcurve: bool = True,
    export_dir: str | Path | None = None,
    display_metadata: bool = False,
    show: bool = True,
) -> dict[str, Any]:
    """Plot a small set of representative individual light curves by EB-triage bin."""
    selected = select_example_candidates(
        df,
        examples_per_bin=examples_per_bin,
        bins=bins,
        require_local_lightcurve=require_local_lightcurve,
        run_dir=run_dir,
    )
    results: list[dict[str, Any]] = []
    exported_paths: list[Path] = []
    out_dir = Path(export_dir).expanduser() if export_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from IPython.display import Markdown, display
    except Exception:
        Markdown = None
        display = None

    current_bin: str | None = None
    bin_ranks: dict[str, int] = {}
    for _, row in selected.iterrows():
        candidate_id = row.get("candidate_id")
        eb_bin = str(row.get("eb_bin")).strip()
        bin_ranks[eb_bin] = bin_ranks.get(eb_bin, 0) + 1
        bin_rank = bin_ranks[eb_bin]
        if show:
            heading = f"`{_normalize_id(candidate_id)}`"
            score = pd.to_numeric(pd.Series([row.get("eb_score")]), errors="coerce").iloc[0]
            if pd.notna(score):
                heading += f" · eb_score={int(score)}"
            period = pd.to_numeric(
                pd.Series([row.get("stats_variability_lomb_scargle_best_period_days")]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(period) and np.isfinite(period) and period > 0:
                heading += f" · P={float(period):.4f} d"
            if display is not None and Markdown is not None:
                if eb_bin != current_bin:
                    display(Markdown(f"### {_pretty_bin_label(eb_bin)}"))
                display(Markdown(heading))
            else:
                if eb_bin != current_bin:
                    print(f"\n{_pretty_bin_label(eb_bin)}")
                print(heading)

        result = inspect_candidate(
            selected,
            candidate_id,
            run_dir=run_dir,
            display_metadata=display_metadata,
            show_figure=show,
        )
        result["eb_bin"] = eb_bin
        result["example_rank"] = bin_rank
        result["export_path"] = None

        if out_dir is not None and result.get("figure") is not None:
            export_path = out_dir / f"{_safe_filename_token(eb_bin)}_{bin_rank:02d}_{_safe_filename_token(candidate_id)}.png"
            result["figure"].savefig(export_path, dpi=150, bbox_inches="tight")
            result["export_path"] = export_path
            exported_paths.append(export_path)

        results.append(result)
        current_bin = eb_bin

    return {
        "selected": selected,
        "results": results,
        "exported_paths": exported_paths,
        "n_selected": int(len(selected)),
        "n_plotted": int(sum(result.get("status") == "plotted" for result in results)),
    }
