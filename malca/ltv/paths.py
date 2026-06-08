"""Shared filesystem layout helpers for LTV run bundles."""
from __future__ import annotations

from pathlib import Path

from malca.config import DEFAULT_OUTPUT_DIR

DEFAULT_LTV_RUN_DIR = DEFAULT_OUTPUT_DIR / "runs" / "ltv"
MIGRATED_LTV_RUN_DIR = DEFAULT_OUTPUT_DIR / "runs" / "ltv_march18"
LEGACY_LTV_OUTPUT_DIR = Path("output") / "ltv"


def ltv_results_dir(run_dir: str | Path | None = None) -> Path:
    return Path(run_dir or DEFAULT_LTV_RUN_DIR).expanduser() / "results"


def ltv_review_dir(run_dir: str | Path | None = None) -> Path:
    return Path(run_dir or DEFAULT_LTV_RUN_DIR).expanduser() / "review"


def ltv_review_db_path(run_dir: str | Path | None = None) -> Path:
    return ltv_review_dir(run_dir) / "review.db"


def ltv_lightcurve_dir(run_dir: str | Path | None = None) -> Path:
    return Path(run_dir or DEFAULT_LTV_RUN_DIR).expanduser() / "bundle_assets" / "lightcurves"


def ltv_core_output_path(mag_bin: str, run_dir: str | Path | None = None) -> Path:
    return ltv_results_dir(run_dir) / f"LTvar{mag_bin.replace('_', '-')}.parquet"


def ltv_pipeline_output_path(mag_bin: str, run_dir: str | Path | None = None) -> Path:
    return ltv_results_dir(run_dir) / f"LTvar{mag_bin.replace('_', '-')}_pipeline.parquet"


def ltv_external_lcs_output_path(mag_bin: str, run_dir: str | Path | None = None) -> Path:
    return ltv_results_dir(run_dir) / f"LTvar{mag_bin.replace('_', '-')}_external_lcs.parquet"


def ltv_multi_survey_output_path(mag_bin: str, run_dir: str | Path | None = None) -> Path:
    return ltv_results_dir(run_dir) / f"LTvar{mag_bin.replace('_', '-')}_ltv_multi_survey.parquet"


def ltv_filtered_output_path(mag_bin: str, run_dir: str | Path | None = None) -> Path:
    return ltv_results_dir(run_dir) / f"LTvar{mag_bin.replace('_', '-')}_filtered.parquet"


def ltv_all_filtered_output_path(run_dir: str | Path | None = None) -> Path:
    return ltv_results_dir(run_dir) / "LTvar_all_filtered.parquet"


def ltv_all_pipeline_output_path(run_dir: str | Path | None = None) -> Path:
    return ltv_results_dir(run_dir) / "LTvar_all_pipeline.parquet"


def ltv_all_external_lcs_output_path(run_dir: str | Path | None = None) -> Path:
    return ltv_results_dir(run_dir) / "LTvar_all_external_lcs.parquet"


def ltv_all_multi_survey_output_path(run_dir: str | Path | None = None) -> Path:
    return ltv_results_dir(run_dir) / "LTvar_all_ltv_multi_survey.parquet"


def ltv_run_dir_from_review_db(db_path: str | Path) -> Path | None:
    path = Path(db_path).expanduser()
    if path.name == "review.db" and path.parent.name == "review":
        return path.parent.parent
    return None


def discover_ltv_output_dir() -> Path:
    """Return the preferred existing LTV results directory.

    Discovery order matches the migration plan: canonical default run,
    named LTV run bundles, then the legacy standalone output directory.
    """
    default_results = ltv_results_dir(DEFAULT_LTV_RUN_DIR)
    if default_results.exists():
        return default_results

    runs_root = DEFAULT_OUTPUT_DIR / "runs"
    for run_dir in sorted(runs_root.glob("ltv_*")):
        results = ltv_results_dir(run_dir)
        if results.exists():
            return results

    legacy = LEGACY_LTV_OUTPUT_DIR / "ltv"
    if legacy.exists():
        return legacy

    return default_results


def default_ltv_review_db_for_output(output_dir: str | Path | None = None) -> Path:
    """Resolve the review DB matching an LTV output/results directory."""
    if output_dir is None:
        run_dir = DEFAULT_LTV_RUN_DIR
    else:
        path = Path(output_dir).expanduser()
        if path.name == "ltv" and path.parent.name == "ltv":
            return path / "ltv_candidates.db"
        run_dir = path.parent if path.name == "results" else path

    db_path = ltv_review_db_path(run_dir)
    if db_path.exists() or run_dir != DEFAULT_LTV_RUN_DIR:
        return db_path

    legacy = LEGACY_LTV_OUTPUT_DIR / "ltv" / "ltv_candidates.db"
    if legacy.exists():
        return legacy

    return db_path
