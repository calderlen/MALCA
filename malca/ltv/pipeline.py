"""
LTV Full Pipeline Integration — Optimized for Scale.

Combines all LTV modules into a complete slowly varying source detection pipeline.

CRITICAL OPTIMIZATION: Pipeline stages run in optimal order:
1. Vectorized filters FIRST (instant, reduce 17M → ~36K)
2. Gaia TAP filters (batch queries on reduced set)
3. Optional stochastic post-filter features (parallel on filtered candidates only)
4. Catalog crossmatches (parallel on filtered candidates only)
5. NEOWISE extraction (parallel on filtered candidates only)
6. Extinction correction (vectorized)

This ordering is crucial for performance:
- Running crossmatch before filtering = weeks
- Running crossmatch after filtering = hours
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from malca.config import SYDNEY_LTV_CSV_PATH
from malca.config import (
    LTV_MIN_SLOPE,
    LTV_MIN_DIFF,
    LTV_MIN_DEC,
    LTV_MAX_PM,
    LTV_MATCH_RADIUS_ARCSEC,
    LTV_WORKERS,
    LTV_CHUNK_SIZE,
    LTV_CORE_CHUNK_SIZE,
    GAIA_EPOCH_DATA_RELEASE,
    GAIA_EPOCH_DATA_STRUCTURE,
    LCV2_ROOT,
    MAG_BINS,
    LTV_DSPRING,
    LTV_MAX_SEASONS,
    LTV_MIN_POINTS_PER_SEASON,
    LTV_MIN_SEASONS_FOR_QUADRATIC,
)
from malca.cli_config import add_config_args, apply_config
from malca.ltv.filter import (
    apply_all_filters,
    apply_all_filters_audit,
    LTV_AUDIT_FAILED_COLUMNS,
    filter_slope_threshold,
    filter_max_diff_threshold,
    filter_south_pole,
    filter_high_proper_motion,
)
from malca.table_io import read_parquet_table, write_parquet_table
from malca.ltv.paths import (
    DEFAULT_LTV_RUN_DIR,
    ltv_all_external_lcs_output_path,
    ltv_all_filtered_output_path,
    ltv_all_multi_survey_output_path,
    ltv_all_pipeline_output_path,
    ltv_core_output_path,
    ltv_external_lcs_output_path,
    ltv_filtered_output_path,
    ltv_multi_survey_output_path,
    ltv_pipeline_output_path,
    ltv_review_db_path,
)
from malca.review.sync import auto_export_review_bundle


LTV_BUILD_CONFIG_DEFAULTS = {
    "skip_filters": False,
    "run_stochastic_postfilter": False,
    "stochastic_include_drw": False,
    "skip_crossmatch": False,
    "no_ztf_periodic": False,
    "no_ogle_periodic": False,
    "skip_neowise": False,
    "skip_extinction": False,
    "skip_dust_flags": False,
    "skip_gaia_epoch": False,
    "gaia_epoch_table": None,
    "gaia_epoch_data_release": GAIA_EPOCH_DATA_RELEASE,
    "gaia_epoch_data_structure": GAIA_EPOCH_DATA_STRUCTURE,
    "gaia_epoch_band": None,
    "gaia_epoch_include_invalid": False,
    "skip_bailer_jones": False,
    "skip_cmd": False,
    "run_external_lcs": None,
    "run_multi_survey_features": None,
    "external_lc_workers": 4,
    "external_lc_refresh_cache": False,
    "external_lc_atlas": False,
    "atlas_token": None,
    "log_rejections": None,
}

LTV_BUILD_CONFIG_PATH_KEYS = {"log_rejections"}

LTV_PIPELINE_STAGE_CHOICES = ("full", "cluster", "home", "full-extended")
LTV_RUN_REUSE_FINGERPRINT_VERSION = 1
LTV_CODE_FINGERPRINT_FILES = (
    "ltv/core.py",
    "ltv/filter.py",
    "ltv/pipeline.py",
    "ltv/review.py",
    "ltv/crossmatch.py",
    "ltv/neowise.py",
    "ltv/dust.py",
    "ltv/cmd.py",
    "ltv/gaia_epoch.py",
    "ltv/stochastic.py",
    "ltv/multi_survey.py",
    "external_lcs.py",
    "vetting.py",
    "table_io.py",
    "config.py",
)

LTV_ORCHESTRATOR_CONFIG_DEFAULTS = {
    **LTV_BUILD_CONFIG_DEFAULTS,
    "stage": "full",
    "run_dir": None,
    "root": LCV2_ROOT,
    "extension": None,
    "overwrite": False,
    "export_bundle": None,
    "import_bundle": None,
    "export_bundle_enabled": True,
    "full_bundle": False,
    "review_sync_enabled": True,
    "review_sync_dir": Path("reviews"),
    "review_sync_hash_assets": False,
    "run_vetting": False,
    "skip_stats": False,
    "stats_compute_ls": False,
    "index_file": None,
    "review_db": None,
    "dspring": LTV_DSPRING,
    "ra_is_deg": True,
    "max_seasons": LTV_MAX_SEASONS,
    "min_points_per_season": LTV_MIN_POINTS_PER_SEASON,
    "min_seasons_for_quadratic": LTV_MIN_SEASONS_FOR_QUADRATIC,
    "band_mode": "pipeline",
}

LTV_ORCHESTRATOR_CONFIG_PATH_KEYS = {
    "root",
    "run_dir",
    "export_bundle",
    "import_bundle",
    "review_sync_dir",
    "index_file",
    "review_db",
    "log_rejections",
}


def add_stochastic_postfilter_features(*args, **kwargs):
    """Lazy wrapper kept for tests and callers that monkeypatch this pipeline hook."""
    from malca.ltv.stochastic import add_stochastic_postfilter_features as _impl

    return _impl(*args, **kwargs)


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_full_pipeline(
    df: pd.DataFrame,
    *,
    # Filtering thresholds
    min_slope: float = LTV_MIN_SLOPE,
    min_diff: float = LTV_MIN_DIFF,
    min_dec: float = LTV_MIN_DEC,
    max_pm: float = LTV_MAX_PM,
    # Pipeline stages
    run_filters: bool = True,
    run_stochastic_postfilter: bool = False,
    stochastic_include_drw: bool = False,
    run_crossmatch: bool = True,
    run_neowise: bool = True,
    run_extinction: bool = True,
    run_dust_flags: bool = True,
    run_cmd: bool = True,
    run_bailer_jones: bool = True,
    mist_path: str | Path | None = None,
    cmd_boundaries: dict | None = None,
    run_gaia_epoch: bool = True,
    gaia_epoch_table: str | None = None,
    gaia_epoch_time_col: str = "time",
    gaia_epoch_g_col: str = "g_mag",
    gaia_epoch_bp_col: str = "bp_mag",
    gaia_epoch_rp_col: str = "rp_mag",
    gaia_epoch_data_release: str = GAIA_EPOCH_DATA_RELEASE,
    gaia_epoch_data_structure: str = GAIA_EPOCH_DATA_STRUCTURE,
    gaia_epoch_valid_data: bool = True,
    gaia_epoch_band: str | None = None,
    # Crossmatch options
    include_gaia_dr3: bool = True,
    include_gaia_alerts: bool = True,
    include_vsx: bool = True,
    include_milliquas: bool = True,
    include_simbad: bool = True,
    include_ztf_periodic: bool = True,
    include_ogle_periodic: bool = True,
    match_radius_arcsec: float = LTV_MATCH_RADIUS_ARCSEC,
    # Parallel processing
    n_workers: int = LTV_WORKERS,
    chunk_size: int = LTV_CHUNK_SIZE,
    # Output
    log_csv: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the complete LTV pipeline — optimized for 17M+ sources.
    
    CRITICAL: Pipeline runs in optimal order to minimize API calls:
    1. Vectorized filters (instant on 17M sources)
    2. Gaia TAP filters (batch on reduced set)
    3. Optional stochastic post-filter features (parallel on filtered candidates)
    4. Catalog crossmatches (parallel on filtered candidates)
    5. NEOWISE extraction (parallel on filtered candidates)
    6. Extinction correction (vectorized)
    
    For 17M sources with paper thresholds (~0.4% pass rate):
    - Filtering: ~36K candidates remain
    - Crossmatch: ~hours (not weeks) because of reduced set
    """
    n0 = len(df)
    
    if verbose:
        print("=" * 60)
        print("LTV FULL PIPELINE — Optimized for Scale")
        print("=" * 60)
        print(f"Input: {n0:,} sources")
        print(f"Workers: {n_workers}, Chunk size: {chunk_size}")
        print()
    
    # =========================================================================
    # Stage 1: Filtering (MUST run first)
    # =========================================================================
    df_rejected = None  # Rows that passed slope+diff but were later filtered out
    if run_filters:
        if verbose:
            print("-" * 60)
            print("STAGE 1: FILTERING")
            print("-" * 60)
        
        df, df_rejected = apply_all_filters(
            df,
            min_slope=min_slope,
            min_diff=min_diff,
            min_dec=min_dec,
            max_pm=max_pm,
            query_gaia=True,
            chunk_size=chunk_size,
            n_workers=n_workers,
            verbose=verbose,
            log_csv=log_csv,
            return_rejected=True,
        )
        
        if verbose:
            reduction = (1 - len(df)/n0) * 100
            print(f"\n→ After filtering: {len(df):,} candidates ({reduction:.1f}% reduction)")
            print()
    
    if df.empty and (df_rejected is None or df_rejected.empty):
        if verbose:
            print("No sources remaining after filtering")
        df = df.copy()
        if "filter_reason" not in df.columns:
            df["filter_reason"] = pd.NA
        return df

    # =========================================================================
    # Stage 1b: Optional stochastic post-filter features
    # =========================================================================
    if run_stochastic_postfilter:
        if verbose:
            print("-" * 60)
            print("STAGE 1b: STOCHASTIC POST-FILTER FEATURES")
            print("-" * 60)

        df = add_stochastic_postfilter_features(
            df,
            include_drw=stochastic_include_drw,
            n_workers=n_workers,
            verbose=verbose,
        )
        if verbose:
            print()

    # =========================================================================
    # Stage 2: Catalog crossmatch (only on filtered candidates)
    # =========================================================================
    if run_crossmatch:
        from malca.ltv.crossmatch import crossmatch_all_catalogs

        if verbose:
            print("-" * 60)
            print("STAGE 2: CATALOG CROSSMATCH")
            print("-" * 60)
            print(f"Crossmatching {len(df):,} filtered candidates")
            print()
        
        df = crossmatch_all_catalogs(
            df,
            include_gaia_dr3=include_gaia_dr3,
            include_gaia_alerts=include_gaia_alerts,
            include_vsx=include_vsx,
            include_milliquas=include_milliquas,
            include_simbad=include_simbad,
            include_ztf_periodic=include_ztf_periodic,
            include_ogle_periodic=include_ogle_periodic,
            sydney_csv_path=SYDNEY_LTV_CSV_PATH,
            match_radius_arcsec=match_radius_arcsec,
            n_workers=n_workers,
            verbose=verbose,
        )
        print()
    
    # =========================================================================
    # Stage 3: NEOWISE extraction (only on filtered candidates)
    # =========================================================================
    if run_neowise:
        from malca.ltv.neowise import extract_neowise_trends

        if verbose:
            print("-" * 60)
            print("STAGE 3: NEOWISE EXTRACTION")
            print("-" * 60)
        
        df = extract_neowise_trends(
            df,
            n_workers=n_workers,
            verbose=verbose,
        )
        print()

    # =========================================================================
    # Stage 3b: Dust-driven variability flags
    # =========================================================================
    if run_dust_flags:
        from malca.ltv.dust import apply_dust_flags

        if verbose:
            print("-" * 60)
            print("STAGE 3b: DUST FLAGS")
            print("-" * 60)
        df = apply_dust_flags(df)
        print()
    
    # =========================================================================
    # Stage 4: Extinction correction
    # =========================================================================
    if run_extinction:
        from malca.characterize import get_dust_extinction

        if verbose:
            print("-" * 60)
            print("STAGE 4: EXTINCTION CORRECTION")
            print("-" * 60)

        df = get_dust_extinction(df)

        if verbose:
            n_with_av = (df["A_v_3d"] > 0).sum() if "A_v_3d" in df.columns else 0
            print(f"[extinction] {n_with_av}/{len(df)} sources have A_V > 0")
        print()

    # =========================================================================
    # Stage 4b: Gaia epoch photometry deltas (optional)
    # =========================================================================
    if run_gaia_epoch:
        from malca.ltv.gaia_epoch import query_gaia_epoch_photometry_batch, apply_gaia_epoch_flags

        if verbose:
            print("-" * 60)
            print("STAGE 4b: GAIA EPOCH PHOTOMETRY")
            print("-" * 60)
        df = query_gaia_epoch_photometry_batch(
            df,
            tap_table=gaia_epoch_table,
            time_col=gaia_epoch_time_col,
            g_col=gaia_epoch_g_col,
            bp_col=gaia_epoch_bp_col,
            rp_col=gaia_epoch_rp_col,
            data_release=gaia_epoch_data_release,
            data_structure=gaia_epoch_data_structure,
            valid_data=gaia_epoch_valid_data,
            band=gaia_epoch_band,
            n_workers=n_workers,
            chunk_size=chunk_size,
            verbose=verbose,
        )
        df = apply_gaia_epoch_flags(df)
        print()

    # =========================================================================
    # Stage 4c: Bailer-Jones (2023) distances (optional, feeds CMD M_G)
    # =========================================================================
    if run_bailer_jones:
        from malca.ltv.cmd import fetch_bailer_jones_distances

        if verbose:
            print("-" * 60)
            print("STAGE 4c: BAILER-JONES DISTANCES")
            print("-" * 60)
        df = fetch_bailer_jones_distances(
            df,
            chunk_size=chunk_size,
            n_workers=n_workers,
            verbose=verbose,
        )
        print()

    # =========================================================================
    # Stage 5: CMD features / grouping (optional scaffolding)
    # =========================================================================
    if run_cmd:
        from malca.ltv.cmd import compute_cmd_features, assign_cmd_groups, load_mist_grid

        if verbose:
            print("-" * 60)
            print("STAGE 5: CMD FEATURES")
            print("-" * 60)
        df = compute_cmd_features(df)
        if mist_path is not None:
            _ = load_mist_grid(mist_path)
        df = assign_cmd_groups(df, boundaries=cmd_boundaries)
        print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        print("=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Final: {len(df):,} candidates ({len(df)/n0*100:.2f}% of input)")
        print()
        
        # Summary statistics
        if "vsx_name" in df.columns:
            n_vsx = df["vsx_name"].notna().sum()
            pct = n_vsx/len(df)*100 if len(df) > 0 else 0
            print(f"Previously classified (VSX): {n_vsx:,} ({pct:.1f}%)")
        
        if "milliquas_name" in df.columns:
            n_agn = df["milliquas_name"].notna().sum()
            print(f"AGN (MILLIQUAS): {n_agn:,}")
        
        if "ls_fap" in df.columns:
            n_periodic = (df["ls_fap"] < 0.1).sum()
            pct = n_periodic/len(df)*100 if len(df) > 0 else 0
            print(f"Periodic sources (FAP < 0.1): {n_periodic:,} ({pct:.1f}%)")
        
        if "neowise_n_epochs" in df.columns:
            n_neowise = (df["neowise_n_epochs"] > 0).sum()
            pct = n_neowise/len(df)*100 if len(df) > 0 else 0
            print(f"With NEOWISE data: {n_neowise:,} ({pct:.1f}%)")

        if "dust_candidate" in df.columns:
            n_dust = df["dust_candidate"].sum()
            pct = n_dust/len(df)*100 if len(df) > 0 else 0
            print(f"Dust candidates: {n_dust:,} ({pct:.1f}%)")

        if "stoch_sf_ml_amplitude" in df.columns:
            n_stoch = df["stoch_sf_ml_amplitude"].notna().sum()
            pct = n_stoch/len(df)*100 if len(df) > 0 else 0
            print(f"With stochastic post-filter features: {n_stoch:,} ({pct:.1f}%)")
        
        print("=" * 60)

    # Output parquet: every light curve that survived filter_max_diff_threshold,
    # with filter_reason = "passed" or the filter that removed it (e.g. south_pole, crowding).
    df = df.copy()
    if "filter_reason" not in df.columns:
        df["filter_reason"] = "passed"
    if df_rejected is not None and not df_rejected.empty:
        df = pd.concat([df, df_rejected], ignore_index=True)
    return df


# =============================================================================
# LTV ORCHESTRATOR
# =============================================================================

def _json_stable(value):
    if isinstance(value, Path):
        return str(value.expanduser())
    if isinstance(value, dict):
        return {str(k): _json_stable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_stable(v) for v in value]
    return value


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _ltv_code_fingerprint() -> dict[str, str | None]:
    module_dir = Path(__file__).resolve().parent.parent
    return {
        rel_path: _sha256_file(module_dir / rel_path)
        for rel_path in LTV_CODE_FINGERPRINT_FILES
        if (module_dir / rel_path).exists()
    }


def _ltv_run_fingerprint(args: argparse.Namespace, mag_bins: list[str]) -> dict:
    params = {
        key: _json_stable(getattr(args, key, None))
        for key in sorted(LTV_ORCHESTRATOR_CONFIG_DEFAULTS)
    }
    params["mag_bin"] = list(mag_bins)
    return {
        "version": LTV_RUN_REUSE_FINGERPRINT_VERSION,
        "params": params,
        "code": _ltv_code_fingerprint(),
    }


def _fingerprint_digest(fingerprint: dict) -> str:
    payload = json.dumps(_json_stable(fingerprint), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_ltv_timestamp_run_dir() -> Path:
    return DEFAULT_LTV_RUN_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_ltv_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return Path(args.run_dir).expanduser()
    if args.import_bundle is not None:
        bundle_path = Path(args.import_bundle).expanduser()
        stem = bundle_path.stem.removesuffix("_bundle")
        return DEFAULT_LTV_RUN_DIR / stem
    return default_ltv_timestamp_run_dir()


def _update_latest_symlink(run_dir: Path) -> None:
    latest = DEFAULT_LTV_RUN_DIR / "latest"
    try:
        if run_dir.resolve() == DEFAULT_LTV_RUN_DIR.resolve():
            return
        latest.parent.mkdir(parents=True, exist_ok=True)
        if latest.is_symlink() or latest.exists():
            if latest.is_dir() and not latest.is_symlink():
                shutil.rmtree(latest)
            else:
                latest.unlink()
        latest.symlink_to(run_dir.resolve(), target_is_directory=True)
    except Exception as exc:
        print(f"Warning: could not update LTV latest symlink: {exc}")


def _clear_parquet_dataset(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        for child in path.glob("chunk_*.parquet*"):
            child.unlink()
        return
    path.unlink()


def _clear_ltv_core_outputs(output_path: Path) -> None:
    _clear_parquet_dataset(output_path)
    output_path.with_name(f"{output_path.stem}_PROCESSED.txt").unlink(missing_ok=True)


def _stage_runs_ltv_upstream(stage: str) -> bool:
    return str(stage) in {"full", "cluster", "full-extended"}


def _stage_runs_ltv_downstream(stage: str) -> bool:
    return str(stage) in {"full", "home", "full-extended"}


def _stage_defaults_to_ltv_extended(stage: str) -> bool:
    return str(stage) == "full-extended"


def _passing_ltv_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "failed_any" not in df.columns:
        return df.copy()
    failed = df["failed_any"]
    if pd.api.types.is_bool_dtype(failed):
        mask = ~failed.fillna(False).astype(bool)
    elif pd.api.types.is_numeric_dtype(failed):
        mask = failed.fillna(0).astype(float) == 0.0
    else:
        lowered = failed.astype("string").str.strip().str.lower()
        mask = ~lowered.isin({"1", "true", "t", "yes", "y"}).fillna(False)
    return df.loc[mask].reset_index(drop=True).copy()


def _ensure_ltv_candidate_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "candidate_id" in out.columns:
        out["candidate_id"] = out["candidate_id"].astype(str)
        return out
    if "asas_sn_id" in out.columns:
        out["candidate_id"] = "ltv_" + out["asas_sn_id"].astype(str)
    return out


def _merge_ltv_candidate_columns(
    base: pd.DataFrame,
    extra: pd.DataFrame,
    value_cols: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    if extra.empty:
        return base
    out = _ensure_ltv_candidate_id(base)
    extra = _ensure_ltv_candidate_id(extra)
    if "candidate_id" not in out.columns or "candidate_id" not in extra.columns:
        return out

    cols = [col for col in value_cols if col in extra.columns]
    if not cols:
        return out

    merge_df = extra[["candidate_id", *cols]].drop_duplicates(subset=["candidate_id"], keep="last")
    merged = out.merge(merge_df, on="candidate_id", how="left", suffixes=("", "_ltv_new"))
    for col in cols:
        new_col = f"{col}_ltv_new"
        if new_col in merged.columns:
            merged[col] = merged[new_col].where(merged[new_col].notna(), merged[col] if col in merged.columns else np.nan)
            merged = merged.drop(columns=[new_col])
    return merged


def classify_ltv_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Assign deterministic source-level LTV classes from LTV features."""
    out = df.copy()
    out["ltv_class"] = "ltv_candidate"
    out["ltv_class_reason"] = "passed_ltv_filters"
    out["ltv_interest_score"] = 3

    def bool_col(name: str) -> pd.Series:
        if name not in out.columns:
            return pd.Series(False, index=out.index, dtype=bool)
        s = out[name]
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False).astype(bool)
        if pd.api.types.is_numeric_dtype(s):
            return s.fillna(0).astype(float) != 0.0
        return s.astype("string").str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})

    def has_text(name: str) -> pd.Series:
        if name not in out.columns:
            return pd.Series(False, index=out.index, dtype=bool)
        s = out[name]
        return s.notna() & (s.astype(str).str.strip() != "")

    def assign(mask: pd.Series, label: str, reason: str, score: int) -> None:
        idx = mask.fillna(False).astype(bool)
        out.loc[idx, "ltv_class"] = label
        out.loc[idx, "ltv_class_reason"] = reason
        out.loc[idx, "ltv_interest_score"] = int(score)

    cmd_group = (
        out["cmd_group"].astype("string").str.lower()
        if "cmd_group" in out.columns
        else pd.Series("", index=out.index, dtype="string")
    )
    known_periodic = (
        bool_col("period_ztf_periodic_match")
        | bool_col("period_ogle_match")
        | bool_col("periodic_flag")
    )
    motion = (
        bool_col("high_pm_flag")
        | bool_col("neighbor_pm_contam")
        | bool_col("ltv_failed_high_pm")
        | bool_col("ltv_failed_neighbor_high_pm")
    )

    assign(cmd_group.str.contains("evolved", na=False), "evolved_star_candidate", "cmd_group", 3)
    assign(bool_col("dust_candidate") | bool_col("ltv_dust_candidate") | bool_col("dust_excess"), "dust_candidate", "dust_or_ir_excess", 4)
    assign(known_periodic, "known_periodic", "periodic_catalog_match", 1)
    assign(motion, "motion_contaminant", "proper_motion_or_neighbor_pm", 0)
    assign(has_text("milliquas_name") | bool_col("ltv_milliquas_match"), "agn", "milliquas_match", 1)
    return out


def _make_core_config(args: argparse.Namespace, mag_bin: str, run_dir: Path):
    from malca.config import LIGHT_CURVE_FILE_EXTENSION
    from malca.ltv.core import Config

    return Config(
        root=Path(args.root).expanduser(),
        mag_bin=mag_bin,
        output=ltv_core_output_path(mag_bin, run_dir),
        dspring=float(args.dspring),
        ra_is_deg=bool(args.ra_is_deg),
        max_seasons=int(args.max_seasons),
        min_points_per_season=int(args.min_points_per_season),
        min_seasons_for_quadratic=int(args.min_seasons_for_quadratic),
        write_per_dir=False,
        band_mode=str(args.band_mode),
        workers=int(args.workers),
        chunk_size=int(args.chunk_size or LTV_CORE_CHUNK_SIZE),
        overwrite=bool(args.overwrite),
        file_ext=str(args.extension or LIGHT_CURVE_FILE_EXTENSION),
    )


def _run_core_if_needed(args: argparse.Namespace, mag_bin: str, run_dir: Path) -> Path:
    from malca.ltv.core import run_mag_bin

    output_path = ltv_core_output_path(mag_bin, run_dir)
    if output_path.exists() and not args.overwrite:
        if args.verbose:
            print(f"[ltv-pipeline] Reusing core output: {output_path}")
        return output_path
    if args.overwrite:
        _clear_ltv_core_outputs(output_path)
    cfg = _make_core_config(args, mag_bin, run_dir)
    run_mag_bin(cfg)
    return output_path


def _write_filtered_audit(
    args: argparse.Namespace,
    mag_bin: str,
    run_dir: Path,
    *,
    query_gaia: bool,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    core_path = ltv_core_output_path(mag_bin, run_dir)
    filtered_path = ltv_filtered_output_path(mag_bin, run_dir)

    if filtered_path.exists() and not args.overwrite:
        audit = read_parquet_table(filtered_path)
        return filtered_path, audit, _passing_ltv_rows(audit)

    if not core_path.exists():
        raise FileNotFoundError(f"LTV core output not found: {core_path}")

    core_df = read_parquet_table(core_path)
    if args.skip_filters:
        audit = core_df.copy()
        for col in LTV_AUDIT_FAILED_COLUMNS.values():
            audit[col] = False
        audit["failed_any"] = False
        audit["ltv_passed_filters"] = True
        audit["ltv_filter_reason"] = None
        passers = audit.copy()
    else:
        audit, passers = apply_all_filters_audit(
            core_df,
            min_slope=args.min_slope,
            min_diff=args.min_diff,
            query_gaia=query_gaia,
            chunk_size=args.chunk_size,
            n_workers=args.workers,
            verbose=args.verbose,
            return_passers=True,
        )
    write_parquet_table(audit, filtered_path)
    if args.verbose:
        print(f"[ltv-pipeline] Saved LTV audit table to {filtered_path}")
    return filtered_path, audit, passers


def _write_pipeline_candidates(
    args: argparse.Namespace,
    mag_bin: str,
    run_dir: Path,
    passers: pd.DataFrame,
) -> tuple[Path, pd.DataFrame]:
    output_path = ltv_pipeline_output_path(mag_bin, run_dir)
    if output_path.exists() and not args.overwrite:
        df = read_parquet_table(output_path)
        return output_path, df

    df = run_full_pipeline(
        passers,
        min_slope=args.min_slope,
        min_diff=args.min_diff,
        run_filters=False,
        run_stochastic_postfilter=args.run_stochastic_postfilter,
        stochastic_include_drw=args.stochastic_include_drw,
        run_crossmatch=not args.skip_crossmatch,
        include_ztf_periodic=not args.no_ztf_periodic,
        include_ogle_periodic=not args.no_ogle_periodic,
        run_neowise=not args.skip_neowise,
        run_extinction=not args.skip_extinction,
        run_dust_flags=not args.skip_dust_flags,
        run_cmd=not args.skip_cmd,
        run_bailer_jones=not args.skip_bailer_jones,
        run_gaia_epoch=not args.skip_gaia_epoch,
        gaia_epoch_table=args.gaia_epoch_table,
        gaia_epoch_data_release=args.gaia_epoch_data_release,
        gaia_epoch_data_structure=args.gaia_epoch_data_structure,
        gaia_epoch_valid_data=not args.gaia_epoch_include_invalid,
        gaia_epoch_band=args.gaia_epoch_band,
        n_workers=args.workers,
        chunk_size=args.chunk_size,
        log_csv=args.log_rejections,
        verbose=args.verbose,
    )
    df = classify_ltv_candidates(df)
    if "_idx" in df.columns:
        df = df.drop(columns=["_idx"])
    if "asas_sn_id" in df.columns:
        df["asas_sn_id"] = df["asas_sn_id"].astype(object).map(lambda x: str(x) if pd.notna(x) else "")
    write_parquet_table(df, output_path)
    if args.verbose:
        print(f"[ltv-pipeline] Saved enriched LTV candidates to {output_path}")
    return output_path, df


def _write_ltv_external_lcs(
    args: argparse.Namespace,
    mag_bin: str,
    run_dir: Path,
    candidates: pd.DataFrame,
) -> tuple[Path, Path, pd.DataFrame]:
    from malca.vetting import fetch_external_lcs

    output_path = ltv_external_lcs_output_path(mag_bin, run_dir)
    external_lc_dir = run_dir / "results" / "external_lcs"
    external_lc_dir.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        return output_path, external_lc_dir, read_parquet_table(output_path)

    run_df = _ensure_ltv_candidate_id(candidates)
    checkpoint_path = external_lc_dir / f"{output_path.stem}_CHECKPOINT.parquet"
    if args.overwrite:
        checkpoint_path.unlink(missing_ok=True)

    out = fetch_external_lcs(
        run_df,
        output_dir=external_lc_dir,
        run_atlas=bool(args.external_lc_atlas),
        run_ztf=True,
        run_gaia_epoch=True,
        run_tess=True,
        run_neowise=True,
        run_kepler=False,
        run_aavso=False,
        run_ps1=True,
        run_crts=True,
        atlas_token=args.atlas_token or os.environ.get("MALCA_ATLAS_TOKEN") or os.environ.get("ATLAS_API_TOKEN"),
        workers=int(args.external_lc_workers or 4),
        checkpoint_path=checkpoint_path,
        refresh_cache=bool(args.external_lc_refresh_cache),
    )
    write_parquet_table(out, output_path)
    checkpoint_path.unlink(missing_ok=True)
    if args.verbose:
        print(f"[ltv-pipeline] Saved external LC summary to {output_path}")
    return output_path, external_lc_dir, out


def _write_ltv_multi_survey_features(
    args: argparse.Namespace,
    mag_bin: str,
    run_dir: Path,
    candidates: pd.DataFrame,
    *,
    external_lc_dir: Path,
) -> tuple[Path, pd.DataFrame]:
    from malca.ltv.multi_survey import compute_ltv_multi_survey_features

    output_path = ltv_multi_survey_output_path(mag_bin, run_dir)
    if output_path.exists() and not args.overwrite:
        return output_path, read_parquet_table(output_path)

    out = compute_ltv_multi_survey_features(
        _ensure_ltv_candidate_id(candidates),
        external_lc_dir=external_lc_dir,
    )
    write_parquet_table(out, output_path)
    if args.verbose:
        print(f"[ltv-pipeline] Saved LTV multi-survey features to {output_path}")
    return output_path, out


def _write_ltv_extended_products(
    args: argparse.Namespace,
    mag_bin: str,
    run_dir: Path,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    updated = _ensure_ltv_candidate_id(candidates)
    pipeline_path = ltv_pipeline_output_path(mag_bin, run_dir)
    stats: dict[str, object] = {
        "external_lcs_output": None,
        "external_lcs_rows": 0,
        "ltv_multi_survey_output": None,
        "ltv_multi_survey_rows": 0,
    }
    external_lc_dir = run_dir / "results" / "external_lcs"

    if bool(args.run_external_lcs):
        from malca.external_lcs import EXTERNAL_LC_COLUMNS

        external_path, external_lc_dir, external_df = _write_ltv_external_lcs(args, mag_bin, run_dir, updated)
        updated = _merge_ltv_candidate_columns(updated, external_df, EXTERNAL_LC_COLUMNS)
        stats["external_lcs_output"] = str(external_path)
        stats["external_lcs_rows"] = int(len(external_df))

    if bool(args.run_multi_survey_features):
        from malca.ltv.multi_survey import LTV_MS_FEATURE_COLUMNS

        multi_path, multi_df = _write_ltv_multi_survey_features(
            args,
            mag_bin,
            run_dir,
            updated,
            external_lc_dir=external_lc_dir,
        )
        updated = _merge_ltv_candidate_columns(updated, multi_df, LTV_MS_FEATURE_COLUMNS)
        stats["ltv_multi_survey_output"] = str(multi_path)
        stats["ltv_multi_survey_rows"] = int(len(multi_df))

    if bool(args.run_external_lcs) or bool(args.run_multi_survey_features):
        write_parquet_table(updated, pipeline_path)
    return updated, stats


def _merge_outputs(paths: list[Path], output_path: Path) -> pd.DataFrame:
    frames = [read_parquet_table(path) for path in paths if path.exists()]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_parquet_table(merged, output_path)
    return merged


def _collect_ltv_candidate_lightcurves(run_dir: Path) -> list[tuple[Path, str]]:
    results_dir = run_dir / "results"
    if not results_dir.exists():
        return []

    tables = sorted(results_dir.glob("*_pipeline.parquet"))
    if not tables:
        tables = sorted(results_dir.glob("*_filtered.parquet"))

    seen_paths: set[Path] = set()
    used_arcnames: set[str] = set()
    collected: list[tuple[Path, str]] = []

    for table_path in tables:
        try:
            df = read_parquet_table(table_path)
        except Exception:
            continue
        if "failed_any" in df.columns:
            df = _passing_ltv_rows(df)
        if "lc_path" not in df.columns:
            continue

        for raw_path in df["lc_path"].dropna().astype(str):
            path = Path(raw_path).expanduser()
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in seen_paths or not path.exists():
                continue
            seen_paths.add(resolved)
            arcname = f"bundle_assets/lightcurves/{path.name}"
            if arcname in used_arcnames:
                arcname = f"bundle_assets/lightcurves/{len(used_arcnames):06d}_{path.name}"
            used_arcnames.add(arcname)
            collected.append((path, arcname))

    return collected


def _export_ltv_run_bundle(run_dir: Path, bundle_zip: Path, *, full_bundle: bool = False) -> list[str]:
    include_roots = [run_dir / "results", run_dir / "review"]
    if full_bundle:
        include_roots.append(run_dir / "bundle_assets")
    include_files = [
        run_dir / "run_params.json",
        run_dir / "run_summary.json",
        run_dir / "run.log",
    ]

    files: set[Path] = set()
    for path in include_files:
        if path.exists() and path.is_file():
            files.add(path)
    for root in include_roots:
        if root.exists():
            files.update(p for p in root.rglob("*") if p.is_file())

    if not files:
        raise FileNotFoundError(f"No LTV run files found under {run_dir}")

    bundle_zip = Path(bundle_zip).expanduser()
    bundle_zip.parent.mkdir(parents=True, exist_ok=True)
    files.discard(bundle_zip)

    names: list[str] = []
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for path in sorted(files, key=lambda p: str(p.relative_to(run_dir))):
            arcname = str(path.relative_to(run_dir))
            zf.write(path, arcname)
            names.append(arcname)
        if full_bundle:
            for path, arcname in _collect_ltv_candidate_lightcurves(run_dir):
                if arcname in names:
                    continue
                zf.write(path, arcname)
                names.append(arcname)
    return names


def _import_ltv_run_bundle(bundle_zip: Path, run_dir: Path, *, overwrite: bool = False) -> None:
    bundle_zip = Path(bundle_zip).expanduser()
    if not bundle_zip.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_zip}")
    if not zipfile.is_zipfile(bundle_zip):
        raise ValueError(f"Bundle is not a valid zip file: {bundle_zip}")
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_zip, "r") as zf:
        zf.extractall(run_dir)


def add_ltv_pipeline_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g_io = parser.add_argument_group("Input / output")
    g_stage = parser.add_argument_group("Stage")
    g_filters = parser.add_argument_group("Filters")
    g_core = parser.add_argument_group("Core metrics")
    g_crossmatch = parser.add_argument_group("Crossmatch")
    g_neowise = parser.add_argument_group("NEOWISE")
    g_extinction = parser.add_argument_group("Extinction & dust")
    g_gaia_epoch = parser.add_argument_group("Gaia epoch photometry")
    g_bailer_jones = parser.add_argument_group("Bailer-Jones")
    g_cmd = parser.add_argument_group("CMD")
    g_stochastic = parser.add_argument_group("Stochastic")
    g_external_lcs = parser.add_argument_group("External light-curve enrichment")
    g_bundle = parser.add_argument_group("Bundle")
    g_review = parser.add_argument_group("Review")
    g_general = parser.add_argument_group("General")

    g_io.add_argument("--mag-bin", nargs="+", default=["13_13.5"], choices=[*MAG_BINS, "all"])
    g_io.add_argument("--root", type=Path, default=LCV2_ROOT, help="Raw ASAS-SN light-curve root")
    g_io.add_argument("--run-dir", type=Path, default=None, help="LTV run directory (default: output/runs/ltv/<timestamp>)")
    g_stage.add_argument("--stage", choices=LTV_PIPELINE_STAGE_CHOICES, default="full")
    g_filters.add_argument("--min-slope", type=float, default=LTV_MIN_SLOPE)
    g_filters.add_argument("--min-diff", type=float, default=LTV_MIN_DIFF)
    g_core.add_argument("--dspring", type=float, default=LTV_DSPRING)
    g_core.add_argument("--ra-is-deg", action=argparse.BooleanOptionalAction, default=True)
    g_core.add_argument("--max-seasons", type=int, default=LTV_MAX_SEASONS)
    g_core.add_argument("--min-points-per-season", type=int, default=LTV_MIN_POINTS_PER_SEASON)
    g_core.add_argument("--min-seasons-for-quadratic", type=int, default=LTV_MIN_SEASONS_FOR_QUADRATIC)
    g_core.add_argument("--band-mode", choices=["pipeline", "g_only"], default="pipeline")
    g_core.add_argument("--extension", "-e", type=str, default=None)
    g_filters.add_argument("--skip-filters", action="store_true")
    g_crossmatch.add_argument("--skip-crossmatch", action="store_true")
    g_crossmatch.add_argument("--no-ztf-periodic", action="store_true")
    g_crossmatch.add_argument("--no-ogle-periodic", action="store_true")
    g_neowise.add_argument("--skip-neowise", action="store_true")
    g_extinction.add_argument("--skip-extinction", action="store_true")
    g_extinction.add_argument("--skip-dust-flags", action="store_true")
    g_gaia_epoch.add_argument("--skip-gaia-epoch", action="store_true")
    g_gaia_epoch.add_argument("--gaia-epoch-table", type=str, default=None)
    g_gaia_epoch.add_argument("--gaia-epoch-data-release", type=str, default=GAIA_EPOCH_DATA_RELEASE)
    g_gaia_epoch.add_argument("--gaia-epoch-data-structure", type=str, default=GAIA_EPOCH_DATA_STRUCTURE)
    g_gaia_epoch.add_argument("--gaia-epoch-band", type=str, default=None)
    g_gaia_epoch.add_argument("--gaia-epoch-include-invalid", action="store_true")
    g_bailer_jones.add_argument("--skip-bailer-jones", action="store_true")
    g_cmd.add_argument("--skip-cmd", action="store_true")
    g_stochastic.add_argument("--run-stochastic-postfilter", action="store_true")
    g_stochastic.add_argument("--stochastic-include-drw", action="store_true")
    g_external_lcs.add_argument("--run-external-lcs", dest="run_external_lcs", action="store_true", default=None)
    g_external_lcs.add_argument("--no-external-lcs", dest="run_external_lcs", action="store_false")
    g_external_lcs.add_argument("--run-multi-survey-features", dest="run_multi_survey_features", action="store_true", default=None)
    g_external_lcs.add_argument("--no-multi-survey-features", dest="run_multi_survey_features", action="store_false")
    g_external_lcs.add_argument("--external-lc-workers", type=int, default=4)
    g_external_lcs.add_argument("--external-lc-refresh-cache", action="store_true")
    g_external_lcs.add_argument("--external-lc-atlas", action="store_true")
    g_external_lcs.add_argument("--atlas-token", "--external-lc-atlas-token", dest="atlas_token", type=str, default=None)
    g_bundle.add_argument("--import-bundle", type=Path, default=None)
    bundle_group = g_bundle.add_mutually_exclusive_group()
    bundle_group.add_argument("--export-bundle", type=Path, default=None)
    bundle_group.add_argument("--no-export-bundle", dest="export_bundle_enabled", action="store_false")
    g_bundle.add_argument("--full-bundle", action="store_true")
    g_review.add_argument("--review-db", type=Path, default=None)
    g_review.add_argument("--run-vetting", action="store_true")
    g_review.add_argument("--skip-stats", action="store_true")
    g_review.add_argument("--stats-compute-ls", action="store_true")
    g_review.add_argument("--index-file", type=Path, default=None)
    g_review.add_argument("--no-review-sync", dest="review_sync_enabled", action="store_false")
    g_review.add_argument("--review-sync-dir", type=Path, default=Path("reviews"))
    g_review.add_argument("--review-sync-hash-assets", action="store_true")
    add_config_args(g_general)
    g_general.add_argument("--log-rejections", type=Path, default=None)
    g_general.add_argument("--workers", type=int, default=LTV_WORKERS)
    g_general.add_argument("--chunk-size", type=int, default=LTV_CHUNK_SIZE)
    g_general.add_argument("-o", "--overwrite", action="store_true")
    g_general.add_argument("-v", "--verbose", action="store_true")

    parser.set_defaults(**LTV_BUILD_CONFIG_DEFAULTS)
    return parser


def _normalize_mag_bins(raw_bins: list[str] | str) -> list[str]:
    if isinstance(raw_bins, str):
        raw_bins = [raw_bins]
    if "all" in raw_bins:
        if len(raw_bins) > 1:
            raise SystemExit("Cannot mix 'all' with specific magnitude bins")
        return list(reversed(MAG_BINS))
    return list(raw_bins)


def run_ltv_pipeline_cli(args: argparse.Namespace) -> dict:
    apply_config(
        args,
        command="ltv-pipeline",
        valid_keys=set(vars(args)) | set(LTV_ORCHESTRATOR_CONFIG_DEFAULTS),
        path_keys=LTV_ORCHESTRATOR_CONFIG_PATH_KEYS,
    )

    mag_bins = _normalize_mag_bins(args.mag_bin)
    if args.run_external_lcs is None:
        args.run_external_lcs = _stage_defaults_to_ltv_extended(args.stage)
    if args.run_multi_survey_features is None:
        args.run_multi_survey_features = _stage_defaults_to_ltv_extended(args.stage)
    if args.stage == "cluster":
        if args.run_external_lcs or args.run_multi_survey_features:
            print("Info: --stage cluster runs raw-dependent LTV products only. Extended external-LC work is skipped.")
        args.run_external_lcs = False
        args.run_multi_survey_features = False

    run_dir = _resolve_ltv_run_dir(args)
    if args.import_bundle is not None:
        _import_ltv_run_bundle(args.import_bundle, run_dir, overwrite=args.overwrite)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir = run_dir / "results"
    review_dir = run_dir / "review"
    results_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = _ltv_run_fingerprint(args, mag_bins)
    fingerprint_hash = _fingerprint_digest(fingerprint)
    started = datetime.now()
    started_perf = time.perf_counter()
    cmd = shlex.join(getattr(sys, "orig_argv", None) or ([sys.executable] + sys.argv))

    run_params = {
        "timestamp": started.isoformat(),
        "command": cmd,
        "stage": args.stage,
        "mag_bin": mag_bins,
        "root": str(Path(args.root).expanduser()),
        "run_dir": str(run_dir),
        "results_dir": str(results_dir),
        "review_db": str(args.review_db or ltv_review_db_path(run_dir)),
        "config": str(args.config) if getattr(args, "config", None) else None,
        "profile": args.profile,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "min_slope": args.min_slope,
        "min_diff": args.min_diff,
        "skip_filters": args.skip_filters,
        "skip_crossmatch": args.skip_crossmatch,
        "no_ztf_periodic": args.no_ztf_periodic,
        "no_ogle_periodic": args.no_ogle_periodic,
        "skip_neowise": args.skip_neowise,
        "skip_extinction": args.skip_extinction,
        "skip_dust_flags": args.skip_dust_flags,
        "skip_gaia_epoch": args.skip_gaia_epoch,
        "skip_bailer_jones": args.skip_bailer_jones,
        "skip_cmd": args.skip_cmd,
        "run_stochastic_postfilter": args.run_stochastic_postfilter,
        "run_external_lcs": args.run_external_lcs,
        "run_multi_survey_features": args.run_multi_survey_features,
        "external_lc_workers": args.external_lc_workers,
        "external_lc_refresh_cache": args.external_lc_refresh_cache,
        "external_lc_atlas": args.external_lc_atlas,
        "atlas_token": bool(args.atlas_token),
        "overwrite": args.overwrite,
        "import_bundle": str(args.import_bundle) if args.import_bundle else None,
        "export_bundle": str(args.export_bundle) if args.export_bundle else None,
        "export_bundle_enabled": args.export_bundle_enabled,
        "full_bundle": args.full_bundle,
        "review_sync_enabled": args.review_sync_enabled,
        "review_sync_dir": str(args.review_sync_dir),
        "run_reuse_fingerprint": fingerprint,
        "run_reuse_fingerprint_hash": fingerprint_hash,
    }
    (run_dir / "run_params.json").write_text(json.dumps(run_params, indent=2, default=str), encoding="ascii")
    (run_dir / "run.log").write_text(
        "\n".join([
            f"timestamp: {started.isoformat()}",
            f"command: {cmd}",
            f"stage: {args.stage}",
            f"run_dir: {run_dir}",
            f"results_dir: {results_dir}",
            f"review_db: {args.review_db or ltv_review_db_path(run_dir)}",
        ]) + "\n",
        encoding="ascii",
    )

    per_bin: dict[str, dict] = {}
    filtered_paths: list[Path] = []
    pipeline_paths: list[Path] = []
    external_lcs_paths: list[Path] = []
    ltv_multi_survey_paths: list[Path] = []

    for mag_bin in mag_bins:
        bin_started = time.perf_counter()
        if _stage_runs_ltv_upstream(args.stage):
            core_path = _run_core_if_needed(args, mag_bin, run_dir)
        else:
            core_path = ltv_core_output_path(mag_bin, run_dir)

        filtered_path: Path | None = None
        pipeline_path: Path | None = None
        extended_stats: dict[str, object] = {
            "external_lcs_output": None,
            "external_lcs_rows": 0,
            "ltv_multi_survey_output": None,
            "ltv_multi_survey_rows": 0,
        }
        audit_rows = passing_rows = candidate_rows = 0

        if args.stage in LTV_PIPELINE_STAGE_CHOICES:
            query_gaia = args.stage != "cluster"
            if core_path.exists():
                filtered_path, audit_df, passers = _write_filtered_audit(
                    args,
                    mag_bin,
                    run_dir,
                    query_gaia=query_gaia,
                )
            else:
                filtered_path = ltv_filtered_output_path(mag_bin, run_dir)
                if not filtered_path.exists():
                    raise FileNotFoundError(f"Home stage needs {core_path} or {filtered_path}")
                audit_df = read_parquet_table(filtered_path)
                passers = _passing_ltv_rows(audit_df)
            audit_rows = len(audit_df)
            passing_rows = len(passers)
            filtered_paths.append(filtered_path)

            if _stage_runs_ltv_downstream(args.stage):
                pipeline_path, candidates = _write_pipeline_candidates(args, mag_bin, run_dir, passers)
                candidates, extended_stats = _write_ltv_extended_products(args, mag_bin, run_dir, candidates)
                candidate_rows = len(candidates)
                pipeline_paths.append(pipeline_path)
                if extended_stats.get("external_lcs_output"):
                    external_lcs_paths.append(Path(str(extended_stats["external_lcs_output"])))
                if extended_stats.get("ltv_multi_survey_output"):
                    ltv_multi_survey_paths.append(Path(str(extended_stats["ltv_multi_survey_output"])))

        per_bin[mag_bin] = {
            "core_output": str(core_path),
            "filtered_output": str(filtered_path) if filtered_path else None,
            "pipeline_output": str(pipeline_path) if pipeline_path else None,
            "audit_rows": int(audit_rows),
            "passing_rows": int(passing_rows),
            "candidate_rows": int(candidate_rows),
            **extended_stats,
            "elapsed_sec": round(time.perf_counter() - bin_started, 3),
        }

    merged_filtered_rows = merged_pipeline_rows = merged_external_lcs_rows = merged_ltv_multi_survey_rows = 0
    if len(filtered_paths) > 1:
        merged_filtered_rows = len(_merge_outputs(filtered_paths, ltv_all_filtered_output_path(run_dir)))
    if len(pipeline_paths) > 1:
        merged_pipeline_rows = len(_merge_outputs(pipeline_paths, ltv_all_pipeline_output_path(run_dir)))
    if len(external_lcs_paths) > 1:
        merged_external_lcs_rows = len(_merge_outputs(external_lcs_paths, ltv_all_external_lcs_output_path(run_dir)))
    if len(ltv_multi_survey_paths) > 1:
        merged_ltv_multi_survey_rows = len(_merge_outputs(ltv_multi_survey_paths, ltv_all_multi_survey_output_path(run_dir)))

    review_stats = None
    review_db_path = Path(args.review_db).expanduser() if args.review_db else ltv_review_db_path(run_dir)
    if _stage_runs_ltv_downstream(args.stage) and pipeline_paths:
        from malca.ltv.review import ingest_ltv_results

        ingest_df = (
            _merge_outputs(pipeline_paths, ltv_all_pipeline_output_path(run_dir))
            if len(pipeline_paths) > 1
            else read_parquet_table(pipeline_paths[0])
        )
        total, new = ingest_ltv_results(
            review_db_path,
            ingest_df,
            run_characterize=False,
            run_vetting=args.run_vetting,
            run_stats=not args.skip_stats,
            stats_compute_ls=args.stats_compute_ls,
            n_workers=args.workers,
            index_path=args.index_file,
            source_path=run_dir,
            verbose=args.verbose,
        )
        review_stats = {"review_db": str(review_db_path), "total": int(total), "new": int(new)}
        if args.review_sync_enabled:
            auto_export_review_bundle(
                review_db_path,
                args.review_sync_dir,
                hash_assets=bool(args.review_sync_hash_assets),
                logger=print if args.verbose else (lambda _msg: None),
            )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "command": cmd,
        "stage": args.stage,
        "run_dir": str(run_dir),
        "root": str(Path(args.root).expanduser()),
        "mag_bin": mag_bins,
        "elapsed_sec": round(time.perf_counter() - started_perf, 3),
        "run_reuse_fingerprint_hash": fingerprint_hash,
        "per_bin": per_bin,
        "merged_filtered_rows": int(merged_filtered_rows),
        "merged_pipeline_rows": int(merged_pipeline_rows),
        "merged_external_lcs_rows": int(merged_external_lcs_rows),
        "merged_ltv_multi_survey_rows": int(merged_ltv_multi_survey_rows),
        "review": review_stats,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="ascii")
    _update_latest_symlink(run_dir)

    if args.export_bundle_enabled:
        bundle_path = args.export_bundle or run_dir / f"{run_dir.name}_bundle.zip"
        try:
            bundled = _export_ltv_run_bundle(run_dir, bundle_path, full_bundle=args.full_bundle)
            summary["export_bundle"] = {"path": str(bundle_path), "files": len(bundled)}
            (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="ascii")
        except Exception as exc:
            print(f"Error creating LTV export bundle: {exc}")

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = add_ltv_pipeline_args(argparse.ArgumentParser(
        prog="malca ltv-pipeline",
        description="Run the LTV workflow with STV-style run metadata, audit filtering, and review ingest.",
    ))
    args = parser.parse_args(argv)
    run_ltv_pipeline_cli(args)


# =============================================================================
# LEGACY BUILD SUBCOMMAND INTERNALS
# =============================================================================

def add_pipeline_args(parser):
    """Add pipeline arguments to argparse."""
    g_io = parser.add_argument_group("Input / output")
    g_filters = parser.add_argument_group("Filters")
    g_crossmatch = parser.add_argument_group("Crossmatch")
    g_neowise = parser.add_argument_group("NEOWISE")
    g_extinction = parser.add_argument_group("Extinction & dust")
    g_gaia_epoch = parser.add_argument_group("Gaia epoch photometry")
    g_bailer_jones = parser.add_argument_group("Bailer-Jones")
    g_cmd = parser.add_argument_group("CMD")
    g_logging = parser.add_argument_group("Logging")
    g_general = parser.add_argument_group("General")

    g_io.add_argument(
        "--mag-bin",
        type=str,
        default=None,
        choices=["12_12.5", "12.5_13", "13_13.5", "13.5_14", "14_14.5", "14.5_15"],
        help="Magnitude bin (auto-resolves input/output in <run-dir>/results/)",
    )
    g_io.add_argument(
        "--run-dir",
        type=str,
        default=str(DEFAULT_LTV_RUN_DIR),
        help="LTV run directory for default inputs/outputs (default: output/runs/ltv)",
    )
    g_io.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input Parquet from ltv.core (default: <run-dir>/results/LTvar<MAG>.parquet)",
    )
    g_io.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output Parquet for pipeline results (default: <run-dir>/results/LTvar<MAG>_pipeline.parquet)",
    )
    g_filters.add_argument(
        "--min-slope",
        type=float,
        default=LTV_MIN_SLOPE,
        help="Minimum |Slope| threshold (mag/yr)",
    )
    g_filters.add_argument(
        "--min-diff",
        type=float,
        default=LTV_MIN_DIFF,
        help="Minimum |max diff| threshold (mag)",
    )
    g_general.add_argument(
        "--workers",
        type=int,
        default=LTV_WORKERS,
        help="Number of parallel workers",
    )
    g_general.add_argument(
        "--chunk-size",
        type=int,
        default=LTV_CHUNK_SIZE,
        help="Chunk size for batch queries",
    )
    add_config_args(g_general)
    g_general.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress",
    )
    parser.set_defaults(**LTV_BUILD_CONFIG_DEFAULTS)
    return parser


def run_pipeline_cli(args):
    """Run pipeline from CLI arguments."""
    apply_config(
        args,
        command="ltv-pipeline",
        valid_keys=set(vars(args)) | set(LTV_BUILD_CONFIG_DEFAULTS),
        path_keys=LTV_BUILD_CONFIG_PATH_KEYS,
    )
    # Resolve input/output from --mag-bin if not given explicitly
    if args.input is None:
        if args.mag_bin is None:
            raise SystemExit("Error: must specify --input or --mag-bin")
        args.input = str(ltv_core_output_path(args.mag_bin, args.run_dir))
    if args.output is None:
        if args.mag_bin is None:
            raise SystemExit("Error: must specify --output or --mag-bin")
        args.output = str(ltv_pipeline_output_path(args.mag_bin, args.run_dir))

    # Load input
    input_path = Path(args.input)
    df = read_parquet_table(input_path)
    
    print(f"Loaded {len(df):,} sources from {input_path}")
    
    # Run pipeline
    df = run_full_pipeline(
        df,
        min_slope=args.min_slope,
        min_diff=args.min_diff,
        run_filters=not args.skip_filters,
        run_stochastic_postfilter=args.run_stochastic_postfilter,
        stochastic_include_drw=args.stochastic_include_drw,
        run_crossmatch=not args.skip_crossmatch,
        include_ztf_periodic=not args.no_ztf_periodic,
        include_ogle_periodic=not args.no_ogle_periodic,
        run_neowise=not args.skip_neowise,
        run_extinction=not args.skip_extinction,
        run_dust_flags=not args.skip_dust_flags,
        run_cmd=not args.skip_cmd,
        run_bailer_jones=not args.skip_bailer_jones,
        run_gaia_epoch=not args.skip_gaia_epoch,
        gaia_epoch_table=args.gaia_epoch_table,
        gaia_epoch_data_release=args.gaia_epoch_data_release,
        gaia_epoch_data_structure=args.gaia_epoch_data_structure,
        gaia_epoch_valid_data=not args.gaia_epoch_include_invalid,
        gaia_epoch_band=args.gaia_epoch_band,
        n_workers=args.workers,
        chunk_size=args.chunk_size,
        log_csv=args.log_rejections,
        verbose=args.verbose,
    )
    
    # Normalize ID columns for Parquet (avoids ArrowTypeError on mixed int/str).
    # By position so every column with these names (including duplicates) is converted.
    id_col_names = ("asas_sn_id",)
    for i, name in enumerate(df.columns):
        if name in id_col_names:
            df.iloc[:, i] = df.iloc[:, i].astype(object).map(
                lambda x: str(x) if pd.notna(x) else ""
            )
    if "_idx" in df.columns:
        df = df.drop(columns=["_idx"])

    # Save output
    output_path = Path(args.output)
    write_parquet_table(df, output_path)
    
    print(f"Saved {len(df):,} candidates to {output_path}")
    return df


if __name__ == "__main__":
    main()
