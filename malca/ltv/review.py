"""
LTV → Review DB ingest bridge.

Maps LTV pipeline output columns to the review DB schema and ingests
candidates into a standalone LTV review DB (separate from STV candidates).

Usage:
    # CLI
    malca ltv-ingest --input ltv_build_output.csv --review-db ltv_candidates.db -v

    # Python API
    from malca.ltv.review import ingest_ltv_results
    n_total, n_new = ingest_ltv_results("ltv_candidates.db", ltv_df)
"""
from __future__ import annotations

from pathlib import Path
import argparse
import os

from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from malca.config import ASASSN_INDEX_PATH
from malca.config import LTV_MAX_PM
from malca.review.store import db_connect, import_candidates
from malca.stats import compute_stats, _enrich_row_worker







# ---------------------------------------------------------------------------
# Column mapping: LTV pipeline output → review DB schema
# ---------------------------------------------------------------------------

# Core output columns from core.py → ltv_* names
_CORE_COL_MAP = {
    "Slope":       "ltv_slope",
    "Quad Slope":  "ltv_slope_quad",
    "max diff":    "ltv_max_diff",
    "Dispersion":  "ltv_dispersion",
    "Median":      "ltv_median",
    "Median_err":  "ltv_median_err",
    "n_seasons":   "ltv_n_seasons",
    "time_span_days": "ltv_time_span_days",
    "n_unique_nights": "ltv_n_unique_nights",
    "ls_period":   "ltv_ls_period",
    "ls_power":    "ltv_ls_power",
    "ls_fap":      "ltv_ls_fap",
    "coeff1":      "ltv_coeff1",
    "coeff2":      "ltv_coeff2",
    "vg_has_v":    "ltv_vg_has_v",
    "vg_overlap_days": "ltv_vg_overlap_days",
    "vg_overlap_fraction": "ltv_vg_overlap_fraction",
    "season_points_min": "ltv_season_points_min",
    "season_points_median": "ltv_season_points_median",
    "season_points_max": "ltv_season_points_max",
    "season_span_days_mean": "ltv_season_span_days_mean",
    "season_span_days_median": "ltv_season_span_days_median",
    "season_span_days_max": "ltv_season_span_days_max",
    "season_step_max_mag": "ltv_season_step_max_mag",
    "season_step_mean_abs_mag": "ltv_season_step_mean_abs_mag",
    "season_step_max_fraction": "ltv_season_step_max_fraction",
    "season_monotonicity_fraction": "ltv_season_monotonicity_fraction",
    "season_spearman_rho": "ltv_season_spearman_rho",
    "season_kendall_tau": "ltv_season_kendall_tau",
    "leave1out_slope_std": "ltv_leave1out_slope_std",
    "leave1out_slope_range": "ltv_leave1out_slope_range",
    "trend_slope_mag_per_year": "ltv_trend_slope_mag_per_year",
    "trend_quad_mag_per_year2": "ltv_trend_quad_mag_per_year2",
    "trend_slope_err_mag_per_year": "ltv_trend_slope_err_mag_per_year",
    "trend_slope_snr": "ltv_trend_slope_snr",
    "trend_r2": "ltv_trend_r2",
    "trend_delta_bic_linear": "ltv_trend_delta_bic_linear",
    "trend_delta_bic_quadratic": "ltv_trend_delta_bic_quadratic",
}

# Pipeline output columns (neowise, crossmatch, dust) → ltv_* names
_PIPELINE_COL_MAP = {
    # NEOWISE
    "w1_slope":         "ltv_neowise_w1_slope",
    "w1_w2_slope":      "ltv_neowise_w1_w2_slope",
    "neowise_n_epochs": "ltv_neowise_n_epochs",
    # Crossmatch
    "vsx_name":         "ltv_vsx_name",
    # Dust flags
    "dust_candidate":   "ltv_dust_candidate",
    "dust_excess":      "ltv_dust_excess",
    # Stochastic post-filter features
    "stoch_sf_ml_amplitude": "ltv_stoch_sf_ml_amplitude",
    "stoch_sf_ml_gamma":     "ltv_stoch_sf_ml_gamma",
    "stoch_iar_phi":         "ltv_stoch_iar_phi",
    "stoch_mhps_high":       "ltv_stoch_mhps_high",
    "stoch_mhps_low":        "ltv_stoch_mhps_low",
    "stoch_mhps_non_zero":   "ltv_stoch_mhps_non_zero",
    "stoch_mhps_pn_flag":    "ltv_stoch_mhps_pn_flag",
    "stoch_mhps_ratio":      "ltv_stoch_mhps_ratio",
    "stoch_gp_drw_sigma":    "ltv_stoch_gp_drw_sigma",
    "stoch_gp_drw_tau":      "ltv_stoch_gp_drw_tau",
}


def map_ltv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename LTV pipeline output columns to review DB schema names.

    Handles both core-only output (from ltv-core) and full pipeline output
    (from ltv-build, which adds crossmatch, NEOWISE, dust, extinction).

    Returns a new DataFrame with:
    - candidate_id = "ltv_{asas_sn_id}"
    - asas_sn_id preserved as the standard LTV source ID column
    - ltv_* columns from pipeline output
    - ra_deg, dec_deg preserved for characterization
    """
    df = df.copy()

    if "asas_sn_id" not in df.columns:
        raise ValueError("LTV DataFrame must have an 'asas_sn_id' column")

    df["asas_sn_id"] = df["asas_sn_id"].astype(str)

    df["candidate_id"] = "ltv_" + df["asas_sn_id"]

    # --- Map core columns ---
    for src, dst in _CORE_COL_MAP.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
            df = df.drop(columns=[src])

    # --- Map pipeline columns ---
    for src, dst in _PIPELINE_COL_MAP.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
            df = df.drop(columns=[src])

    # --- Derive boolean crossmatch flags from name columns ---
    if "ltv_vsx_name" in df.columns and "ltv_vsx_match" not in df.columns:
        df["ltv_vsx_match"] = df["ltv_vsx_name"].notna().astype(int)

    if "milliquas_name" in df.columns and "ltv_milliquas_match" not in df.columns:
        df["ltv_milliquas_match"] = df["milliquas_name"].notna().astype(int)

    if "gaia_alert_name" in df.columns and "ltv_gaia_alert_match" not in df.columns:
        df["ltv_gaia_alert_match"] = df["gaia_alert_name"].notna().astype(int)

    # --- All ingested candidates passed the LTV filters (implied by inclusion) ---
    if "ltv_passed_filters" not in df.columns:
        df["ltv_passed_filters"] = 1

    # --- Rename PanSTARRS g-mag to baseline_mag if not already present ---
    if "Pstarss gmag" in df.columns and "baseline_mag" not in df.columns:
        df["baseline_mag"] = df["Pstarss gmag"]
        df = df.drop(columns=["Pstarss gmag"])

    # --- Normalize Gaia proper motion outputs to review-standard names ---
    if "gaia_pmra" in df.columns and "pmra" not in df.columns:
        df["pmra"] = df["gaia_pmra"]
    if "gaia_pmdec" in df.columns and "pmdec" not in df.columns:
        df["pmdec"] = df["gaia_pmdec"]
    if "gaia_pm_total" in df.columns and "pm_total" not in df.columns:
        df["pm_total"] = df["gaia_pm_total"]
    if "pm_total" not in df.columns and {"pmra", "pmdec"}.issubset(df.columns):
        pmra = pd.to_numeric(df["pmra"], errors="coerce")
        pmdec = pd.to_numeric(df["pmdec"], errors="coerce")
        df["pm_total"] = np.sqrt(pmra * pmra + pmdec * pmdec)
    if "pm_total" in df.columns and "high_pm_flag" not in df.columns:
        pm_total = pd.to_numeric(df["pm_total"], errors="coerce")
        df["high_pm_flag"] = (pm_total > float(LTV_MAX_PM)).fillna(False).astype(int)

    # --- Cast int bool columns ---
    for col in (
        "ltv_passed_filters",
        "ltv_dust_candidate",
        "ltv_dust_excess",
        "ltv_vsx_match",
        "ltv_milliquas_match",
        "ltv_gaia_alert_match",
        "ltv_vg_has_v",
        "ltv_stoch_mhps_pn_flag",
        "high_pm_flag",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def enrich_with_stats(
    df: pd.DataFrame,
    *,
    compute_ls: bool = False,
    n_workers: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run compute_stats on each LTV candidate's raw light curve and merge
    the resulting stats_* columns into the DataFrame.

    Requires a 'lc_path' column pointing to the raw LC CSV file (added by
    core.py's process_one_lc). Rows without a valid lc_path are skipped.

    The flattening logic mirrors detect.py's --run-enrich step, producing
    the same stats_* column names that live in _CANDIDATE_COLUMNS.
    """
    if "lc_path" not in df.columns:
        if verbose:
            print("[ltv-ingest] No lc_path column — skipping compute_stats enrichment")
        return df

    valid_mask = df["lc_path"].notna() & (df["lc_path"].astype(str) != "")
    n_valid = valid_mask.sum()
    if n_valid == 0:
        if verbose:
            print("[ltv-ingest] No valid lc_path values — skipping compute_stats")
        return df

    if verbose:
        print(f"[ltv-ingest] Running compute_stats on {n_valid:,} candidates ({n_workers} workers)...")

    rows = df.to_dict("records")
    # Build parallel tasks; rows with no valid path pass through unchanged.
    tasks: list[tuple] = []
    task_indices: list[int] = []
    results: list[dict] = list(rows)  # default: unchanged

    for i, row in enumerate(rows):
        lc_path_str = row.get("lc_path", "")
        if not lc_path_str or pd.isna(lc_path_str):
            continue
        lc_path = Path(str(lc_path_str))
        if not lc_path.exists():
            continue
        asassn_id = lc_path.stem.split("-")[0]
        tasks.append((row, asassn_id, str(lc_path.parent), compute_ls))
        task_indices.append(i)

    with ProcessPoolExecutor(max_workers=max(1, n_workers)) as executor:
        for idx, result in zip(task_indices, tqdm(
            executor.map(_enrich_row_worker, tasks),
            total=len(tasks),
            desc="compute_stats",
            disable=not verbose,
        )):
            results[idx] = result

    return pd.DataFrame(results)


def _resolve_ltv_index_path(index_override: Path | None = None) -> Path | None:
    """Resolve the ASASSN index parquet path, trying common locations."""
    default = ASASSN_INDEX_PATH.expanduser()
    candidates = []
    if index_override is not None:
        candidates.append(Path(index_override).expanduser())
    candidates.append(default)
    # also search input/ relative to cwd
    for pattern in ("asassn_index*.parquet", "asassn_index*.pq"):
        candidates.extend(sorted(Path("input").glob(pattern)))
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def _add_gaia_ids_from_index_ltv(df: pd.DataFrame, index_path: Path, verbose: bool = True) -> pd.DataFrame:
    """
    Merge gaia_id from the ASASSN index into an LTV DataFrame.
    Only fills rows where gaia_id is currently missing.
    """
    if "asas_sn_id" not in df.columns:
        return df
    needs_fill = "gaia_id" not in df.columns or df["gaia_id"].isna().all()
    already_filled = "gaia_id" in df.columns and not df["gaia_id"].isna().any()
    if already_filled:
        return df
    try:
        if verbose:
            print(f"[ltv-ingest] Loading ASASSN index for gaia_id lookup: {index_path.name}")
        if index_path.suffix in (".parquet", ".pq"):
            df_idx = pd.read_parquet(index_path, columns=["asas_sn_id", "gaia_id"])
        else:
            df_idx = pd.read_csv(index_path, usecols=["asas_sn_id", "gaia_id"], low_memory=False)
        df_idx["asas_sn_id"] = pd.to_numeric(df_idx["asas_sn_id"], errors="coerce")
        df_idx = df_idx.dropna(subset=["asas_sn_id"])
        df_idx["asas_sn_id"] = df_idx["asas_sn_id"].astype("int64").astype(str)
        # Store gaia_id as clean integer string so it survives pandas dtype coercion
        # and passes _normalize_gaia_ids's str.isdigit() check (float "123.0" would fail).
        df_idx["gaia_id"] = pd.to_numeric(df_idx["gaia_id"], errors="coerce")
        df_idx = df_idx.dropna(subset=["gaia_id"])
        df_idx["gaia_id"] = df_idx["gaia_id"].astype("int64").astype(str)
        df_idx = df_idx.drop_duplicates(subset=["asas_sn_id"])

        out = df.copy()
        if "gaia_id" not in out.columns:
            out["gaia_id"] = pd.NA
        # merge index gaia_id, then fill only missing values
        merged = out.merge(df_idx.rename(columns={"gaia_id": "_gaia_id_idx"}), on="asas_sn_id", how="left")
        missing = out["gaia_id"].isna()
        out.loc[missing, "gaia_id"] = merged.loc[missing, "_gaia_id_idx"]
        n_filled = int(missing.sum()) - int(out["gaia_id"].isna().sum())
        if verbose:
            print(f"[ltv-ingest] Filled gaia_id for {n_filled}/{len(out)} candidates from ASASSN index")
        return out
    except Exception as e:
        if verbose:
            print(f"[ltv-ingest] Warning: gaia_id index lookup failed: {e}")
        return df


def ingest_ltv_results(
    db_path: str | Path,
    ltv_df: pd.DataFrame,
    *,
    run_characterize: bool = True,
    run_vetting: bool = False,
    run_stats: bool = True,
    stats_compute_ls: bool = False,
    n_workers: int = 1,
    index_path: Path | str | None = None,
    verbose: bool = True,
) -> tuple[int, int]:
    """
    Ingest LTV pipeline results into a standalone LTV review DB.

    Accepts either core.py output (raw seasonal trend metrics) or full
    pipeline output (with crossmatch, NEOWISE, dust, extinction columns).

    Args:
        db_path: Path to the LTV review SQLite DB (created if it doesn't exist).
        ltv_df: DataFrame from ltv-core or ltv-build.
        run_characterize: Run Gaia/dust characterization at ingest time.
            Disable if the pipeline already ran extinction correction and
            Gaia crossmatching.
        run_vetting: Run the STV vetting pipeline (SIMBAD, Gaia alerts, ZTF,
            TNS, eROSITA). Defaults to False because LTV uses its own
            crossmatch step in pipeline.py.
        run_stats: Run compute_stats on each candidate's raw LC to populate
            stats_variability_* columns (von Neumann, Stetson, etc.).
            Requires a valid lc_path column in ltv_df (added by ltv-core).
        stats_compute_ls: Also compute Lomb-Scargle in compute_stats (slower).
        n_workers: Number of parallel workers for compute_stats enrichment.
        index_path: Path to the ASASSN index parquet for gaia_id lookup.
            Auto-resolved from ASASSN_INDEX_PATH if not provided.
        verbose: Print progress.

    Returns:
        (total_rows, new_rows): Total rows processed and how many were new.
    """


    db_path = Path(db_path)
    if verbose:
        print(f"[ltv-ingest] Ingesting {len(ltv_df):,} LTV candidates → {db_path}")

    df = map_ltv_columns(ltv_df)

    # Pre-populate gaia_id from the ASASSN index so characterize_candidates_df
    # has real Gaia IDs for all sources (not just the ~99K in the VSX crossmatch).
    if run_characterize:
        _idx_path = _resolve_ltv_index_path(Path(index_path) if index_path else None)
        if _idx_path:
            df = _add_gaia_ids_from_index_ltv(df, _idx_path, verbose=verbose)
        elif verbose:
            print("[ltv-ingest] Warning: ASASSN index not found; Gaia characterization will have limited coverage")

    # The DB upsert requires candidate_id values to be unique within the
    # ingested frame. Some pipeline outputs may contain duplicates (e.g. if
    # a source appears multiple times in a bin). Keep the first occurrence
    # deterministically (input order) and warn.
    if "candidate_id" in df.columns:
        dup_mask = df["candidate_id"].duplicated(keep="first")
        n_dups = int(dup_mask.sum())
        if n_dups > 0:
            if verbose:
                n_unique = int(df["candidate_id"].nunique())
                print(
                    f"[ltv-ingest] Warning: dropping {n_dups} duplicate candidate_id row(s) "
                    f"({n_unique:,} unique / {len(df):,} total) before DB upsert."
                )
            df = df.loc[~dup_mask].copy()

    if verbose:
        print(f"[ltv-ingest] Mapped columns. candidate_id sample: {df['candidate_id'].iloc[0]!r}")

    if run_stats:
        df = enrich_with_stats(df, compute_ls=stats_compute_ls, n_workers=n_workers, verbose=verbose)

    conn = db_connect(db_path)
    total, new = import_candidates(
        conn,
        df,
        source_path=str(db_path),
        characterize_before_import=run_characterize,
        vet_before_import=run_vetting,
    )

    if verbose:
        print(f"[ltv-ingest] Done. {new} new / {total} total rows.")

    return total, new


# ---------------------------------------------------------------------------
# CLI (malca ltv-ingest)
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="malca ltv-ingest",
        description="Ingest LTV pipeline results into a review DB.",
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        type=str,
        help="Path to LTV build output file (CSV or Parquet) or directory containing such files",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Glob pattern (relative to --input directory) when --input is a directory "
             "(default: '*_pipeline.parquet')",
    )
    p.add_argument(
        "--review-db",
        default="ltv_candidates.db",
        type=str,
        help="Path to LTV review SQLite DB (default: ltv_candidates.db)",
    )
    p.add_argument(
        "--skip-characterize",
        action="store_true",
        help="Skip Gaia/dust characterization at ingest (use if pipeline already ran it)",
    )
    p.add_argument(
        "--run-vetting",
        action="store_true",
        help="Run STV vetting pipeline at ingest (SIMBAD, ZTF, TNS, ...)",
    )
    p.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip compute_stats enrichment (von Neumann, Stetson, etc.)",
    )
    p.add_argument(
        "--stats-compute-ls",
        action="store_true",
        help="Also compute Lomb-Scargle in compute_stats (slower)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for compute_stats enrichment (default: 1)",
    )
    p.add_argument(
        "--index-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to ASASSN index parquet for gaia_id lookup (auto-resolved if not given)",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Single-file mode: preserve existing behaviour
    if input_path.is_file():
        if input_path.suffix == ".parquet":
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)

        print(f"Loaded {len(df):,} rows from {input_path}")

        ingest_ltv_results(
            args.review_db,
            df,
            run_characterize=not args.skip_characterize,
            run_vetting=args.run_vetting,
            run_stats=not args.skip_stats,
            stats_compute_ls=args.stats_compute_ls,
            n_workers=args.workers,
            index_path=args.index_file,
            verbose=args.verbose,
        )
        return

    # Directory mode: ingest multiple files deterministically
    if input_path.is_dir():
        pattern = args.pattern or "*_pipeline.parquet"
        files = sorted(input_path.glob(pattern))

        if not files:
            raise SystemExit(
                f"No files matching '{pattern}' found in directory {input_path}"
            )

        total_files = 0
        last_total_rows = 0
        sum_new_rows = 0

        for fp in files:
            if args.verbose:
                print(f"[ltv-ingest] Loading {fp} ...")

            if fp.suffix == ".parquet":
                df = pd.read_parquet(fp)
            else:
                df = pd.read_csv(fp)

            if args.verbose:
                print(f"[ltv-ingest] Loaded {len(df):,} rows from {fp}")

            total_rows, new_rows = ingest_ltv_results(
                args.review_db,
                df,
                run_characterize=not args.skip_characterize,
                run_vetting=args.run_vetting,
                run_stats=not args.skip_stats,
                stats_compute_ls=args.stats_compute_ls,
                n_workers=args.workers,
                index_path=args.index_file,
                verbose=args.verbose,
            )

            total_files += 1
            last_total_rows = total_rows
            sum_new_rows += new_rows

        if args.verbose:
            print(
                f"[ltv-ingest] Ingested {total_files} file(s) from {input_path}. "
                f"{sum_new_rows} new / {last_total_rows} total rows in DB."
            )
        return

    raise SystemExit(f"Input path is neither a file nor a directory: {input_path}")


if __name__ == "__main__":
    main()
