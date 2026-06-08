"""
LTV → Review DB ingest bridge.

Maps LTV pipeline output columns to the review DB schema and ingests
candidates into a standalone LTV review DB (separate from STV candidates).

Usage:
    # Python API used by malca ltv-pipeline
    from malca.ltv.review import ingest_ltv_results
    n_total, n_new = ingest_ltv_results("<ltv-run>/review/review.db", ltv_df)
"""
from __future__ import annotations

from pathlib import Path
import argparse

from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from malca.candidates import passing_candidates_mask
from malca.config import ASASSN_INDEX_PATH
from malca.config import LTV_MAX_PM
from malca.feature_layers import to_layer_first_frame, with_feature_columns
from malca.product_schema import add_ltv_identity, assert_ltv_product_schema
from malca.review.store import db_connect, import_candidates
from malca.table_io import read_feature_table, require_parquet_path
from malca.ltv.paths import (
    DEFAULT_LTV_RUN_DIR,
    ltv_results_dir,
    ltv_review_db_path,
    ltv_run_dir_from_review_db,
)







def map_ltv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize canonical LTV pipeline output for review DB ingest.

    Returns a new DataFrame with:
    - candidate_id = "ltv_{asas_sn_id}" when missing
    - canonical shared product columns preserved
    - derived review convenience flags for canonical crossmatch/context columns
    """
    df = with_feature_columns(
        df,
        [
            "failed_any",
            "vsx_name",
            "ltv_vsx_name",
            "milliquas_name",
            "gaia_alert_name",
            "pm_total",
            "pmra",
            "pmdec",
            "high_pm_flag",
        ],
    )
    df = add_ltv_identity(df)
    df["asas_sn_id"] = df["asas_sn_id"].astype(str)

    # Generic VSX catalog columns are canonical shared enrichment output;
    # mirror the name into the LTV-specific review column used by the UI.
    if "vsx_name" in df.columns and "ltv_vsx_name" not in df.columns:
        df["ltv_vsx_name"] = df["vsx_name"]
    if "ltv_vsx_name" in df.columns and "ltv_vsx_match" not in df.columns:
        df["ltv_vsx_match"] = df["ltv_vsx_name"].notna().astype(int)

    if "milliquas_name" in df.columns and "ltv_milliquas_match" not in df.columns:
        df["ltv_milliquas_match"] = df["milliquas_name"].notna().astype(int)

    if "gaia_alert_name" in df.columns and "ltv_gaia_alert_match" not in df.columns:
        df["ltv_gaia_alert_match"] = df["gaia_alert_name"].notna().astype(int)

    pm_total_missing = "pm_total" not in df.columns or df["pm_total"].isna().all()
    if pm_total_missing and {"pmra", "pmdec"}.issubset(df.columns):
        pmra = pd.to_numeric(df["pmra"], errors="coerce")
        pmdec = pd.to_numeric(df["pmdec"], errors="coerce")
        df["pm_total"] = np.sqrt(pmra * pmra + pmdec * pmdec)
    high_pm_missing = "high_pm_flag" not in df.columns or df["high_pm_flag"].isna().all()
    if "pm_total" in df.columns and high_pm_missing:
        pm_total = pd.to_numeric(df["pm_total"], errors="coerce")
        df["high_pm_flag"] = (pm_total > float(LTV_MAX_PM)).fillna(False).astype(int)

    # --- Cast int bool columns ---
    for col in (
        "failed_any",
        "ltv_failed_slope",
        "ltv_failed_max_diff",
        "ltv_failed_dec",
        "ltv_failed_refcat_offset",
        "ltv_failed_photometric_scatter",
        "ltv_failed_high_pm",
        "ltv_failed_neighbor_high_pm",
        "ltv_failed_crowding",
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

    out = to_layer_first_frame(df)
    assert_ltv_product_schema(out, stage="review_ingest")
    return out


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
    from malca.stats import _enrich_row_worker

    if "lc_path" not in df.columns:
        if verbose:
            print("[ltv-review] No lc_path column; skipping compute_stats enrichment")
        return df

    valid_mask = df["lc_path"].notna() & (df["lc_path"].astype(str) != "")
    n_valid = valid_mask.sum()
    if n_valid == 0:
        if verbose:
            print("[ltv-review] No valid lc_path values; skipping compute_stats")
        return df

    if verbose:
        print(f"[ltv-review] Running compute_stats on {n_valid:,} candidates ({n_workers} workers)...")

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

    return to_layer_first_frame(pd.DataFrame(results))


def _resolve_ltv_index_path(index_override: Path | None = None) -> Path | None:
    """Resolve the ASASSN index parquet path, trying common locations."""
    default = ASASSN_INDEX_PATH.expanduser()
    candidates = []
    if index_override is not None:
        candidates.append(Path(index_override).expanduser())
    candidates.append(default)
    # also search input/ relative to cwd
    for pattern in ("asassn_index*.parquet",):
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
    df = with_feature_columns(df, ["gaia_id"])
    needs_fill = "gaia_id" not in df.columns or df["gaia_id"].isna().all()
    already_filled = "gaia_id" in df.columns and not df["gaia_id"].isna().any()
    if already_filled:
        return df
    try:
        if verbose:
            print(f"[ltv-review] Loading ASASSN index for gaia_id lookup: {index_path.name}")
        df_idx = pd.read_parquet(require_parquet_path(index_path), columns=["asas_sn_id", "gaia_id"])
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
            print(f"[ltv-review] Filled gaia_id for {n_filled}/{len(out)} candidates from ASASSN index")
        return out
    except Exception as e:
        if verbose:
            print(f"[ltv-review] Warning: gaia_id index lookup failed: {e}")
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
    source_path: Path | str | None = None,
    verbose: bool = True,
) -> tuple[int, int]:
    """
    Ingest LTV pipeline results into a standalone LTV review DB.

    Accepts either core.py output (raw seasonal trend metrics) or full
    pipeline output (with crossmatch, NEOWISE, dust, extinction columns).

    Args:
        db_path: Path to the LTV review SQLite DB (created if it doesn't exist).
        ltv_df: DataFrame from LTV core metrics or enriched pipeline output.
        run_characterize: Run Gaia/dust characterization at ingest time.
            Disable if the pipeline already ran extinction correction and
            Gaia crossmatching.
        run_vetting: Run the STV vetting pipeline (SIMBAD, Gaia alerts, ZTF,
            TNS, eROSITA). Defaults to False because LTV uses its own
            crossmatch step in pipeline.py.
        run_stats: Run compute_stats on each candidate's raw LC to populate
            stats_variability_* columns (von Neumann, Stetson, etc.).
            Requires a valid lc_path column in ltv_df (added by LTV core metrics).
        stats_compute_ls: Also compute Lomb-Scargle in compute_stats (slower).
        n_workers: Number of parallel workers for compute_stats enrichment.
        index_path: Path to the ASASSN index parquet for gaia_id lookup.
            Auto-resolved from ASASSN_INDEX_PATH if not provided.
        source_path: Source path stored in the review DB. Defaults to the
            enclosing LTV run directory when db_path is <run>/review/review.db.
        verbose: Print progress.

    Returns:
        (total_rows, new_rows): Total rows processed and how many were new.
    """


    db_path = Path(db_path)
    if source_path is None:
        source_path = ltv_run_dir_from_review_db(db_path) or db_path
    if verbose:
        print(f"[ltv-review] Ingesting {len(ltv_df):,} LTV candidates -> {db_path}")

    df = map_ltv_columns(ltv_df)

    pass_mask = passing_candidates_mask(df)
    n_before = len(df)
    df = df.loc[pass_mask].copy()
    if verbose and len(df) != n_before:
        print(f"[ltv-review] Importing {len(df):,}/{n_before:,} rows with failed_any=False")

    # Pre-populate gaia_id from the ASASSN index so characterize_candidates_df
    # has real Gaia IDs for all sources (not just the ~99K in the VSX crossmatch).
    if run_characterize:
        _idx_path = _resolve_ltv_index_path(Path(index_path) if index_path else None)
        if _idx_path:
            df = _add_gaia_ids_from_index_ltv(df, _idx_path, verbose=verbose)
        elif verbose:
            print("[ltv-review] Warning: ASASSN index not found; Gaia characterization will have limited coverage")

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
                    f"[ltv-review] Warning: dropping {n_dups} duplicate candidate_id row(s) "
                    f"({n_unique:,} unique / {len(df):,} total) before DB upsert."
                )
            df = df.loc[~dup_mask].copy()

    if df.empty:
        conn = db_connect(db_path)
        conn.close()
        if verbose:
            print("[ltv-review] No passing LTV candidates to import.")
        return 0, 0

    if verbose:
        print(f"[ltv-review] Mapped columns. candidate_id sample: {df['candidate_id'].iloc[0]!r}")

    if run_stats:
        df = enrich_with_stats(df, compute_ls=stats_compute_ls, n_workers=n_workers, verbose=verbose)
    else:
        df = to_layer_first_frame(df)

    conn = db_connect(db_path)
    total, new = import_candidates(
        conn,
        df,
        source_path=str(source_path),
        characterize_before_import=run_characterize,
        vet_before_import=run_vetting,
    )

    if verbose:
        print(f"[ltv-review] Done. {new} new / {total} total rows.")

    return total, new


# ---------------------------------------------------------------------------
# Internal CLI retained for module-level debugging; public workflow is malca ltv-pipeline.
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m malca.ltv.review",
        description="Ingest LTV pipeline results into a review DB.",
    )
    p.add_argument(
        "--run-dir",
        default=str(DEFAULT_LTV_RUN_DIR),
        type=str,
        help=f"LTV run directory for default input and review DB (default: {DEFAULT_LTV_RUN_DIR})",
    )
    p.add_argument(
        "--input", "-i",
        default=None,
        type=str,
        help="Path to LTV build output Parquet or directory containing such files (default: <run-dir>/results)",
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
        default=None,
        type=str,
        help="Path to LTV review SQLite DB (default: <run-dir>/review/review.db)",
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

    run_dir = Path(args.run_dir).expanduser()
    if args.input is None:
        args.input = str(ltv_results_dir(run_dir))
    if args.review_db is None:
        args.review_db = str(ltv_review_db_path(run_dir))

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Single-file mode: preserve existing behaviour
    if input_path.is_file():
        df = read_feature_table(input_path)

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
            source_path=run_dir,
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
                print(f"[ltv-review] Loading {fp} ...")

            df = read_feature_table(fp)

            if args.verbose:
                print(f"[ltv-review] Loaded {len(df):,} rows from {fp}")

            total_rows, new_rows = ingest_ltv_results(
                args.review_db,
                df,
                run_characterize=not args.skip_characterize,
                run_vetting=args.run_vetting,
                run_stats=not args.skip_stats,
                stats_compute_ls=args.stats_compute_ls,
                n_workers=args.workers,
                index_path=args.index_file,
                source_path=run_dir,
                verbose=args.verbose,
            )

            total_files += 1
            last_total_rows = total_rows
            sum_new_rows += new_rows

        if args.verbose:
            print(
                f"[ltv-review] Ingested {total_files} file(s) from {input_path}. "
                f"{sum_new_rows} new / {last_total_rows} total rows in DB."
            )
        return

    raise SystemExit(f"Input path is neither a file nor a directory: {input_path}")


if __name__ == "__main__":
    main()
