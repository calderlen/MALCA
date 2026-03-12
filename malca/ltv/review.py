"""
LTV → Review DB ingest bridge.

Maps LTV pipeline output columns to the review DB schema and ingests
candidates into a standalone LTV review DB (separate from STV candidates).

Usage:
    # CLI
    malca ltv-ingest --input ltv_pipeline_output.csv --db ltv_candidates.db -v

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

from malca.review.store import db_connect, import_candidates
from malca.stats import compute_stats







# ---------------------------------------------------------------------------
# Column mapping: LTV pipeline output → review DB schema
# ---------------------------------------------------------------------------

# Core output columns from core.py (legacy names → ltv_* names)
_CORE_COL_MAP = {
    "Slope":       "ltv_slope",
    "Quad Slope":  "ltv_slope_quad",
    "max diff":    "ltv_max_diff",
    "Dispersion":  "ltv_dispersion",
    "Median":      "ltv_median",
    "n_seasons":   "ltv_n_seasons",
    "ls_period":   "ltv_ls_period",
    "ls_power":    "ltv_ls_power",
    "ls_fap":      "ltv_ls_fap",
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
    (from ltv-pipeline, which adds crossmatch, NEOWISE, dust, extinction).

    Returns a new DataFrame with:
    - candidate_id = "ltv_{asas_sn_id}"
    - asas_sn_id from "ASAS-SN ID"
    - ltv_* columns from pipeline output
    - ra_deg, dec_deg preserved for characterization
    """
    df = df.copy()

    # --- Identify the source ID column ---
    id_col = None
    for candidate in ("ASAS-SN ID", "asas_sn_id", "asassn_id"):
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError(
            "LTV DataFrame must have an 'ASAS-SN ID' or 'asas_sn_id' column"
        )

    # Normalise to asas_sn_id (string) + candidate_id
    if id_col != "asas_sn_id":
        df["asas_sn_id"] = df[id_col].astype(str)
        if id_col != "asas_sn_id":
            df = df.drop(columns=[id_col])
    else:
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

    # --- Cast int bool columns ---
    for col in (
        "ltv_passed_filters",
        "ltv_dust_candidate",
        "ltv_dust_excess",
        "ltv_vsx_match",
        "ltv_milliquas_match",
        "ltv_gaia_alert_match",
        "ltv_stoch_mhps_pn_flag",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def enrich_with_stats(
    df: pd.DataFrame,
    *,
    compute_ls: bool = False,
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
        print(f"[ltv-ingest] Running compute_stats on {n_valid:,} candidates...")

    rows = df.to_dict("records")
    enriched = []

    for row in tqdm(rows, desc="compute_stats", disable=not verbose):
        lc_path_str = row.get("lc_path", "")
        if not lc_path_str or pd.isna(lc_path_str):
            enriched.append(row)
            continue

        lc_path = Path(str(lc_path_str))
        if not lc_path.exists():
            enriched.append(row)
            continue

        asassn_id = lc_path.stem.split("-")[0]
        dir_path = str(lc_path.parent)

        try:
            _, stats_dict = compute_stats(
                asassn_id,
                dir_path,
                use_only_good=True,
                compute_ls=compute_ls,
            )
            merged = dict(row)
            for k, v in stats_dict.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        col = f"stats_{k}_{sub_k}"
                        if col not in merged:
                            merged[col] = sub_v
                elif isinstance(v, (pd.DataFrame, pd.Series)):
                    continue
                elif f"stats_{k}" not in merged:
                    merged[f"stats_{k}"] = v
            enriched.append(merged)
        except Exception as e:
            if verbose:
                print(f"  Warning: compute_stats failed for {lc_path}: {e}")
            enriched.append(row)

    return pd.DataFrame(enriched)


def ingest_ltv_results(
    db_path: str | Path,
    ltv_df: pd.DataFrame,
    *,
    run_characterize: bool = True,
    run_vetting: bool = False,
    run_stats: bool = True,
    stats_compute_ls: bool = False,
    verbose: bool = True,
) -> tuple[int, int]:
    """
    Ingest LTV pipeline results into a standalone LTV review DB.

    Accepts either core.py output (raw seasonal trend metrics) or full
    pipeline output (with crossmatch, NEOWISE, dust, extinction columns).

    Args:
        db_path: Path to the LTV review SQLite DB (created if it doesn't exist).
        ltv_df: DataFrame from ltv-core or ltv-pipeline.
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
        verbose: Print progress.

    Returns:
        (total_rows, new_rows): Total rows processed and how many were new.
    """


    db_path = Path(db_path)
    if verbose:
        print(f"[ltv-ingest] Ingesting {len(ltv_df):,} LTV candidates → {db_path}")

    df = map_ltv_columns(ltv_df)

    if verbose:
        print(f"[ltv-ingest] Mapped columns. candidate_id sample: {df['candidate_id'].iloc[0]!r}")

    if run_stats:
        df = enrich_with_stats(df, compute_ls=stats_compute_ls, verbose=verbose)

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
        help="Path to ltv-pipeline output (CSV or Parquet)",
    )
    p.add_argument(
        "--db",
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
        "-v", "--verbose",
        action="store_true",
        help="Print progress",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    print(f"Loaded {len(df):,} rows from {input_path}")

    ingest_ltv_results(
        args.db,
        df,
        run_characterize=not args.skip_characterize,
        run_vetting=args.run_vetting,
        run_stats=not args.skip_stats,
        stats_compute_ls=args.stats_compute_ls,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
