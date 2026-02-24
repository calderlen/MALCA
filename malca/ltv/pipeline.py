"""
LTV Full Pipeline Integration — Optimized for Scale.

Combines all LTV modules into a complete slowly varying source detection pipeline.

CRITICAL OPTIMIZATION: Pipeline stages run in optimal order:
1. Vectorized filters FIRST (instant, reduce 17M → ~36K)
2. Gaia TAP filters (batch queries on reduced set)
3. Catalog crossmatches (parallel on filtered candidates only)
4. NEOWISE extraction (parallel on filtered candidates only)
5. Extinction correction (vectorized)

This ordering is crucial for performance:
- Running crossmatch before filtering = weeks
- Running crossmatch after filtering = hours
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from malca.config.config_paths import LTV_OUTPUT_DIR
from malca.config.config_ltv import (
    LTV_MIN_SLOPE,
    LTV_MIN_DIFF,
    LTV_MIN_DEC,
    LTV_MAX_PM,
    LTV_MATCH_RADIUS_ARCSEC,
    LTV_WORKERS,
    LTV_CHUNK_SIZE,
    GAIA_EPOCH_DATA_RELEASE,
    GAIA_EPOCH_DATA_STRUCTURE,
)
from malca.config.config_io import PARQUET_OUTPUT_COMPRESSION
from malca.ltv.filter import (
    apply_all_filters,
    filter_slope_threshold,
    filter_max_diff_threshold,
    filter_south_pole,
    filter_high_proper_motion,
)
from malca.ltv.crossmatch import (
    crossmatch_all_catalogs,
    crossmatch_milliquas,
    query_simbad_classification,
)
from malca.ltv.neowise import extract_neowise_trends
from malca.ltv.dust import apply_dust_flags
from malca.ltv.cmd import compute_cmd_features, assign_cmd_groups, load_mist_grid, fetch_bailer_jones_distances
from malca.ltv.gaia_epoch import query_gaia_epoch_photometry_batch, apply_gaia_epoch_flags
from malca.characterize import get_dust_extinction


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
    3. Catalog crossmatches (parallel on filtered candidates)
    4. NEOWISE extraction (parallel on filtered candidates)
    5. Extinction correction (vectorized)
    
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
    if run_filters:
        if verbose:
            print("-" * 60)
            print("STAGE 1: FILTERING")
            print("-" * 60)
        
        df = apply_all_filters(
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
        )
        
        if verbose:
            reduction = (1 - len(df)/n0) * 100
            print(f"\n→ After filtering: {len(df):,} candidates ({reduction:.1f}% reduction)")
            print()
    
    if df.empty:
        if verbose:
            print("No sources remaining after filtering")
        return df
    
    # =========================================================================
    # Stage 2: Catalog crossmatch (only on filtered candidates)
    # =========================================================================
    if run_crossmatch:
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
            match_radius_arcsec=match_radius_arcsec,
            n_workers=n_workers,
            verbose=verbose,
        )
        print()
    
    # =========================================================================
    # Stage 3: NEOWISE extraction (only on filtered candidates)
    # =========================================================================
    if run_neowise:
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
        
        print("=" * 60)
    
    return df


# =============================================================================
# CLI SUBCOMMAND
# =============================================================================

def add_pipeline_args(parser):
    """Add pipeline arguments to argparse."""
    parser.add_argument(
        "--mag-bin",
        type=str,
        default=None,
        choices=["12_12.5", "12.5_13", "13_13.5", "13.5_14", "14_14.5", "14.5_15"],
        help="Magnitude bin (auto-resolves input/output in output/ltv/)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV/Parquet from ltv.core (default: output/ltv/LTvar<MAG>.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV/Parquet for pipeline results (default: output/ltv/LTvar<MAG>_pipeline.csv)",
    )
    parser.add_argument(
        "--min-slope",
        type=float,
        default=LTV_MIN_SLOPE,
        help="Minimum |Slope| threshold (mag/yr)",
    )
    parser.add_argument(
        "--min-diff",
        type=float,
        default=LTV_MIN_DIFF,
        help="Minimum |max diff| threshold (mag)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=LTV_WORKERS,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=LTV_CHUNK_SIZE,
        help="Chunk size for batch queries",
    )
    parser.add_argument(
        "--skip-filters",
        action="store_true",
        help="Skip filtering stage",
    )
    parser.add_argument(
        "--skip-crossmatch",
        action="store_true",
        help="Skip catalog crossmatch stage",
    )
    parser.add_argument(
        "--skip-neowise",
        action="store_true",
        help="Skip NEOWISE extraction stage",
    )
    parser.add_argument(
        "--skip-extinction",
        action="store_true",
        help="Skip extinction correction stage",
    )
    parser.add_argument(
        "--skip-dust-flags",
        action="store_true",
        help="Skip dust-driven variability flags",
    )
    parser.add_argument(
        "--skip-gaia-epoch",
        action="store_true",
        help="Skip Gaia epoch photometry stage",
    )
    parser.add_argument(
        "--gaia-epoch-table",
        type=str,
        default=None,
        help="Gaia TAP epoch photometry table (optional; if omitted, DataLink is used)",
    )
    parser.add_argument(
        "--gaia-epoch-data-release",
        type=str,
        default=GAIA_EPOCH_DATA_RELEASE,
        help="Gaia data release for DataLink (default: Gaia DR3)",
    )
    parser.add_argument(
        "--gaia-epoch-data-structure",
        type=str,
        default=GAIA_EPOCH_DATA_STRUCTURE,
        help="DataLink structure: RAW or INDIVIDUAL (default: RAW)",
    )
    parser.add_argument(
        "--gaia-epoch-band",
        type=str,
        default=None,
        help="Optional band restriction for DataLink (e.g., G, BP, RP)",
    )
    parser.add_argument(
        "--gaia-epoch-include-invalid",
        action="store_true",
        help="Include invalid epoch photometry (valid_data=False)",
    )
    parser.add_argument(
        "--skip-bailer-jones",
        action="store_true",
        help="Skip Bailer-Jones distance fetch stage",
    )
    parser.add_argument(
        "--skip-cmd",
        action="store_true",
        help="Skip CMD features stage",
    )
    parser.add_argument(
        "--log-rejections",
        type=str,
        default=None,
        help="Log rejected sources to this CSV",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print progress",
    )
    return parser


def run_pipeline_cli(args):
    """Run pipeline from CLI arguments."""
    # Resolve input/output from --mag-bin if not given explicitly
    if args.input is None:
        if args.mag_bin is None:
            raise SystemExit("Error: must specify --input or --mag-bin")
        args.input = str(LTV_OUTPUT_DIR / f"LTvar{args.mag_bin.replace('_', '-')}.csv")
    if args.output is None:
        if args.mag_bin is None:
            raise SystemExit("Error: must specify --output or --mag-bin")
        args.output = str(LTV_OUTPUT_DIR / f"LTvar{args.mag_bin.replace('_', '-')}_pipeline.csv")

    # Load input
    input_path = Path(args.input)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    
    print(f"Loaded {len(df):,} sources from {input_path}")
    
    # Run pipeline
    df = run_full_pipeline(
        df,
        min_slope=args.min_slope,
        min_diff=args.min_diff,
        run_filters=not args.skip_filters,
        run_crossmatch=not args.skip_crossmatch,
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
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
        log_csv=args.log_rejections,
        verbose=args.verbose,
    )
    
    # Save output
    output_path = Path(args.output)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)
    else:
        df.to_csv(output_path, index=False)
    
    print(f"Saved {len(df):,} candidates to {output_path}")
    return df
