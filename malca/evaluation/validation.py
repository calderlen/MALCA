"""
Validate detection results against known candidates.

This module compares detection results from events.py against a list of known
candidates to compute validation metrics (precision, recall, etc.) WITHOUT
requiring access to the original light curve data.

Usage:
    malca validate --results events_output.csv --candidates known_targets.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np


# Default validation candidates (Brayden's list)
DEFAULT_CANDIDATES = [
    {"source": "J042214+152530", "source_id": "377957522430", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J202402+383938", "source_id": "42950993887", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J174328+343315", "source_id": "223339338105", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J080327-261620", "source_id": "601296043597", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": False},
    {"source": "J184916-473251", "source_id": "472447294641", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Known", "expected_detected": True},
    {"source": "J183153-284827", "source_id": "455267102087", "category": "Dippers", "mag_bin": "13.5_14", "search_method": "Known", "expected_detected": False},
    {"source": "J070519+061219", "source_id": "266288137752", "category": "Dippers", "mag_bin": "13.5_14", "search_method": "Known", "expected_detected": False},
    {"source": "J081523-385923", "source_id": "532576686103", "category": "Dippers", "mag_bin": "13.5_14", "search_method": "Known", "expected_detected": False},
    {"source": "J085816-430955", "source_id": "352187470767", "category": "Dippers", "mag_bin": "12_12.5", "search_method": "Known", "expected_detected": False},
    {"source": "J114712-621037", "source_id": "609886184506", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Known", "expected_detected": False},
    {"source": "J005437+644347", "source_id": "68720274411", "category": "Multiple Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Known", "expected_detected": True},
    {"source": "J062510-075341", "source_id": "377958261591", "category": "Multiple Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J124745-622756", "source_id": "515397118400", "category": "Multiple Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J175912-120956", "source_id": "326417831663", "category": "Multiple Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J181752-580749", "source_id": "644245387906", "category": "Multiple Eclipse Binaries", "mag_bin": "12_12.5", "search_method": "Known", "expected_detected": True},
    {"source": "J160757-574540", "source_id": "661425129485", "category": "Multiple Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": False},
    {"source": "J073924-272916", "source_id": "438086977939", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J074007-161608", "source_id": "360777377116", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J094848-545959", "source_id": "635655234580", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J162209-444247", "source_id": "412317159120", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J183606-314826", "source_id": "438086901547", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J205245-713514", "source_id": "463856535113", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J212132+480140", "source_id": "120259184943", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J225702+562312", "source_id": "25770019815", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J190316-195739", "source_id": "515396514761", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J175602+013135", "source_id": "231929175915", "category": "Single Eclipse Binaries", "mag_bin": "14_14.5", "search_method": "Known", "expected_detected": True},
    {"source": "J073234-200049", "source_id": "335007754417", "category": "Single Eclipse Binaries", "mag_bin": "14.5_15", "search_method": "Known", "expected_detected": True},
    {"source": "J223332+565552", "source_id": "60130040391", "category": "Single Eclipse Binaries", "mag_bin": "12.5_13", "search_method": "Known", "expected_detected": True},
    {"source": "J183210-173432", "source_id": "317827964025", "category": "Single Eclipse Binaries", "mag_bin": "12.5_13", "search_method": "Pipeline", "expected_detected": False},
]


def validate_detections(
    results_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    *,
    id_column: str = "source_id",
    match_tolerance_arcsec: float = 3.0,
    event_type: Literal["dip", "jump", "either"] = "dip",
    significance_column: str | None = None,
) -> dict:
    """
    Validate detection results against known candidates.
    
    Args:
        results_df: Detection results from events.py
        candidates_df: Known candidates to validate against
        id_column: Column name for source ID
        match_tolerance_arcsec: Matching tolerance for coordinate-based matching
        event_type: Which event type to validate ("dip", "jump", or "either")
        significance_column: Column indicating significance (e.g., "dip_significant")
    
    Returns:
        Dictionary with validation metrics
    """
    # Determine significance column if not provided
    if significance_column is None:
        if event_type == "dip":
            significance_column = "dip_significant" if "dip_significant" in results_df.columns else None
        elif event_type == "jump":
            significance_column = "jump_significant" if "jump_significant" in results_df.columns else None
        else:  # either
            significance_column = None
    
    # Extract IDs from both datasets
    if id_column in results_df.columns:
        detected_ids = set(results_df[id_column].astype(str))
    else:
        # Try to extract from path column
        if "path" in results_df.columns:
            detected_ids = set(
                results_df["path"].apply(lambda x: Path(x).stem.replace(".dat2", "").replace(".csv", ""))
            )
        else:
            raise ValueError(f"Cannot find ID column '{id_column}' or 'path' in results")
    
    if id_column in candidates_df.columns:
        expected_ids = set(candidates_df[id_column].astype(str))
    else:
        raise ValueError(f"Cannot find ID column '{id_column}' in candidates")
    
    if significance_column is None or significance_column not in results_df.columns:
        raise ValueError(
            f"Missing required significance column '{significance_column}' in results."
        )

    significant_mask = results_df[significance_column].astype(bool)
    if event_type == "either" and "jump_significant" in results_df.columns:
        significant_mask |= results_df["jump_significant"].astype(bool)
    detected_ids = set(
        results_df[significant_mask][id_column].astype(str)
        if id_column in results_df.columns
        else results_df[significant_mask]["path"].apply(
            lambda x: Path(x).stem.replace(".dat2", "").replace(".csv", "")
        )
    )
    
    # Compute validation metrics
    true_positives = detected_ids & expected_ids
    false_positives = detected_ids - expected_ids
    false_negatives = expected_ids - detected_ids
    
    n_tp = len(true_positives)
    n_fp = len(false_positives)
    n_fn = len(false_negatives)
    n_expected = len(expected_ids)
    n_detected = len(detected_ids)
    
    # Compute metrics
    precision = n_tp / n_detected if n_detected > 0 else 0.0
    recall = n_tp / n_expected if n_expected > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "n_expected": n_expected,
        "n_detected": n_detected,
        "n_true_positives": n_tp,
        "n_false_positives": n_fp,
        "n_false_negatives": n_fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": sorted(true_positives),
        "false_positives": sorted(false_positives),
        "false_negatives": sorted(false_negatives),
    }


def print_validation_report(metrics: dict, verbose: bool = False) -> None:
    """Print a formatted validation report."""
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    print(f"\nExpected candidates:  {metrics['n_expected']}")
    print(f"Detected candidates:  {metrics['n_detected']}")
    print(f"\nTrue Positives:       {metrics['n_true_positives']}")
    print(f"False Positives:      {metrics['n_false_positives']}")
    print(f"False Negatives:      {metrics['n_false_negatives']}")
    print(f"\nPrecision:            {metrics['precision']:.2%}")
    print(f"Recall:               {metrics['recall']:.2%}")
    print(f"F1 Score:             {metrics['f1_score']:.2%}")
    
    if verbose:
        if metrics['false_negatives']:
            print(f"\nMissed candidates ({len(metrics['false_negatives'])}):")
            for fn_id in metrics['false_negatives'][:20]:
                print(f"  - {fn_id}")
            if len(metrics['false_negatives']) > 20:
                print(f"  ... and {len(metrics['false_negatives']) - 20} more")
        
        if metrics['false_positives']:
            print(f"\nFalse positives ({len(metrics['false_positives'])}):")
            for fp_id in metrics['false_positives'][:20]:
                print(f"  - {fp_id}")
            if len(metrics['false_positives']) > 20:
                print(f"  ... and {len(metrics['false_positives']) - 20} more")
    
    print("=" * 60 + "\n")


def discover_results_files(
    base_dir: Path,
    method: str,
    mag_bin: str | None = None,
) -> list[Path]:
    """
    Discover events results files in the appropriate subdirectory.
    
    Args:
        base_dir: Base output directory (e.g., output/)
        method: Detection method ("loo" or "bf")
        mag_bin: Optional magnitude bin filter (e.g., "13_13.5"). If None, all files.
    
    Returns:
        List of paths to results files
    """
    # Map method to subdirectory name
    subdir_map = {
        "loo": "loo_events_results",
        "bf": "logbf_events_results",
    }
    
    subdir = base_dir / subdir_map[method]
    
    if not subdir.exists():
        raise FileNotFoundError(f"Results directory not found: {subdir}")
    
    # Find all CSV/Parquet files
    files = list(subdir.glob("*.csv")) + list(subdir.glob("*.parquet"))
    
    if not files:
        raise FileNotFoundError(f"No results files found in: {subdir}")
    
    # Filter by mag_bin if specified
    if mag_bin is not None:
        # Match files containing the mag_bin pattern (e.g., "13_13.5" in filename)
        files = [f for f in files if mag_bin in f.stem]
        if not files:
            raise FileNotFoundError(f"No results files found for mag_bin={mag_bin} in: {subdir}")
    
    return sorted(files)


def resolve_run_results_dir(run_dir: Path) -> Path:
    """
    Resolve a run directory to its results directory.

    Accepts either the run root (contains "results/") or the results directory itself.
    """
    run_dir = Path(run_dir)
    if run_dir.is_file():
        raise FileNotFoundError(f"Run directory is a file: {run_dir}")
    if run_dir.name == "results":
        return run_dir
    results_dir = run_dir / "results"
    if results_dir.exists():
        return results_dir
    if run_dir.exists():
        # Allow pointing directly at a directory that already contains results files.
        return run_dir
    raise FileNotFoundError(f"Run directory not found: {run_dir}")


def discover_run_results_files(run_dir: Path, mag_bin: str | None = None) -> list[Path]:
    """Discover results files within a run directory."""
    results_dir = resolve_run_results_dir(run_dir)
    files = list(results_dir.glob("*.csv")) + list(results_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No results files found in: {results_dir}")
    if mag_bin is not None:
        files = [f for f in files if mag_bin in f.stem]
        if not files:
            raise FileNotFoundError(f"No results files found for mag_bin={mag_bin} in: {results_dir}")
    return sorted(files)


def load_and_aggregate_results(files: list[Path]) -> pd.DataFrame:
    """Load multiple results files and aggregate into single DataFrame."""
    dfs = []
    for f in files:
        if f.suffix == ".parquet":
            df = pd.read_parquet(f)
        else:
            df = pd.read_csv(f)
        df["_source_file"] = f.name
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Validate detection results against known candidates"
    )
    
    # Method-based discovery (new)
    parser.add_argument(
        "--method",
        type=str,
        choices=["loo", "bf"],
        default=None,
        help="Detection method: 'loo' (leave-one-out) or 'bf' (Bayes factor). "
             "Auto-discovers files in output/{loo,logbf}_events_results/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Base output directory containing results subdirectories (default: output)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory (e.g., output/runs/20250119_1349) or results dir to scan",
    )
    parser.add_argument(
        "--latest-run",
        action="store_true",
        help="Use most recent run under output/runs/ (default if no --method/--results)",
    )
    parser.add_argument(
        "--mag-bin",
        type=str,
        default=None,
        help="Magnitude bin to filter (e.g., '13_13.5'). If not set, uses ALL files.",
    )
    parser.add_argument(
        "--all-mag-bins",
        action="store_true",
        help="Explicitly search all magnitude bins (same as not setting --mag-bin)",
    )
    
    # Direct file specification (original behavior)
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Direct path to a single results file (overrides --method auto-discovery)",
    )
    
    # Candidates
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Path to known candidates CSV (optional, uses default Brayden list if not provided)",
    )
    
    # Validation options
    parser.add_argument(
        "--id-column",
        type=str,
        default="source_id",
        help="Column name for source ID (default: source_id)",
    )
    parser.add_argument(
        "--event-type",
        type=str,
        choices=["dip", "jump", "either"],
        default="dip",
        help="Event type to validate (default: dip)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output CSV for detailed validation results",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed validation information",
    )
    
    args = parser.parse_args()
    
    # Determine mag_bin filter
    mag_bin = None if args.all_mag_bins else args.mag_bin
    
    # Load results
    if args.results:
        # Direct file specification
        print(f"Loading results from: {args.results}")
        if args.results.endswith(".parquet"):
            results_df = pd.read_parquet(args.results)
        else:
            results_df = pd.read_csv(args.results)
    elif args.run_dir or args.latest_run or (args.method is None and args.results is None):
        base_dir = Path(args.output_dir)
        run_dir = Path(args.run_dir).expanduser() if args.run_dir else None
        if run_dir is None:
            runs_root = base_dir / "runs"
            if not runs_root.exists():
                raise FileNotFoundError(f"No runs directory found: {runs_root}")
            run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()])
            if not run_dirs:
                raise FileNotFoundError(f"No run directories found in: {runs_root}")
            run_dir = run_dirs[-1]
            print(f"Using latest run dir: {run_dir}")
        else:
            print(f"Using run dir: {run_dir}")
        if mag_bin:
            print(f"  Filtering to mag_bin={mag_bin}")
        else:
            print("  Using ALL magnitude bins")
        files = discover_run_results_files(run_dir, mag_bin)
        print(f"  Found {len(files)} results files:")
        for f in files[:10]:
            print(f"    - {f.name}")
        if len(files) > 10:
            print(f"    ... and {len(files) - 10} more")
        results_df = load_and_aggregate_results(files)
        print(f"  Loaded {len(results_df):,} total detection records")
    elif args.method:
        # Method-based discovery
        base_dir = Path(args.output_dir)
        print(f"Discovering results for method={args.method} in {base_dir}/")
        if mag_bin:
            print(f"  Filtering to mag_bin={mag_bin}")
        else:
            print(f"  Using ALL magnitude bins")
        
        files = discover_results_files(base_dir, args.method, mag_bin)
        print(f"  Found {len(files)} results files:")
        for f in files[:10]:
            print(f"    - {f.name}")
        if len(files) > 10:
            print(f"    ... and {len(files) - 10} more")
        
        results_df = load_and_aggregate_results(files)
        print(f"  Loaded {len(results_df):,} total detection records")
    else:
        parser.error("Either --method, --results, or --run-dir must be specified")
    
    # Load or use default candidates
    if args.candidates:
        print(f"Loading candidates from: {args.candidates}")
        candidates_df = pd.read_csv(args.candidates)
    else:
        print("Using default Brayden candidate list")
        candidates_df = pd.DataFrame(DEFAULT_CANDIDATES)
        # Filter to only expected detections
        candidates_df = candidates_df[candidates_df["expected_detected"] == True].copy()
        
        # Filter by mag_bin if specified
        if mag_bin and "mag_bin" in candidates_df.columns:
            candidates_df = candidates_df[candidates_df["mag_bin"] == mag_bin].copy()
            print(f"  Filtered to {len(candidates_df)} candidates in mag_bin={mag_bin}")
    
    # Validate
    metrics = validate_detections(
        results_df,
        candidates_df,
        id_column=args.id_column,
        event_type=args.event_type,
    )
    
    # Print report
    print_validation_report(metrics, verbose=args.verbose)
    
    # Save detailed results if requested
    if args.output:
        output_df = pd.DataFrame({
            "metric": ["n_expected", "n_detected", "n_true_positives", 
                      "n_false_positives", "n_false_negatives",
                      "precision", "recall", "f1_score"],
            "value": [
                metrics["n_expected"],
                metrics["n_detected"],
                metrics["n_true_positives"],
                metrics["n_false_positives"],
                metrics["n_false_negatives"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
            ],
        })
        output_df.to_csv(args.output, index=False)
        print(f"Saved validation metrics to: {args.output}")


if __name__ == "__main__":
    main()
