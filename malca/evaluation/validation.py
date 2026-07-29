"""
Validate detection results against known candidates.

This module compares detection results from events.py against a list of known
candidates to compute validation metrics (precision, recall, etc.) WITHOUT
requiring access to the original light curve data.

Usage:
    malca validate --results events_output.parquet --candidates known_targets.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np

from malca.config import DEFAULT_OUTPUT_DIR, VSX_MAX_SEP_ARCSEC
from malca.io.table_io import read_feature_table, write_parquet_table
from malca.products.candidates import coerce_strict_bool_series


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
    match_tolerance_arcsec: float = VSX_MAX_SEP_ARCSEC,
    event_type: Literal["dip", "jump", "either"] = "dip",
    significance_column: str | None = None,
    match_mode: Literal["id", "coordinates"] = "id",
    label_column: str = "expected_detected",
    results_ra_column: str = "ra",
    results_dec_column: str = "dec",
    candidates_ra_column: str = "ra",
    candidates_dec_column: str = "dec",
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
    if results_df.empty:
        significant_mask = pd.Series(False, index=results_df.index, dtype="bool")
    elif significance_column is not None:
        if significance_column not in results_df.columns:
            raise ValueError(f"Missing required significance column {significance_column!r} in results")
        significant_mask = coerce_strict_bool_series(
            results_df[significance_column], field_name=significance_column
        )
    else:
        required_significance = {
            "dip": ("dip_significant",),
            "jump": ("jump_significant",),
            "either": ("dip_significant", "jump_significant"),
        }[event_type]
        missing_significance = [col for col in required_significance if col not in results_df.columns]
        if missing_significance:
            raise ValueError("Missing required significance column(s): " + ", ".join(missing_significance))
        masks = [
            coerce_strict_bool_series(results_df[col], field_name=col)
            for col in required_significance
        ]
        significant_mask = masks[0].copy()
        for mask in masks[1:]:
            significant_mask |= mask

    candidates = candidates_df.copy()
    if label_column in candidates.columns:
        labels = coerce_strict_bool_series(candidates[label_column], field_name=label_column)
        label_scope = "explicit_positive_and_negative_labels"
    else:
        labels = pd.Series(True, index=candidates.index, dtype="bool")
        label_scope = "positive_reference_only"

    candidate_keys = _candidate_keys(candidates, id_column)
    if candidate_keys.duplicated().any():
        duplicate_values = sorted(candidate_keys.loc[candidate_keys.duplicated(keep=False)].unique())
        raise ValueError(f"Candidate labels contain duplicate {id_column} values: {duplicate_values[:5]}")
    candidates = candidates.assign(_validation_key=candidate_keys, _validation_positive=labels.to_numpy())

    significant_results = results_df.loc[significant_mask].copy()
    if match_mode == "id":
        result_keys = _strict_id_series(results_df, id_column, dataset="results")
        significant_keys = set(result_keys.loc[significant_mask])
        candidates["_validation_detected"] = candidates["_validation_key"].isin(significant_keys)
        labelled_keys = set(candidates["_validation_key"])
        unlabeled_detections = significant_keys - labelled_keys
    elif match_mode == "coordinates":
        candidates["_validation_detected"] = _coordinate_detection_mask(
            significant_results,
            candidates,
            tolerance_arcsec=match_tolerance_arcsec,
            results_ra_column=results_ra_column,
            results_dec_column=results_dec_column,
            candidates_ra_column=candidates_ra_column,
            candidates_dec_column=candidates_dec_column,
        )
        matched_result_mask = _coordinate_detection_mask(
            candidates,
            significant_results,
            tolerance_arcsec=match_tolerance_arcsec,
            results_ra_column=candidates_ra_column,
            results_dec_column=candidates_dec_column,
            candidates_ra_column=results_ra_column,
            candidates_dec_column=results_dec_column,
        ) if not significant_results.empty else pd.Series(False, index=significant_results.index)
        unlabeled_detections = {
            f"result_row_{idx}" for idx in significant_results.index[~matched_result_mask]
        }
    else:
        raise ValueError(f"Unsupported match_mode: {match_mode!r}")

    positive = candidates["_validation_positive"]
    detected = candidates["_validation_detected"].astype(bool)
    true_positives = set(candidates.loc[positive & detected, "_validation_key"])
    false_negatives = set(candidates.loc[positive & ~detected, "_validation_key"])
    labelled_negative = ~positive if label_column in candidates_df.columns else pd.Series(False, index=candidates.index)
    false_positives = set(candidates.loc[labelled_negative & detected, "_validation_key"])
    true_negatives = set(candidates.loc[labelled_negative & ~detected, "_validation_key"])

    n_tp = len(true_positives)
    n_fp = len(false_positives)
    n_fn = len(false_negatives)
    n_tn = len(true_negatives)
    n_expected = int(positive.sum())
    n_detected = int(detected.sum())

    precision = n_tp / (n_tp + n_fp) if label_column in candidates_df.columns and (n_tp + n_fp) else None
    recall = n_tp / n_expected if n_expected > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision is not None and (precision + recall) > 0
        else None
    )
    recall_low, recall_high = _wilson_interval(n_tp, n_expected)
    precision_low, precision_high = _wilson_interval(n_tp, n_tp + n_fp) if precision is not None else (None, None)
    
    return {
        "match_mode": match_mode,
        "label_scope": label_scope,
        "n_expected": n_expected,
        "n_detected": n_detected,
        "n_true_positives": n_tp,
        "n_false_positives": n_fp,
        "n_false_negatives": n_fn,
        "n_true_negatives": n_tn,
        "n_unlabeled_detections": len(unlabeled_detections),
        "precision": precision,
        "precision_ci95_low": precision_low,
        "precision_ci95_high": precision_high,
        "recall": recall,
        "recall_ci95_low": recall_low,
        "recall_ci95_high": recall_high,
        "f1_score": f1_score,
        "true_positives": sorted(true_positives),
        "false_positives": sorted(false_positives),
        "false_negatives": sorted(false_negatives),
        "true_negatives": sorted(true_negatives),
        "unlabeled_detections": sorted(unlabeled_detections),
    }


def _strict_id_series(frame: pd.DataFrame, id_column: str, *, dataset: str) -> pd.Series:
    if id_column not in frame.columns:
        raise ValueError(f"Cannot use ID matching: {dataset} is missing {id_column!r}")
    ids = frame[id_column].astype("string").str.strip()
    missing = ids.isna() | ids.eq("")
    if bool(missing.any()):
        raise ValueError(f"{dataset} contains {int(missing.sum())} blank/null {id_column} value(s)")
    return ids.astype(str)


def _candidate_keys(frame: pd.DataFrame, id_column: str) -> pd.Series:
    if id_column in frame.columns:
        return _strict_id_series(frame, id_column, dataset="candidates")
    return pd.Series([f"candidate_row_{idx}" for idx in frame.index], index=frame.index, dtype="string")


def _coordinate_detection_mask(
    results: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    tolerance_arcsec: float,
    results_ra_column: str,
    results_dec_column: str,
    candidates_ra_column: str,
    candidates_dec_column: str,
) -> pd.Series:
    required = [results_ra_column, results_dec_column]
    missing_results = [col for col in required if col not in results.columns]
    required_candidates = [candidates_ra_column, candidates_dec_column]
    missing_candidates = [col for col in required_candidates if col not in candidates.columns]
    if missing_results or missing_candidates:
        raise ValueError(
            "Coordinate matching requires explicit RA/Dec columns; "
            f"missing results={missing_results}, candidates={missing_candidates}"
        )
    if not np.isfinite(tolerance_arcsec) or tolerance_arcsec <= 0:
        raise ValueError("match_tolerance_arcsec must be finite and positive")
    result_ra = pd.to_numeric(results[results_ra_column], errors="coerce").to_numpy(float)
    result_dec = pd.to_numeric(results[results_dec_column], errors="coerce").to_numpy(float)
    candidate_ra = pd.to_numeric(candidates[candidates_ra_column], errors="coerce").to_numpy(float)
    candidate_dec = pd.to_numeric(candidates[candidates_dec_column], errors="coerce").to_numpy(float)
    if not (np.isfinite(result_ra).all() and np.isfinite(result_dec).all()):
        raise ValueError("Results contain missing/non-finite matching coordinates")
    if not (np.isfinite(candidate_ra).all() and np.isfinite(candidate_dec).all()):
        raise ValueError("Candidates contain missing/non-finite matching coordinates")
    detected = np.zeros(len(candidates), dtype=bool)
    if len(results):
        ra1 = np.deg2rad(candidate_ra)[:, None]
        dec1 = np.deg2rad(candidate_dec)[:, None]
        ra2 = np.deg2rad(result_ra)[None, :]
        dec2 = np.deg2rad(result_dec)[None, :]
        cosine = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
        separation_arcsec = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))) * 3600.0
        detected = np.nanmin(separation_arcsec, axis=1) <= tolerance_arcsec
    return pd.Series(detected, index=candidates.index, dtype="bool")


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if trials <= 0:
        return None, None
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = z * np.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2)) / denominator
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


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
    precision = metrics.get("precision")
    print(f"\nPrecision:            {precision:.2%}" if precision is not None else "\nPrecision:            not estimable (no labelled negatives)")
    print(f"Recall:               {metrics['recall']:.2%}")
    f1_score = metrics.get("f1_score")
    print(f"F1 Score:             {f1_score:.2%}" if f1_score is not None else "F1 Score:             not estimable")
    if metrics.get("n_unlabeled_detections"):
        print(f"Unlabelled detections: {metrics['n_unlabeled_detections']} (excluded from precision)")
    
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
    
    files = list(subdir.glob("*.parquet"))
    
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
    files = list(results_dir.glob("*.parquet"))
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
        df = read_feature_table(f)
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
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Base output directory containing results subdirectories (default: {DEFAULT_OUTPUT_DIR})",
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
        help="Path to known candidates Parquet (optional, uses default Brayden list if not provided)",
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
        "--match-mode",
        choices=["id", "coordinates"],
        default="id",
        help="Use exact IDs or explicit sky-coordinate matching (default: id)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output Parquet for detailed validation results",
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
        results_df = read_feature_table(args.results)
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
        candidates_df = read_feature_table(args.candidates)
    else:
        print("Using default Brayden candidate list")
        candidates_df = pd.DataFrame(DEFAULT_CANDIDATES)
        
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
        match_mode=args.match_mode,
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
        write_parquet_table(output_df, args.output)
        print(f"Saved validation metrics to: {args.output}")


if __name__ == "__main__":
    main()
