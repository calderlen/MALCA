#!/usr/bin/env python3
"""
Analyze Phase 1 diagnostic results to identify bottlenecks.

This script compares:
1. Injection results (completeness): efficiency cubes showing detection rates
2. Detection rate results (contamination): false positive rates on clean samples

Together, these show the completeness vs contamination trade-off for each configuration.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json
import re

# Import from malca
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from malca.analysis.injection import load_efficiency_cube, plot_detection_efficiency


def find_latest_run_dir(base_dir: Path, tag: str) -> Path | None:
    """
    Find the latest run directory for a given tag.

    Searches for directories matching patterns:
    - {tag}/  (exact match)
    - {timestamp}_{tag}/  (timestamped, e.g., 20260121_041628_1a_baseline)
    - {tag}/latest/  (symlink)

    Returns the latest by timestamp, or None if not found.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return None

    # Check for exact match first
    exact = base_dir / tag
    if exact.exists():
        # Check for latest symlink inside
        if (exact / "latest").exists():
            return exact / "latest"
        return exact

    # Search for timestamped directories ending with tag
    pattern = re.compile(r"^(\d{8}_\d{6})_" + re.escape(tag) + r"$")
    matches = []
    for d in base_dir.iterdir():
        if d.is_dir():
            m = pattern.match(d.name)
            if m:
                matches.append((m.group(1), d))

    if not matches:
        return None

    # Sort by timestamp descending, return latest
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[0][1]


def compare_efficiency_cubes(
    cubes: Dict[str, dict],
    labels: List[str],
    output_dir: Path,
    phase_name: str,
):
    """
    Compare multiple efficiency cubes and generate diagnostic plots.
    
    Parameters
    ----------
    cubes : dict
        Dictionary mapping run_tag to loaded efficiency cube
    labels : list
        Human-readable labels for each run
    output_dir : Path
        Where to save comparison plots
    phase_name : str
        Name of this diagnostic phase (e.g., "Shallow Dips")
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get baseline (first cube)
    baseline_tag = list(cubes.keys())[0]
    baseline_cube = cubes[baseline_tag]
    
    print(f"\n{'='*70}")
    print(f"Phase: {phase_name}")
    print(f"{'='*70}\n")
    
    # For each test cube, compute delta from baseline
    for i, (run_tag, cube) in enumerate(cubes.items()):
        if run_tag == baseline_tag:
            continue
            
        # Compute delta efficiency (averaged over magnitude)
        delta_eff = np.nanmean(
            cube["efficiency"] - baseline_cube["efficiency"],
            axis=2
        )
        
        # Compute overall improvement metrics
        total_improvement = np.nanmean(delta_eff)
        max_improvement = np.nanmax(delta_eff)
        percent_improved = np.sum(delta_eff > 0.05) / delta_eff.size * 100
        
        print(f"Run: {labels[i]}")
        print(f"  Average efficiency change: {total_improvement:+.3f}")
        print(f"  Max local improvement: {max_improvement:+.3f}")
        print(f"  Percent of grid improved >5%: {percent_improved:.1f}%")
        
        # Find where the biggest improvements are
        if max_improvement > 0.1:
            max_idx = np.unravel_index(np.nanargmax(delta_eff), delta_eff.shape)
            depth_idx, dur_idx = max_idx
            best_depth = baseline_cube["depth_centers"][depth_idx]
            best_dur = baseline_cube["duration_centers"][dur_idx]
            print(f"  Best improvement at: depth={best_depth:.2f}, duration={best_dur:.1f} days")
        
        print()
        
        # Plot delta efficiency
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.pcolormesh(
            baseline_cube["duration_centers"],
            baseline_cube["depth_centers"],
            delta_eff,
            cmap="RdBu_r",
            vmin=-0.5,
            vmax=0.5,
            shading="nearest",
        )
        
        ax.set_xscale("log")
        ax.set_xlabel("Duration (days)", fontsize=14)
        ax.set_ylabel("Fractional Depth", fontsize=14)
        ax.set_title(
            f"Efficiency Change: {labels[i]}\n(vs Baseline)",
            fontsize=16
        )
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Δ Efficiency", fontsize=12)
        
        # Add zero contour
        ax.contour(
            baseline_cube["duration_centers"],
            baseline_cube["depth_centers"],
            delta_eff,
            levels=[0],
            colors="black",
            linewidths=2,
            linestyles="--",
        )
        
        plt.tight_layout()
        plt.savefig(output_dir / f"delta_{run_tag}.png", dpi=150)
        plt.close()
        
        # Also plot side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Baseline
        im0 = axes[0].pcolormesh(
            baseline_cube["duration_centers"],
            baseline_cube["depth_centers"],
            np.nanmean(baseline_cube["efficiency"], axis=2),
            cmap="viridis",
            vmin=0,
            vmax=1,
            shading="nearest",
        )
        axes[0].set_xscale("log")
        axes[0].set_xlabel("Duration (days)", fontsize=12)
        axes[0].set_ylabel("Fractional Depth", fontsize=12)
        axes[0].set_title(f"{labels[0]} (Baseline)", fontsize=14)
        plt.colorbar(im0, ax=axes[0], label="Efficiency")
        
        # Modified
        im1 = axes[1].pcolormesh(
            cube["duration_centers"],
            cube["depth_centers"],
            np.nanmean(cube["efficiency"], axis=2),
            cmap="viridis",
            vmin=0,
            vmax=1,
            shading="nearest",
        )
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Duration (days)", fontsize=12)
        axes[1].set_ylabel("Fractional Depth", fontsize=12)
        axes[1].set_title(f"{labels[i]}", fontsize=14)
        plt.colorbar(im1, ax=axes[1], label="Efficiency")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"comparison_{run_tag}.png", dpi=150)
        plt.close()
    
    print(f"Plots saved to {output_dir}/")


def main():
    """Run diagnostic analysis on Phase 1 results."""
    
    base_dir = Path("output/injection")
    analysis_dir = Path("diagnostics/results")
    
    # Phase 1A: Shallow Dips
    print("\n" + "="*70)
    print("ANALYZING PHASE 1A: SHALLOW DIP BOTTLENECK")
    print("="*70)
    
    phase1a_runs = {
        "1a_baseline": "Baseline",
        "1a_no_mag_offset": "No Min-Mag-Offset",
        "1a_low_logbf": "Lower LogBF Threshold (3.0)",
    }
    
    cubes_1a = {}
    for run_tag in phase1a_runs.keys():
        run_dir = find_latest_run_dir(base_dir, run_tag)
        if run_dir is None:
            print(f"WARNING: No run directory found for {run_tag}")
            continue

        # Check for cubes in various locations
        cube_file = None
        for candidate in [
            run_dir / "cubes" / "efficiency_cube.npz",
            run_dir / "latest" / "cubes" / "efficiency_cube.npz",
            run_dir / "efficiency_cube.npz",
        ]:
            if candidate.exists():
                cube_file = candidate
                break

        if cube_file:
            cubes_1a[run_tag] = load_efficiency_cube(str(cube_file))
            print(f"Loaded: {run_tag} from {cube_file}")
        else:
            print(f"WARNING: Missing efficiency_cube.npz in {run_dir}")
    
    if len(cubes_1a) > 1:
        compare_efficiency_cubes(
            cubes_1a,
            list(phase1a_runs.values()),
            analysis_dir / "phase1a_shallow_dips",
            "Shallow Dip Bottleneck",
        )
    
    # Phase 1B: Short Duration
    print("\n" + "="*70)
    print("ANALYZING PHASE 1B: SHORT DURATION BOTTLENECK")
    print("="*70)
    
    phase1b_runs = {
        "1b_baseline": "Baseline",
        "1b_short_gp": "Shorter GP Timescale (~100d)",
        "1b_trend_baseline": "Trend Baseline (no GP)",
        "1b_masked_gp": "Masked GP",
    }
    
    cubes_1b = {}
    for run_tag in phase1b_runs.keys():
        run_dir = find_latest_run_dir(base_dir, run_tag)
        if run_dir is None:
            print(f"WARNING: No run directory found for {run_tag}")
            continue

        cube_file = None
        for candidate in [
            run_dir / "cubes" / "efficiency_cube.npz",
            run_dir / "latest" / "cubes" / "efficiency_cube.npz",
            run_dir / "efficiency_cube.npz",
        ]:
            if candidate.exists():
                cube_file = candidate
                break

        if cube_file:
            cubes_1b[run_tag] = load_efficiency_cube(str(cube_file))
            print(f"Loaded: {run_tag} from {cube_file}")
        else:
            print(f"WARNING: Missing efficiency_cube.npz in {run_dir}")
    
    if len(cubes_1b) > 1:
        compare_efficiency_cubes(
            cubes_1b,
            list(phase1b_runs.values()),
            analysis_dir / "phase1b_short_duration",
            "Short Duration Bottleneck",
        )
    
    # Phase 1C: Deep-Short Combination
    print("\n" + "="*70)
    print("ANALYZING PHASE 1C: DEEP-SHORT COMBINATION BOTTLENECK")
    print("="*70)
    
    phase1c_runs = {
        "1c_baseline": "Baseline",
        "1c_extended_mag_grid": "Extended Mag Grid (18.0)",
        "1c_deep_mag_grid": "Deep Mag Grid (20.0)",
    }
    
    cubes_1c = {}
    for run_tag in phase1c_runs.keys():
        run_dir = find_latest_run_dir(base_dir, run_tag)
        if run_dir is None:
            print(f"WARNING: No run directory found for {run_tag}")
            continue

        cube_file = None
        for candidate in [
            run_dir / "cubes" / "efficiency_cube.npz",
            run_dir / "latest" / "cubes" / "efficiency_cube.npz",
            run_dir / "efficiency_cube.npz",
        ]:
            if candidate.exists():
                cube_file = candidate
                break

        if cube_file:
            cubes_1c[run_tag] = load_efficiency_cube(str(cube_file))
            print(f"Loaded: {run_tag} from {cube_file}")
        else:
            print(f"WARNING: Missing efficiency_cube.npz in {run_dir}")
    
    if len(cubes_1c) > 1:
        compare_efficiency_cubes(
            cubes_1c,
            list(phase1c_runs.values()),
            analysis_dir / "phase1c_deep_short",
            "Deep-Short Combination Bottleneck",
        )
    
    # Compare detection rates (contamination)
    print("\n" + "="*70)
    print("DETECTION RATE COMPARISON (FALSE POSITIVES)")
    print("="*70)
    
    detection_rate_dir = Path("output/detection_rate")
    
    # Combine all phases
    all_runs = {**phase1a_runs, **phase1b_runs, **phase1c_runs}
    
    detection_stats = {}
    for run_tag, label in all_runs.items():
        run_dir = find_latest_run_dir(detection_rate_dir, run_tag)
        if run_dir is None:
            print(f"\nWARNING: No detection_rate directory found for {run_tag}")
            continue

        summary_file = None
        for candidate in [
            run_dir / "detection_summary.json",
            run_dir / "latest" / "detection_summary.json",
            run_dir / "results" / "detection_summary.json",
        ]:
            if candidate.exists():
                summary_file = candidate
                break

        if summary_file:
            with open(summary_file, "r") as f:
                summary = json.load(f)
                det_rate = summary.get("detection_rate_percent") or summary.get("detection_rate", 0.0) or 0.0
                total_det = summary.get("total_detected", 0) or 0
                sample_sz = summary.get("sample_size", 0) or 0

                detection_stats[run_tag] = {
                    "label": label,
                    "detection_rate": det_rate,
                    "total_detected": total_det,
                    "sample_size": sample_sz,
                }

                print(f"\n{label}:")
                print(f"  Detection rate: {det_rate:.2f}%")
                print(f"  Detected: {total_det} / {sample_sz}")
        else:
            print(f"\nWARNING: Missing detection_summary.json in {run_dir}")
    
    # Create comparison plot
    if detection_stats:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        labels_list = [v["label"] for v in detection_stats.values()]
        rates = [v["detection_rate"] for v in detection_stats.values()]
        
        bars = ax.bar(range(len(labels_list)), rates, color='coral', alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(labels_list)))
        ax.set_xticklabels(labels_list, rotation=45, ha='right')
        ax.set_ylabel("Detection Rate (%)", fontsize=12)
        ax.set_title("False Positive Rate Comparison\n(Lower is Better)", fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, rates)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.2f}%',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(analysis_dir / "detection_rate_comparison.png", dpi=150)
        plt.close()
        
        print(f"\nDetection rate comparison plot saved to {analysis_dir}/detection_rate_comparison.png")
    
    # Generate summary report
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)
    print(f"\nAll diagnostic plots saved to: {analysis_dir}/")
    print("\nNext steps:")
    print("1. Review delta efficiency plots to identify which changes helped")
    print("2. Create optimized configuration combining successful changes")
    print("3. Run Phase 2: Full grid test with optimized config")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
