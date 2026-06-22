#!/usr/bin/env python3
"""
Analyze Phase 2 results: Baseline vs Optimized configuration.

Compares:
1. Detection efficiency across full parameter space
2. False positive rates
3. Generates comparison plots and summary statistics
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
import json
import re

from malca.lightcurve_publication import (
    apply_publication_rcparams,
    save_publication_figure,
    FIG_SINGLE_COL_LC_WIDE,
    FIG_TWO_COL_TRIPLE,
)

apply_publication_rcparams(plt)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from malca.config import DEFAULT_OUTPUT_DIR
from malca.evaluation.dip_injection import load_efficiency_cube


def find_latest_run_dir(base_dir: Path, tag: str) -> Path | None:
    """
    Find the latest run directory for a given tag.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return None

    exact = base_dir / tag
    if exact.exists():
        if (exact / "latest").exists():
            return exact / "latest"
        return exact

    pattern = re.compile(r"^(\d{8}_\d{6})_" + re.escape(tag) + r"$")
    matches = []
    for d in base_dir.iterdir():
        if d.is_dir():
            m = pattern.match(d.name)
            if m:
                matches.append((m.group(1), d))

    if not matches:
        return None

    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[0][1]


def load_cube_for_tag(base_dir: Path, tag: str) -> dict | None:
    """Load efficiency cube for a given run tag."""
    run_dir = find_latest_run_dir(base_dir, tag)
    if run_dir is None:
        print(f"WARNING: No run directory found for {tag}")
        return None

    for candidate in [
        run_dir / "cubes" / "efficiency_cube.npz",
        run_dir / "latest" / "cubes" / "efficiency_cube.npz",
        run_dir / "efficiency_cube.npz",
    ]:
        if candidate.exists():
            print(f"Loaded: {tag} from {candidate}")
            return load_efficiency_cube(str(candidate))

    print(f"WARNING: Missing efficiency_cube.npz in {run_dir}")
    return None


def load_detection_rate(base_dir: Path, tag: str) -> dict | None:
    """Load detection rate summary for a given run tag."""
    run_dir = find_latest_run_dir(base_dir, tag)
    if run_dir is None:
        return None

    for candidate in [
        run_dir / "detection_summary.json",
        run_dir / "latest" / "detection_summary.json",
        run_dir / "results" / "detection_summary.json",
    ]:
        if candidate.exists():
            with open(candidate, "r") as f:
                return json.load(f)

    return None


def compute_efficiency_stats(cube: dict) -> dict:
    """Compute summary statistics for an efficiency cube."""
    eff = cube["efficiency"]
    eff_flat = eff[np.isfinite(eff)]

    # Marginalize over magnitude
    eff_2d = np.nanmean(eff, axis=2)

    return {
        "mean": float(np.nanmean(eff_flat)),
        "median": float(np.nanmedian(eff_flat)),
        "std": float(np.nanstd(eff_flat)),
        "min": float(np.nanmin(eff_flat)),
        "max": float(np.nanmax(eff_flat)),
        "pct_above_50": float(np.sum(eff_flat > 0.5) / len(eff_flat) * 100),
        "pct_above_90": float(np.sum(eff_flat > 0.9) / len(eff_flat) * 100),
        "eff_2d": eff_2d,
    }


def main():
    """Run Phase 2 analysis."""

    injection_dir = DEFAULT_OUTPUT_DIR / "injection"
    detection_dir = DEFAULT_OUTPUT_DIR / "detection_rate"
    output_dir = Path("diagnostics/results/phase2")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 2 ANALYSIS: BASELINE vs OPTIMIZED")
    print("=" * 70)

    # Load cubes
    cube_baseline = load_cube_for_tag(injection_dir, "2_baseline")
    cube_optimized = load_cube_for_tag(injection_dir, "2_optimized")

    if cube_baseline is None or cube_optimized is None:
        print("\nERROR: Could not load both cubes. Run phase2_optimized.sh first.")
        return

    # Compute statistics
    stats_baseline = compute_efficiency_stats(cube_baseline)
    stats_optimized = compute_efficiency_stats(cube_optimized)

    print("\n" + "=" * 70)
    print("EFFICIENCY COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Baseline':>15} {'Optimized':>15} {'Change':>15}")
    print("-" * 75)

    for metric in ["mean", "median", "pct_above_50", "pct_above_90"]:
        base_val = stats_baseline[metric]
        opt_val = stats_optimized[metric]
        change = opt_val - base_val
        sign = "+" if change >= 0 else ""

        if "pct" in metric:
            print(f"{metric:<30} {base_val:>14.1f}% {opt_val:>14.1f}% {sign}{change:>13.1f}%")
        else:
            print(f"{metric:<30} {base_val:>15.3f} {opt_val:>15.3f} {sign}{change:>14.3f}")

    # Delta efficiency
    delta_eff = stats_optimized["eff_2d"] - stats_baseline["eff_2d"]

    print(f"\n{'Grid improvement statistics':}")
    print(f"  Average delta efficiency: {np.nanmean(delta_eff):+.3f}")
    print(f"  Max improvement: {np.nanmax(delta_eff):+.3f}")
    print(f"  Max degradation: {np.nanmin(delta_eff):+.3f}")
    print(f"  Cells improved >5%: {np.sum(delta_eff > 0.05) / delta_eff.size * 100:.1f}%")
    print(f"  Cells degraded >5%: {np.sum(delta_eff < -0.05) / delta_eff.size * 100:.1f}%")

    # Detection rates
    print("\n" + "=" * 70)
    print("FALSE POSITIVE RATES")
    print("=" * 70)

    dr_baseline = load_detection_rate(detection_dir, "2_baseline")
    dr_optimized = load_detection_rate(detection_dir, "2_optimized")

    if dr_baseline:
        rate = dr_baseline.get("detection_rate_percent") or dr_baseline.get("detection_rate", 0) or 0
        print(f"\nBaseline:  {rate:.2f}%")
    else:
        print("\nBaseline:  (not found)")

    if dr_optimized:
        rate = dr_optimized.get("detection_rate_percent") or dr_optimized.get("detection_rate", 0) or 0
        print(f"Optimized: {rate:.2f}%")
    else:
        print("Optimized: (not found)")

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    # Plot 1: Side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=FIG_TWO_COL_TRIPLE)

    depth_centers = cube_baseline["depth_centers"]
    dur_centers = cube_baseline["duration_centers"]

    # Baseline
    im0 = axes[0].pcolormesh(
        dur_centers, depth_centers, stats_baseline["eff_2d"],
        cmap="viridis", vmin=0, vmax=1, shading="nearest"
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Duration [days]", fontsize=12)
    axes[0].set_ylabel("Fractional Depth", fontsize=12)
    plt.colorbar(im0, ax=axes[0], label="Efficiency")

    # Optimized
    im1 = axes[1].pcolormesh(
        dur_centers, depth_centers, stats_optimized["eff_2d"],
        cmap="viridis", vmin=0, vmax=1, shading="nearest"
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Duration [days]", fontsize=12)
    axes[1].set_ylabel("Fractional Depth", fontsize=12)
    plt.colorbar(im1, ax=axes[1], label="Efficiency")

    # Delta
    im2 = axes[2].pcolormesh(
        dur_centers, depth_centers, delta_eff,
        cmap="RdBu_r", vmin=-0.5, vmax=0.5, shading="nearest"
    )
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Duration [days]", fontsize=12)
    axes[2].set_ylabel("Fractional Depth", fontsize=12)
    cbar = plt.colorbar(im2, ax=axes[2], label="Δ Efficiency")

    # Add zero contour
    axes[2].contour(
        dur_centers, depth_centers, delta_eff,
        levels=[0], colors="black", linewidths=2, linestyles="--"
    )

    save_publication_figure(fig, output_dir / "phase2_comparison.png", dpi=150)
    print(f"  Saved: {output_dir / 'phase2_comparison.png'}")

    # Plot 2: Efficiency vs depth (marginalized over duration)
    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_LC_WIDE)

    eff_vs_depth_baseline = np.nanmean(stats_baseline["eff_2d"], axis=1)
    eff_vs_depth_optimized = np.nanmean(stats_optimized["eff_2d"], axis=1)

    ax.plot(depth_centers, eff_vs_depth_baseline, 'b-', linewidth=2, label="Baseline")
    ax.plot(depth_centers, eff_vs_depth_optimized, 'r-', linewidth=2, label="Optimized")
    ax.fill_between(depth_centers, eff_vs_depth_baseline, eff_vs_depth_optimized,
                    alpha=0.3, color='green', where=(eff_vs_depth_optimized > eff_vs_depth_baseline))
    ax.fill_between(depth_centers, eff_vs_depth_baseline, eff_vs_depth_optimized,
                    alpha=0.3, color='red', where=(eff_vs_depth_optimized < eff_vs_depth_baseline))

    ax.set_xlabel("Fractional Depth", fontsize=12)
    ax.set_ylabel("Detection Efficiency", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    save_publication_figure(fig, output_dir / "efficiency_vs_depth.png", dpi=150)
    print(f"  Saved: {output_dir / 'efficiency_vs_depth.png'}")

    # Plot 3: Efficiency vs duration (marginalized over depth)
    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_LC_WIDE)

    eff_vs_dur_baseline = np.nanmean(stats_baseline["eff_2d"], axis=0)
    eff_vs_dur_optimized = np.nanmean(stats_optimized["eff_2d"], axis=0)

    ax.semilogx(dur_centers, eff_vs_dur_baseline, 'b-', linewidth=2, label="Baseline")
    ax.semilogx(dur_centers, eff_vs_dur_optimized, 'r-', linewidth=2, label="Optimized")
    ax.fill_between(dur_centers, eff_vs_dur_baseline, eff_vs_dur_optimized,
                    alpha=0.3, color='green', where=(eff_vs_dur_optimized > eff_vs_dur_baseline))
    ax.fill_between(dur_centers, eff_vs_dur_baseline, eff_vs_dur_optimized,
                    alpha=0.3, color='red', where=(eff_vs_dur_optimized < eff_vs_dur_baseline))

    ax.set_xlabel("Duration [days]", fontsize=12)
    ax.set_ylabel("Detection Efficiency", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    save_publication_figure(fig, output_dir / "efficiency_vs_duration.png", dpi=150)
    print(f"  Saved: {output_dir / 'efficiency_vs_duration.png'}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nOptimized config (--min-mag-offset 0.0) vs Baseline:")
    print(f"  Mean efficiency: {stats_baseline['mean']:.2f} -> {stats_optimized['mean']:.2f} ({np.nanmean(delta_eff):+.2f})")
    print(f"  Cells >50% eff:  {stats_baseline['pct_above_50']:.1f}% -> {stats_optimized['pct_above_50']:.1f}%")
    print(f"  Cells >90% eff:  {stats_baseline['pct_above_90']:.1f}% -> {stats_optimized['pct_above_90']:.1f}%")

    if dr_baseline and dr_optimized:
        rate_base = dr_baseline.get("detection_rate_percent") or dr_baseline.get("detection_rate", 0) or 0
        rate_opt = dr_optimized.get("detection_rate_percent") or dr_optimized.get("detection_rate", 0) or 0
        print(f"  False positive:  {rate_base:.2f}% -> {rate_opt:.2f}%")

    print(f"\nAll plots saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
