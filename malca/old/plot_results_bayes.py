                      
"""
Plot light curves with Bayesian event detection results, showing run fits overlaid.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import ast
import json
import traceback

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from malca.baseline import (
    per_camera_gp_baseline,
    global_mean_baseline,
    global_median_baseline,
    global_rolling_median_baseline,
    global_rolling_mean_baseline,
    per_camera_mean_baseline,
    per_camera_median_baseline,
    per_camera_trend_baseline,
)
from malca.events import run_bayesian_significance
from malca.plot import read_skypatrol_csv, JD_OFFSET
from malca.utils import gaussian, paczynski_kernel






BASELINE_FUNCTIONS = {
    "global_mean": global_mean_baseline,
    "global_median": global_median_baseline,
    "global_rolling_median": global_rolling_median_baseline,
    "global_rolling_mean": global_rolling_mean_baseline,
    "per_camera_mean": per_camera_mean_baseline,
    "per_camera_median": per_camera_median_baseline,
    "per_camera_trend": per_camera_trend_baseline,
    "per_camera_gp": per_camera_gp_baseline,
}

PER_CAMERA_BASELINES = {
    per_camera_mean_baseline,
    per_camera_median_baseline,
    per_camera_trend_baseline,
    per_camera_gp_baseline,
}


def plot_bayes_results(
    csv_path: Path,
    results_csv: Path | None = None,
    *,
    out_path: Path | None = None,
    figsize=(16, 10),
    show=False,
    baseline_func=None,
    baseline_kwargs=None,
    logbf_threshold_dip=5.0,
    logbf_threshold_jump=5.0,
):
    """Plot a light curve with Bayesian detection results and run fits."""
                      
    df = read_skypatrol_csv(csv_path)
    if df.empty:
        print(f"Warning: {csv_path.name} is empty")
        return None
    
                              
    asas_sn_id = csv_path.stem.split("-")[0]
    
                                                                   
    if baseline_func is None:
        baseline_func = per_camera_gp_baseline
    if baseline_kwargs is None:
        baseline_kwargs = {}
    per_camera_baseline = baseline_func in PER_CAMERA_BASELINES
    
    print(f"Analyzing {asas_sn_id}...")
    
                                             
    df_g = df[df["v_g_band"] == 0].copy()
    df_v = df[df["v_g_band"] == 1].copy()
    
    res_g = run_bayesian_significance(
        df_g,
        baseline_func=baseline_func,
        baseline_kwargs=baseline_kwargs,
        logbf_threshold_dip=logbf_threshold_dip,
        logbf_threshold_jump=logbf_threshold_jump,
        compute_event_prob=True,
    ) if not df_g.empty else {"dip": {"significant": False, "run_summaries": []}, "jump": {"significant": False, "run_summaries": []}}
    
    res_v = run_bayesian_significance(
        df_v,
        baseline_func=baseline_func,
        baseline_kwargs=baseline_kwargs,
        logbf_threshold_dip=logbf_threshold_dip,
        logbf_threshold_jump=logbf_threshold_jump,
        compute_event_prob=True,
    ) if not df_v.empty else {"dip": {"significant": False, "run_summaries": []}, "jump": {"significant": False, "run_summaries": []}}
    
                           
    band_results = {0: res_g, 1: res_v}
    
                               
    df = df[np.isfinite(df["JD"]) & np.isfinite(df["mag"])].copy()
    median_jd = df["JD"].median()
    if median_jd > 2000000:
        df["JD_plot"] = df["JD"] - JD_OFFSET
    else:
        df["JD_plot"] = df["JD"] - 8000.0
    
                      
    bands = [0, 1]        
    band_labels = {0: "g band", 1: "V band"}
    band_markers = {0: "o", 1: "s"}
    
                                                                           
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True, sharex="col")
    
                       
    camera_ids = sorted(df["camera#"].dropna().unique())
    cmap = plt.get_cmap("tab20", max(len(camera_ids), 1))
    camera_colors = {cam: cmap(i % cmap.N) for i, cam in enumerate(camera_ids)}
    
                       
    for band_idx, band in enumerate([1, 0]):                             
        band_df = df[df["v_g_band"] == band].copy()
        if band_df.empty:
            continue
        
                                    
        if baseline_func:
            band_df_baseline = baseline_func(band_df, **baseline_kwargs)
            if "baseline" in band_df_baseline.columns:
                band_df["baseline"] = band_df_baseline["baseline"]
                band_df["resid"] = band_df["mag"] - band_df["baseline"]
        
                                         
        ax_main = axes[0, band_idx]
        ax_main.invert_yaxis()
        
                                    
        for cam in camera_ids:
            subset = band_df[band_df["camera#"] == cam]
            if subset.empty:
                continue
            color = camera_colors[cam]
            marker = band_markers.get(band, "o")
            ax_main.errorbar(
                subset["JD_plot"],
                subset["mag"],
                yerr=subset["error"],
                fmt=marker,
                ms=3,
                color=color,
                alpha=0.7,
                ecolor=color,
                elinewidth=0.8,
                capsize=1.5,
                markeredgecolor="black",
                markeredgewidth=0.3,
                label=f"Camera {cam}",
            )
        
                       
        if "baseline" in band_df.columns:
            baseline_finite = band_df[np.isfinite(band_df["baseline"])]
            if not baseline_finite.empty:
                if per_camera_baseline:
                    for cam in camera_ids:
                        cam_baseline = baseline_finite[baseline_finite["camera#"] == cam]
                        if cam_baseline.empty:
                            continue
                        cam_sorted = cam_baseline.sort_values("JD_plot")
                        ax_main.plot(
                            cam_sorted["JD_plot"],
                            cam_sorted["baseline"],
                            color=camera_colors[cam],
                            linestyle="-",
                            linewidth=1.6,
                            alpha=0.8,
                            zorder=5,
                        )
                else:
                    baseline_sorted = baseline_finite.sort_values("JD_plot")
                    ax_main.plot(
                        baseline_sorted["JD_plot"],
                        baseline_sorted["baseline"],
                        color="orange",
                        linestyle="-",
                        linewidth=2,
                        alpha=0.8,
                        label="Baseline",
                        zorder=5,
                    )
        
                                                                  
        band_res = band_results[band]
        dip = band_res["dip"]
        jump = band_res["jump"]
        
                       
        if dip["significant"] and dip["run_summaries"]:
            for run_summary in dip["run_summaries"]:
                jd_start = run_summary["start_jd"]
                jd_end = run_summary["end_jd"]
                
                                    
                jd_plot_start = jd_start - (JD_OFFSET if median_jd > 2000000 else 8000.0)
                jd_plot_end = jd_end - (JD_OFFSET if median_jd > 2000000 else 8000.0)
                ax_main.axvline(jd_plot_start, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
                if jd_plot_end != jd_plot_start:
                    ax_main.axvline(jd_plot_end, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
                
                                                                 
                morph = run_summary.get("morphology", "none")
                params = run_summary.get("params", {})
                
                if morph == "gaussian" and params:
                    t0 = params.get("t0", (jd_start + jd_end) / 2)
                    sigma = params.get("sigma", (jd_end - jd_start) / 4)
                    amp = params.get("amp", 0.1)
                    baseline = params.get("baseline", band_df["mag"].median())
                    
                                               
                    t_fit = np.linspace(jd_start - 3*sigma, jd_end + 3*sigma, 100)
                    mag_fit = gaussian(t_fit, amp, t0, sigma, baseline)
                    
                    t_fit_plot = t_fit - (JD_OFFSET if median_jd > 2000000 else 8000.0)
                    ax_main.plot(
                        t_fit_plot,
                        mag_fit,
                        color="red",
                        linestyle="-",
                        linewidth=2,
                        alpha=0.8,
                        label="Gaussian fit" if run_summary == dip["run_summaries"][0] else "",
                    )
                
                elif morph == "paczynski" and params:
                    t0 = params.get("t0", (jd_start + jd_end) / 2)
                    tE = params.get("tE", (jd_end - jd_start) / 2)
                    amp = params.get("amp", -0.1)
                    baseline = params.get("baseline", band_df["mag"].median())
                    
                                               
                    t_fit = np.linspace(jd_start - 3*tE, jd_end + 3*tE, 100)
                    mag_fit = paczynski_kernel(t_fit, amp, t0, tE, baseline)
                    
                    t_fit_plot = t_fit - (JD_OFFSET if median_jd > 2000000 else 8000.0)
                    ax_main.plot(
                        t_fit_plot,
                        mag_fit,
                        color="blue",
                        linestyle="-",
                        linewidth=2,
                        alpha=0.8,
                        label="Paczynski fit" if run_summary == dip["run_summaries"][0] else "",
                    )
        
                                        
        if jump["significant"] and jump["run_summaries"]:
            for run_summary in jump["run_summaries"]:
                jd_start = run_summary["start_jd"]
                jd_end = run_summary["end_jd"]
                
                jd_plot_start = jd_start - (JD_OFFSET if median_jd > 2000000 else 8000.0)
                jd_plot_end = jd_end - (JD_OFFSET if median_jd > 2000000 else 8000.0)
                ax_main.axvline(jd_plot_start, color="green", linestyle="--", alpha=0.7, linewidth=1.5)
                if jd_plot_end != jd_plot_start:
                    ax_main.axvline(jd_plot_end, color="green", linestyle="--", alpha=0.7, linewidth=1.5)
                
                morph = run_summary.get("morphology", "none")
                params = run_summary.get("params", {})
                
                if morph == "gaussian" and params:
                    t0 = params.get("t0", (jd_start + jd_end) / 2)
                    sigma = params.get("sigma", (jd_end - jd_start) / 4)
                    amp = params.get("amp", -0.1)
                    baseline = params.get("baseline", band_df["mag"].median())
                    
                    t_fit = np.linspace(jd_start - 3*sigma, jd_end + 3*sigma, 100)
                    mag_fit = gaussian(t_fit, amp, t0, sigma, baseline)
                    
                    t_fit_plot = t_fit - (JD_OFFSET if median_jd > 2000000 else 8000.0)
                    ax_main.plot(
                        t_fit_plot,
                        mag_fit,
                        color="green",
                        linestyle="-",
                        linewidth=2,
                        alpha=0.8,
                        label="Jump (Gaussian)" if run_summary == jump["run_summaries"][0] else "",
                    )
                
                elif morph == "paczynski" and params:
                    t0 = params.get("t0", (jd_start + jd_end) / 2)
                    tE = params.get("tE", (jd_end - jd_start) / 2)
                    amp = params.get("amp", -0.1)
                    baseline = params.get("baseline", band_df["mag"].median())
                    
                    t_fit = np.linspace(jd_start - 3*tE, jd_end + 3*tE, 100)
                    mag_fit = paczynski_kernel(t_fit, amp, t0, tE, baseline)
                    
                    t_fit_plot = t_fit - (JD_OFFSET if median_jd > 2000000 else 8000.0)
                    ax_main.plot(
                        t_fit_plot,
                        mag_fit,
                        color="cyan",
                        linestyle="-",
                        linewidth=2,
                        alpha=0.8,
                        label="Jump (Paczynski)" if run_summary == jump["run_summaries"][0] else "",
                    )
        
        ax_main.set_ylabel(f"{band_labels[band]} [mag]", fontsize=12)
        ax_main.grid(True, alpha=0.3)
        if band_idx == 0:
            ax_main.legend(loc="best", fontsize=8, ncol=2)
        
                                     
        ax_resid = axes[1, band_idx]
        if "resid" in band_df.columns:
            for cam in camera_ids:
                subset = band_df[band_df["camera#"] == cam]
                if subset.empty:
                    continue
                color = camera_colors[cam]
                marker = band_markers.get(band, "o")
                ax_resid.scatter(
                    subset["JD_plot"],
                    subset["resid"],
                    s=10,
                    color=color,
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.3,
                    marker=marker,
                )
            
            ax_resid.axhline(0, color="orange", linestyle="-", linewidth=1, alpha=0.5)
            ax_resid.set_ylabel(f"{band_labels[band]} residual [mag]", fontsize=12)
            ax_resid.grid(True, alpha=0.3)
            ax_resid.invert_yaxis()
        
        if band_idx == 1:
            ax_main.set_xlabel("Julian Date - 2458000 [d]", fontsize=12)
            ax_resid.set_xlabel("Julian Date - 2458000 [d]", fontsize=12)
    
           
    title_parts = [f"ID: {asas_sn_id}"]
    g_dip = band_results[0]["dip"]
    g_jump = band_results[0]["jump"]
    v_dip = band_results[1]["dip"]
    v_jump = band_results[1]["jump"]
    
    if g_dip["significant"] or v_dip["significant"]:
        total_dips = g_dip.get("n_runs", 0) + v_dip.get("n_runs", 0)
        title_parts.append(f"Dips: {total_dips} runs (g:{g_dip.get('n_runs', 0)}, V:{v_dip.get('n_runs', 0)})")
    if g_jump["significant"] or v_jump["significant"]:
        total_jumps = g_jump.get("n_runs", 0) + v_jump.get("n_runs", 0)
        title_parts.append(f"Jumps: {total_jumps} runs (g:{g_jump.get('n_runs', 0)}, V:{v_jump.get('n_runs', 0)})")
    fig.suptitle(" | ".join(title_parts), fontsize=14, fontweight="bold")
    
                  
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot light curves with Bayesian event detection results"
    )
    parser.add_argument("csv_paths", nargs="+", help="Path(s) to light curve CSV file(s)")
    parser.add_argument(
        "--results-csv",
        type=Path,
        help="Path to results CSV (optional, for filtering which to plot)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results_bayes_logbf"),
        help="Output directory for plots (default: results_bayes_logbf)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=list(BASELINE_FUNCTIONS.keys()),
        default="per_camera_gp",
        help="Baseline function to use",
    )
    parser.add_argument(
        "--logbf-threshold-dip",
        type=float,
        default=5.0,
        help="Log BF threshold for dips",
    )
    parser.add_argument(
        "--logbf-threshold-jump",
        type=float,
        default=5.0,
        help="Log BF threshold for jumps",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    
    args = parser.parse_args()
    
    baseline_func = BASELINE_FUNCTIONS[args.baseline]
    baseline_kwargs = {}
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
                                              
    csv_paths = [Path(p) for p in args.csv_paths]
    if args.results_csv and args.results_csv.exists():
        results_df = pd.read_csv(args.results_csv)
                                  
        results_ids = set()
        for path in results_df["path"]:
                                                                                 
            if "-light-curves.csv" in str(path):
                id_str = str(path).split("/")[-1].replace("-light-curves.csv", "")
                results_ids.add(id_str)
        
                          
        csv_paths = [
            p for p in csv_paths
            if p.stem.split("-")[0] in results_ids
        ]
        print(f"Filtered to {len(csv_paths)} light curves from results CSV")
    
                           
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"Warning: {csv_path} does not exist, skipping")
            continue
        
        asas_sn_id = csv_path.stem.split("-")[0]
        out_path = args.out_dir / f"{asas_sn_id}_dips.png"
        
        try:
            plot_bayes_results(
                csv_path,
                out_path=out_path,
                show=args.show,
                baseline_func=baseline_func,
                baseline_kwargs=baseline_kwargs,
                logbf_threshold_dip=args.logbf_threshold_dip,
                logbf_threshold_jump=args.logbf_threshold_jump,
            )
        except Exception as e:
            print(f"Error plotting {csv_path}: {e}")

            traceback.print_exc()


if __name__ == "__main__":
    main()
