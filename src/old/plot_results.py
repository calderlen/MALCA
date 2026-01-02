                      
"""
Plot SkyPatrol light curves with vertical lines marking detected peaks.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from df_plot import read_skypatrol_csv, JD_OFFSET
from baseline import (
    global_mean_baseline,
    global_median_baseline,
    global_rolling_median_baseline,
    global_rolling_mean_baseline,
    per_camera_mean_baseline,
    per_camera_median_baseline,
    per_camera_trend_baseline,
)


def parse_peaks_jd(peaks_jd_str):
    """Parse peaks_jd string from CSV (e.g., "[2458327.89, 2458480.59]") into list of floats."""
    if pd.isna(peaks_jd_str) or peaks_jd_str == "" or peaks_jd_str == "[]":
        return []
    try:
                                              
        if isinstance(peaks_jd_str, str):
            peaks = ast.literal_eval(peaks_jd_str)
        else:
            peaks = peaks_jd_str
        return [float(p) for p in peaks if pd.notna(p)]
    except (ValueError, SyntaxError):
        return []


def plot_skypatrol_with_peaks(
    csv_path: Path,
    results_df: pd.DataFrame,
    *,
    out_path: Path | None = None,
    figsize=(16, 10),
    show=False,
    baseline_func=None,
    baseline_kwargs=None,
):
    """Plot a SkyPatrol light curve with detected peaks marked."""
                      
    df = read_skypatrol_csv(csv_path)
    if df.empty:
        print(f"Warning: {csv_path.name} is empty")
        return None
    
                              
    asas_sn_id = csv_path.stem.split("-")[0]
    
                                                                                        
    results_df["asas_sn_id"] = results_df["asas_sn_id"].astype(str)
    
                           
    result_row = results_df[results_df["asas_sn_id"] == str(asas_sn_id)]
    
                                                                    
    if result_row.empty:
                                                                 
        asas_sn_id_clean = str(asas_sn_id).rstrip('0').rstrip('.')
        for idx, row_id in enumerate(results_df["asas_sn_id"]):
            row_id_clean = str(row_id).rstrip('0').rstrip('.')
                                                                     
            if asas_sn_id_clean.startswith(row_id_clean) or row_id_clean.startswith(asas_sn_id_clean):
                                                                                      
                try:
                    id_float = float(asas_sn_id)
                    row_float = float(row_id)
                                                                                   
                    if abs(id_float - row_float) < 1000:
                        result_row = results_df.iloc[[idx]]
                        break
                except (ValueError, TypeError):
                    pass
    if result_row.empty:
        print(f"Warning: No results found for {asas_sn_id}")
        return None
    
    result = result_row.iloc[0]
    
                    
    g_peaks_jd = parse_peaks_jd(result.get("g_peaks_jd", []))
    v_peaks_jd = parse_peaks_jd(result.get("v_peaks_jd", []))
    
                  
    df = df[np.isfinite(df["JD"]) & np.isfinite(df["mag"])].copy()
    median_jd = df["JD"].median()
    if median_jd > 2000000:
        df["JD_plot"] = df["JD"] - JD_OFFSET
        g_peaks_plot = [jd - JD_OFFSET for jd in g_peaks_jd]
        v_peaks_plot = [jd - JD_OFFSET for jd in v_peaks_jd]
    else:
        df["JD_plot"] = df["JD"] - 8000.0
        g_peaks_plot = [jd - 8000.0 for jd in g_peaks_jd]
        v_peaks_plot = [jd - 8000.0 for jd in v_peaks_jd]
    
                             
    bands_present = []
    if (df["v_g_band"] == 0).any():
        bands_present.append(0)          
    if (df["v_g_band"] == 1).any():
        bands_present.append(1)          
    
    if not bands_present:
        print(f"Warning: No valid bands in {csv_path.name}")
        return None
    
                                                                
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True, sharex="col")
    
                       
    camera_ids = sorted(df["camera#"].dropna().unique())
    cmap = plt.get_cmap("tab20", max(len(camera_ids), 1))
    camera_colors = {cam: cmap(i % cmap.N) for i, cam in enumerate(camera_ids)}
    
    band_labels = {0: "g", 1: "V"}
    band_markers = {0: "o", 1: "s"}
    band_peaks = {0: g_peaks_plot, 1: v_peaks_plot}
    band_peak_counts = {0: result.get("g_n_peaks", 0), 1: result.get("v_n_peaks", 0)}
    
                                                      
    for band_idx, band in enumerate([1, 0]):                           
        if band not in bands_present:
                                         
            axes[0, band_idx].set_visible(False)
            axes[1, band_idx].set_visible(False)
            continue
        
                       
        band_df = df[df["v_g_band"] == band].copy()
        
                                                               
                                                                  
        if "error" in band_df.columns:
            error_threshold = 1.0       
            band_df = band_df[band_df["error"] <= error_threshold].copy()
        
                                                                
        if not band_df.empty and "mag" in band_df.columns:
            mag_median = band_df["mag"].median()
            mag_std = band_df["mag"].std()
            if pd.notna(mag_std) and mag_std > 0:
                mag_lower = mag_median - 5 * mag_std
                mag_upper = mag_median + 5 * mag_std
                band_df = band_df[
                    (band_df["mag"] >= mag_lower) & (band_df["mag"] <= mag_upper)
                ].copy()
        
                                          
                                                                                 
        if not band_df.empty:
                                                                                     
            baseline_func_to_use = baseline_func or per_camera_median_baseline
            baseline_kwargs_to_use = baseline_kwargs or {}
            
                                                                          
            if baseline_func_to_use in (per_camera_median_baseline, per_camera_mean_baseline):
                if "days" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["days"] = 300.0
                if "min_points" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["min_points"] = 10
                if "t_col" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["t_col"] = "JD"
                if "mag_col" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["mag_col"] = "mag"
                if "err_col" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["err_col"] = "error"
                if "cam_col" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["cam_col"] = "camera#"
            elif baseline_func_to_use in (global_rolling_median_baseline, global_rolling_mean_baseline):
                if "days" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["days"] = 1000.0
                if "min_points" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["min_points"] = 10
                if "t_col" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["t_col"] = "JD"
                if "mag_col" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["mag_col"] = "mag"
                if "err_col" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["err_col"] = "error"
            elif baseline_func_to_use == per_camera_trend_baseline:
                if "days_short" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["days_short"] = 50.0
                if "days_long" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["days_long"] = 800.0
                if "min_points" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["min_points"] = 10
                if "t_col" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["t_col"] = "JD"
                if "mag_col" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["mag_col"] = "mag"
                if "err_col" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["err_col"] = "error"
                if "cam_col" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["cam_col"] = "camera#"
            else:
                                                                     
                if "t_col" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["t_col"] = "JD"
                if "mag_col" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["mag_col"] = "mag"
                if "err_col" not in baseline_kwargs_to_use:
                    baseline_kwargs_to_use["err_col"] = "error"
            
            band_df_baseline = baseline_func_to_use(band_df, **baseline_kwargs_to_use)
                                                      
            if "JD_plot" not in band_df_baseline.columns:
                band_df_baseline["JD_plot"] = band_df["JD_plot"].values
            band_df_baseline["residual"] = band_df_baseline["mag"] - band_df_baseline["baseline"]
        else:
            band_df_baseline = band_df.copy()
            band_df_baseline["baseline"] = np.nan
            band_df_baseline["residual"] = np.nan
        
                                         
        ax_main = axes[0, band_idx]
        ax_main.invert_yaxis()
        
                                            
        mag_data = []
        err_data = []
        
                                    
        legend_handles = {}
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
                                                
            mag_data.extend(subset["mag"].dropna().tolist())
            err_data.extend(subset["error"].dropna().tolist())
            if cam not in legend_handles:
                legend_handles[cam] = Line2D(
                    [], [],
                    color=color,
                    marker=marker,
                    linestyle="",
                    markeredgecolor="black",
                    markeredgewidth=0.3,
                    label=f"Camera {cam}"
                )
        
                       
        if not band_df_baseline.empty and "baseline" in band_df_baseline.columns:
            baseline_finite = band_df_baseline[np.isfinite(band_df_baseline["baseline"])]
            if not baseline_finite.empty:
                ax_main.plot(
                    baseline_finite["JD_plot"],
                    baseline_finite["baseline"],
                    color="orange",
                    linestyle="-",
                    linewidth=2,
                    alpha=0.8,
                    label="Baseline",
                    zorder=5,
                )
        
                                                
        peaks_plot = band_peaks[band]
        n_peaks = band_peak_counts[band]
        if peaks_plot and n_peaks > 0:
            for peak_jd in peaks_plot:
                ax_main.axvline(
                    peak_jd,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=10,
                )
        
                                             
        band_label = band_labels.get(band, f"band {band}")
        ax_main.set_ylabel(f"{band_label} [mag]")
        ax_main.grid(True, which="both", linestyle="--", alpha=0.3)
        
                                 
        title_parts = [f"{band_label.upper()} band"]
        if n_peaks > 0:
            title_parts.append(f"({n_peaks} peak{'s' if n_peaks > 1 else ''})")
        ax_main.set_title(" ".join(title_parts))
        
                                                                       
        if mag_data:
            mag_with_err = []
            for i, mag_val in enumerate(mag_data):
                if pd.notna(mag_val):
                    err_val = err_data[i] if i < len(err_data) and pd.notna(err_data[i]) else 0
                    mag_with_err.extend([mag_val - err_val, mag_val + err_val])
                                            
            if not band_df_baseline.empty and "baseline" in band_df_baseline.columns:
                baseline_vals = band_df_baseline["baseline"].dropna().tolist()
                mag_with_err.extend(baseline_vals)
            if mag_with_err:
                mag_min = min(mag_with_err)
                mag_max = max(mag_with_err)
                mag_range = mag_max - mag_min
                                                                   
                padding = max(mag_range * 0.01, 0.05)
                ax_main.set_ylim(mag_max + padding, mag_min - padding)                          
        
        if legend_handles:
            ax_main.legend(
                handles=list(legend_handles.values()),
                title="Cameras",
                loc="best",
                fontsize="small",
                ncol=min(len(legend_handles), 5),
            )
        
                                     
        ax_resid = axes[1, band_idx]
        
                                                     
        resid_data = []
        resid_err_data = []
        
        for cam in camera_ids:
            subset = band_df_baseline[band_df_baseline["camera#"] == cam]
            if subset.empty:
                continue
            color = camera_colors[cam]
            marker = band_markers.get(band, "o")
            resid_finite = subset[np.isfinite(subset["residual"])]
            if not resid_finite.empty:
                ax_resid.errorbar(
                    resid_finite["JD_plot"],
                    resid_finite["residual"],
                    yerr=resid_finite["error"],
                    fmt=marker,
                    ms=3,
                    color=color,
                    alpha=0.7,
                    ecolor=color,
                    elinewidth=0.8,
                    capsize=1.5,
                    markeredgecolor="black",
                    markeredgewidth=0.3,
                )
                                                    
                resid_data.extend(resid_finite["residual"].dropna().tolist())
                resid_err_data.extend(resid_finite["error"].dropna().tolist())
        
                                 
        ax_resid.axhline(0, color="gray", linestyle="-", linewidth=1, alpha=0.5, zorder=1)
        
                                                             
        if peaks_plot and n_peaks > 0:
            for peak_jd in peaks_plot:
                ax_resid.axvline(
                    peak_jd,
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=10,
                )
        
                                             
        ax_resid.set_ylabel(f"{band_label} residual [mag]")
        ax_resid.grid(True, which="both", linestyle="--", alpha=0.3)
        ax_resid.set_xlabel(f"Julian Date - {int(JD_OFFSET)} [d]")
        
                                                                           
        if resid_data:
            resid_with_err = []
            for i, resid_val in enumerate(resid_data):
                if pd.notna(resid_val):
                    err_val = resid_err_data[i] if i < len(resid_err_data) and pd.notna(resid_err_data[i]) else 0
                    resid_with_err.extend([resid_val - err_val, resid_val + err_val])
            if resid_with_err:
                resid_min = min(resid_with_err)
                resid_max = max(resid_with_err)
                resid_range = resid_max - resid_min
                                                                   
                padding = max(resid_range * 0.01, 0.05)
                ax_resid.set_ylim(resid_min - padding, resid_max + padding)
        
                                                                           
        if not band_df_baseline.empty:
            resid_with_err = []
            for _, row in band_df_baseline.iterrows():
                resid_val = row.get("residual", np.nan)
                err_val = row.get("error", 0)
                if pd.notna(resid_val):
                    resid_with_err.extend([resid_val - err_val, resid_val + err_val])
            if resid_with_err:
                resid_min = min(resid_with_err)
                resid_max = max(resid_with_err)
                resid_range = resid_max - resid_min
                                                                   
                padding = max(resid_range * 0.01, 0.05)
                ax_resid.set_ylim(resid_min - padding, resid_max + padding)
    
                                              
                                              
                                                             
    
                   
    source_name = result.get("source", asas_sn_id) if "source" in result else asas_sn_id
    category = result.get("category", "") if "category" in result else ""
    title_parts = [source_name]
    if category:
        title_parts.append(f"({category})")
    title_parts.append("Known" if "Known" in str(category) else "New")
    fig.suptitle(" – ".join(title_parts), fontsize=12, fontweight="bold")
    
                  
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def main():
                               
    BASELINE_FUNCTIONS = {
        "global_mean": global_mean_baseline,
        "global_median": global_median_baseline,
        "global_rolling_median": global_rolling_median_baseline,
        "global_rolling_mean": global_rolling_mean_baseline,
        "per_camera_mean": per_camera_mean_baseline,
        "per_camera_median": per_camera_median_baseline,
        "per_camera_trend": per_camera_trend_baseline,
    }
    
    parser = argparse.ArgumentParser(
        description="Plot SkyPatrol light curves with detected peaks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available baseline functions: {', '.join(BASELINE_FUNCTIONS.keys())}"
    )
    parser.add_argument("results_csv", type=Path, help="CSV file with detection results (from test_skypatrol.py)")
    parser.add_argument("--csv-dir", default="data/skypatrol2", help="Directory with SkyPatrol CSV files")
    parser.add_argument("--out-dir", default="plots", help="Output directory for plots")
    parser.add_argument("--format", choices=("pdf", "png", "jpg"), default="png", help="Output format")
    parser.add_argument("--ids", nargs="+", help="Specific ASAS-SN IDs to plot (default: all in results)")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument(
        "--baseline",
        choices=list(BASELINE_FUNCTIONS.keys()),
        default="per_camera_median",
        help="Baseline function to use for calculating residuals (default: per_camera_median)"
    )
    parser.add_argument(
        "--baseline-days",
        type=float,
        help="Days parameter for rolling baselines (ignored for per_camera_trend which uses days_short/days_long)"
    )
    parser.add_argument(
        "--baseline-days-short",
        type=float,
        help="Days_short parameter for per_camera_trend baseline (default: 50.0)"
    )
    parser.add_argument(
        "--baseline-days-long",
        type=float,
        help="Days_long parameter for per_camera_trend baseline (default: 800.0)"
    )
    parser.add_argument(
        "--baseline-min-points",
        type=int,
        default=10,
        help="Minimum points parameter for rolling baselines (default: 10)"
    )
    args = parser.parse_args()
    
                           
    baseline_func = BASELINE_FUNCTIONS[args.baseline]
    
                            
    baseline_kwargs = {}
    
                                                                        
    if baseline_func == per_camera_trend_baseline:
        if args.baseline_days_short is not None:
            baseline_kwargs["days_short"] = args.baseline_days_short
        if args.baseline_days_long is not None:
            baseline_kwargs["days_long"] = args.baseline_days_long
        if args.baseline_min_points is not None:
            baseline_kwargs["min_points"] = args.baseline_min_points
    else:
                                                         
        if args.baseline_days is not None:
            baseline_kwargs["days"] = args.baseline_days
        if args.baseline_min_points is not None:
            baseline_kwargs["min_points"] = args.baseline_min_points
        
                                          
        if args.baseline_days is None:
            if baseline_func in (per_camera_median_baseline, per_camera_mean_baseline):
                baseline_kwargs["days"] = 300.0
            elif baseline_func in (global_rolling_median_baseline, global_rolling_mean_baseline):
                baseline_kwargs["days"] = 1000.0
    
                  
    if not args.results_csv.exists():
        parser.error(f"Results CSV file not found: {args.results_csv}")
    results_df = pd.read_csv(args.results_csv)
    
                                                            
    id_column = None
    for col_name in ["asas_sn_id", "Source_ID", "source_id", "id"]:
        if col_name in results_df.columns:
            id_column = col_name
            break
    
    if id_column is None:
        parser.error(
            f"Could not find ID column in CSV. Expected one of: asas_sn_id, Source_ID, source_id, id\n"
            f"Found columns: {', '.join(results_df.columns)}"
        )
    
                                                
    if id_column != "asas_sn_id":
        results_df["asas_sn_id"] = results_df[id_column].astype(str)
    
    results_df["asas_sn_id"] = results_df["asas_sn_id"].astype(str)
    print(f"Loaded {len(results_df)} results from {args.results_csv}")
    print(f"Using ID column: {id_column}")
    
                                 
    if args.ids:
        ids_to_plot = [str(id) for id in args.ids]
    else:
        ids_to_plot = results_df["asas_sn_id"].tolist()
    
                    
    csv_dir = Path(args.csv_dir)
    if not csv_dir.exists():
        print(f"Error: CSV directory not found: {csv_dir}")
        return 1
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
                           
    plotted = 0
    skipped = 0
    
                                                                 
                                                                                                
    available_csvs = {}
    for csv_path in csv_dir.glob("*-light-curves.csv"):
        csv_id = csv_path.stem.split("-")[0]
                                                                          
        available_csvs[csv_id] = csv_path
        if len(csv_id) >= 9:
            prefix = csv_id[:9]
            if prefix not in available_csvs:
                available_csvs[prefix] = csv_path
    
    for asas_sn_id in ids_to_plot:
                               
        csv_path = csv_dir / f"{asas_sn_id}-light-curves.csv"
        
                                                                   
        if not csv_path.exists():
            csv_path = None
            asas_sn_id_str = str(asas_sn_id).rstrip('0').rstrip('.')
            
                                 
            if asas_sn_id_str in available_csvs:
                csv_path = available_csvs[asas_sn_id_str]
            elif len(asas_sn_id_str) >= 9:
                prefix = asas_sn_id_str[:9]
                if prefix in available_csvs:
                    csv_path = available_csvs[prefix]
            else:
                                                        
                try:
                    id_float = float(asas_sn_id)
                    for csv_id, csv_file in available_csvs.items():
                        try:
                            csv_float = float(csv_id)
                                                                                           
                            if abs(id_float - csv_float) < 1000 and abs(id_float - csv_float) > 0:
                                csv_path = csv_file
                                break
                        except (ValueError, TypeError):
                            continue
                except (ValueError, TypeError):
                    pass
        
        if csv_path is None or not csv_path.exists():
            print(f"Warning: CSV not found for {asas_sn_id}")
            skipped += 1
            continue
        
        out_path = out_dir / f"{asas_sn_id}_peaks.{args.format}"
        try:
            fig = plot_skypatrol_with_peaks(
                csv_path,
                results_df,
                out_path=out_path,
                show=args.show,
                baseline_func=baseline_func,
                baseline_kwargs=baseline_kwargs,
            )
            if fig is not None:
                plotted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"Error plotting {asas_sn_id}: {e}")
            import traceback
            traceback.print_exc()
            skipped += 1
    
    print(f"\nPlotted {plotted} light curve(s) to {out_dir}/")
    if skipped > 0:
        print(f"Skipped {skipped} light curve(s) (no results or errors)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
