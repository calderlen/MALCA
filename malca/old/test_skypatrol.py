                      
"""
Test the event finder pipeline on SkyPatrol light curves.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys

from old.lc_events import prefix_metrics
import numpy as np
import pandas as pd

from malca.baseline import (
    global_mean_baseline,
    global_median_baseline,
    global_rolling_median_baseline,
    global_rolling_mean_baseline,
    per_camera_mean_baseline,
    per_camera_median_baseline,
    per_camera_trend_baseline,
)
from malca.old.lc_events import lc_band_proc, per_camera_median_baseline
from malca.plot import read_skypatrol_csv






def process_skypatrol_csv(csv_path: Path, **kwargs) -> dict:
    """Process a single SkyPatrol CSV file through the pipeline."""
                  
    df = read_skypatrol_csv(csv_path)
    
    if df.empty:
        asas_sn_id = csv_path.stem.split("-")[0]
        return {"asas_sn_id": asas_sn_id, "error": "empty file"}
    
                                                                                        
    asas_sn_id = csv_path.stem.split("-")[0]
    
                              
    df_g = df[df["v_g_band"] == 0].copy()
    df_v = df[df["v_g_band"] == 1].copy()
    
                    
    peak_kwargs = kwargs.get("peak_kwargs", {})
    sigma_threshold = float(peak_kwargs.get("sigma_threshold", 3.0))
    metrics_dip_threshold = sigma_threshold
    baseline_func = kwargs.get("baseline_func", per_camera_median_baseline)
    baseline_kwargs = kwargs.get("baseline_kwargs", {})
    mode = kwargs.get("mode", "dips")
    
                       
    g_res = lc_band_proc(
        df_g,
        mode=mode,
        peak_kwargs=peak_kwargs,
        sigma_threshold=sigma_threshold,
        metrics_dip_threshold=metrics_dip_threshold,
        baseline_func=baseline_func,
        baseline_kwargs=baseline_kwargs,
    )
    v_res = lc_band_proc(
        df_v,
        mode=mode,
        peak_kwargs=peak_kwargs,
        sigma_threshold=sigma_threshold,
        metrics_dip_threshold=metrics_dip_threshold,
        baseline_func=baseline_func,
        baseline_kwargs=baseline_kwargs,
    )
    
                  
    jd_first = np.nan
    jd_last = np.nan
    if not df_g.empty:
        jd_first = float(df_g["JD"].iloc[0])
        jd_last = float(df_g["JD"].iloc[-1])
    if not df_v.empty:
        if np.isnan(jd_first):
            jd_first = float(df_v["JD"].iloc[0])
        if np.isnan(jd_last):
            jd_last = float(df_v["JD"].iloc[-1])
    
                      

    g_metrics = prefix_metrics("g", g_res["metrics"])
    v_metrics = prefix_metrics("v", v_res["metrics"])
    
                                                                      
    g_peaks_jd = g_res["peaks_jd"]
    v_peaks_jd = v_res["peaks_jd"]
    
    row = {
        "asas_sn_id": asas_sn_id,
        "mag_bin": "test",
        "g_n_peaks": g_res["n"],
        "g_biweight_R": g_res["R"],
        "g_peaks_idx": g_res["peaks_idx"],
        "g_peaks_jd": g_peaks_jd,
        "g_total_score": g_res["fit"]["total_score"],
        "g_best_score": g_res["fit"]["best_score"],
        "v_n_peaks": v_res["n"],
        "v_biweight_R": v_res["R"],
        "v_peaks_idx": v_res["peaks_idx"],
        "v_peaks_jd": v_peaks_jd,
        "v_total_score": v_res["fit"]["total_score"],
        "v_best_score": v_res["fit"]["best_score"],
        "jd_first": jd_first,
        "jd_last": jd_last,
        "n_rows_g": int(len(df_g)) if not df_g.empty else 0,
        "n_rows_v": int(len(df_v)) if not df_v.empty else 0,
    }
    row.update(g_metrics)
    row.update(v_metrics)
    
    return row


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
        description="Test event finder on SkyPatrol light curves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available baseline functions: {', '.join(BASELINE_FUNCTIONS.keys())}"
    )
    parser.add_argument("csv_files", nargs="+", help="SkyPatrol CSV file(s) or directory")
    parser.add_argument("--out", default="test_results.csv", help="Output CSV file")
    parser.add_argument("--sigma-threshold", type=float, default=3.0, help="Sigma threshold for peak detection")
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
    
                           
    csv_paths = []
    for path_str in args.csv_files:
        path = Path(path_str)
        if path.is_dir():
            csv_paths.extend(sorted(path.glob("*-light-curves.csv")))
        elif path.exists():
            csv_paths.append(path)
    
    if not csv_paths:
        print("No CSV files found!")
        return 1
    
    print(f"Processing {len(csv_paths)} light curve(s)...")
    
    results = []
    for csv_path in csv_paths:
        print(f"Processing {csv_path.name}...")
        result = process_skypatrol_csv(
            csv_path,
            mode="dips",
            peak_kwargs={"sigma_threshold": args.sigma_threshold},
            baseline_func=baseline_func,
            baseline_kwargs=baseline_kwargs,
        )
        results.append(result)
    
                  
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print(f"\nResults saved to {args.out}")
    
                   
    detected = df[
        (df["g_n_peaks"].fillna(0) > 0) | (df["v_n_peaks"].fillna(0) > 0)
    ]
    print(f"\nSummary:")
    print(f"  Total processed: {len(df)}")
    print(f"  Detected (g or v peaks > 0): {len(detected)}")
    print(f"  g-band detections: {(df['g_n_peaks'].fillna(0) > 0).sum()}")
    print(f"  v-band detections: {(df['v_n_peaks'].fillna(0) > 0).sum()}")
    
    if len(detected) > 0:
        print(f"\nDetected objects:")
        for _, row in detected.iterrows():
            print(f"  {row['asas_sn_id']}: g={int(row['g_n_peaks'])}, v={int(row['v_n_peaks'])}")
    
    return 0


if __name__ == "__main__":

    sys.exit(main())
