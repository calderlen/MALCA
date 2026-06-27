import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from malca.evaluation.dip_injection import load_efficiency_cube, plot_efficiency_marginalized
from malca.io.fetch import download_lightcurve_by_id
from malca.io.lightcurve_io import load_lightcurve_df
from malca.config import SKYPATROL_CACHE_DIR

def measure_dipper_properties(df_dippers):
    results = []
    print(f"Measuring empirical properties for {len(df_dippers)} dippers...")
    for idx, row in df_dippers.iterrows():
        asas_sn_id = str(row['asas_sn_id'])
        lc_path, _ = download_lightcurve_by_id(asas_sn_id, cache_dir=SKYPATROL_CACHE_DIR)
        if not lc_path:
            print(f"  Warning: Could not download LC for {asas_sn_id}")
            continue
            
        df_lc = load_lightcurve_df(lc_path)
        if 'jd' not in df_lc.columns:
            if 'JD' in df_lc.columns:
                df_lc['jd'] = df_lc['JD']
            else:
                continue
                
        # Simple empirical measurement
        median_mag = df_lc['mag'].median()
        max_mag = np.percentile(df_lc['mag'], 99)
        delta_mag = max_mag - median_mag
        fractional_depth = 1.0 - 10.0**(-delta_mag / 2.5)
        
        thresh = median_mag + 0.3 * delta_mag
        dipping = df_lc['mag'] > thresh
        df_lc['group'] = (dipping != dipping.shift()).cumsum()
        dip_groups = df_lc[dipping].groupby('group')
        
        max_duration = 0
        if len(dip_groups) > 0:
            for name, group in dip_groups:
                if len(group) > 1:
                    dur = group['jd'].max() - group['jd'].min()
                    if dur > max_duration:
                        max_duration = dur
                        
        if max_duration > 0:
            results.append({
                'candidate_id': row['candidate_id'],
                'asas_sn_id': asas_sn_id,
                'delta_mag': delta_mag,
                'fractional_depth': fractional_depth,
                'duration_days': max_duration
            })
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Plot dipper occurrence using inverse-efficiency from dip-injection")
    parser.add_argument("--efficiency-parquet", type=str, required=True, help="Path to dip_efficiency.parquet from cluster")
    parser.add_argument("--review-csv", type=str, default="output/runs/runs_march18_bundle_all/results/march18_review_cmd_dustmaps_full.csv")
    parser.add_argument("--vetted-parquet", type=str, default="output/runs/runs_march18_bundle_all/results/lc_events_vetted.parquet")
    parser.add_argument("--output", type=str, default="output/dipper_occurrence.pdf", help="Path to save the plot")
    
    args = parser.parse_args()
    
    eff_path = Path(args.efficiency_parquet)
    if not eff_path.exists():
        print(f"Error: Efficiency parquet not found at {eff_path}")
        print("Please run `malca dip-injection ...` on the cluster first and download the resulting file.")
        return
        
    print(f"Loading efficiency cube from {eff_path}...")
    cube = load_efficiency_cube(eff_path)
    
    # The cube marginalized over magnitude (which is what we plot)
    eff_2d = np.nanmean(cube["efficiency"], axis=2)
    # Axes: x=duration, y=depth
    x_centers = cube["duration_centers"]
    y_centers = cube["depth_centers"]
    
    print("Loading dipper candidates...")
    df_csv = pd.read_csv(args.review_csv)
    df_csv['candidate_id'] = df_csv['candidate_id'].astype(str)
    df_dippers = df_csv[df_csv['event_class'] == 'dipper']
    
    df_vetted = pd.read_parquet(args.vetted_parquet, columns=['candidate_id', 'asas_sn_id'])
    df_vetted['candidate_id'] = df_vetted['candidate_id'].astype(str)
    
    df_dippers = df_dippers.merge(df_vetted, on='candidate_id', how='left')
    print(f"Found {len(df_dippers)} vetted dipper candidates.")
    
    # Measure empirical depth/duration
    df_measured = measure_dipper_properties(df_dippers)
    
    # Interpolate inverse-efficiency
    print("Interpolating efficiencies...")
    # x_centers is duration, y_centers is depth
    # RegularGridInterpolator expects (x, y) coordinates matching the grid axes
    # Replace NaNs or zeros in efficiency with a small number to avoid div by zero
    eff_safe = np.clip(eff_2d, 1e-5, 1.0)
    
    interp = RegularGridInterpolator((x_centers, y_centers), eff_safe, bounds_error=False, fill_value=np.nan)
    
    durations = df_measured['duration_days'].values
    depths = df_measured['fractional_depth'].values
    
    # Query the interpolator
    pts = np.vstack((durations, depths)).T
    efficiencies = interp(pts)
    
    # Filter points that are outside the grid (NaN)
    valid = ~np.isnan(efficiencies)
    
    sum_inverse_eff = np.sum(1.0 / efficiencies[valid])
    
    print(f"\n--- Occurrence Rate Calculation ---")
    print(f"Total dippers measured: {len(df_measured)}")
    print(f"Dippers within efficiency grid: {np.sum(valid)}")
    print(f"Estimated True Occurrence (N_true): {sum_inverse_eff:.1f} events")
    
    # Plotting
    print(f"\nGenerating plot...")
    fig = plot_efficiency_marginalized(cube, axis="mag", show=False)
    
    # The main heatmap is axes[0]
    ax = fig.axes[0]
    
    # Overlay the dippers
    scatter = ax.scatter(
        durations, 
        depths, 
        marker='*', 
        s=150, 
        color='white', 
        edgecolor='black', 
        linewidth=0.5,
        zorder=10,
        label='Vetted Dippers'
    )
    
    # Only add legend if we successfully plotted
    if len(durations) > 0:
        ax.legend(loc='upper right', framealpha=0.9)
    
    # Add rate text to plot
    text_str = f"$N_{{true}} \\approx {sum_inverse_eff:.0f}$"
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Plot saved to {out_path}")
    
if __name__ == "__main__":
    main()
