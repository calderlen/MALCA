import pandas as pd
import numpy as np
import os
from malca.io.fetch import download_lightcurve_by_id
from malca.io.lightcurve_io import load_lightcurve_df
from malca.config import SKYPATROL_CACHE_DIR
import warnings
warnings.filterwarnings('ignore')

csv_path = 'output/runs/runs_march18_bundle_all/results/march18_review_cmd_dustmaps_full.csv'
vetted_path = 'output/runs/runs_march18_bundle_all/results/lc_events_vetted.parquet'

print("Loading data...")
df_csv = pd.read_csv(csv_path)
df_csv['candidate_id'] = df_csv['candidate_id'].astype(str)
dippers = df_csv[df_csv['event_class'] == 'dipper']

df_vetted = pd.read_parquet(vetted_path, columns=['candidate_id', 'asas_sn_id'])
df_vetted['candidate_id'] = df_vetted['candidate_id'].astype(str)

df = dippers.merge(df_vetted, on='candidate_id', how='left')

print(f"Found {len(df)} dippers.")

results = []

for _, row in df.head(10).iterrows(): # Let's test on the first 10
    asas_sn_id = str(row['asas_sn_id'])
    lc_path, _ = download_lightcurve_by_id(asas_sn_id, cache_dir=SKYPATROL_CACHE_DIR)
    if not lc_path:
        print(f"Could not download LC for {asas_sn_id}")
        continue
        
    df_lc = load_lightcurve_df(lc_path)
    
    # Simple measurement: 
    # 1. Baseline is the median of the top 50% of points, or just median
    median_mag = df_lc['mag'].median()
    
    # 2. Depth: Difference between the max magnitude (faintest point) and baseline
    # We use a robust max, like 99th percentile
    max_mag = np.percentile(df_lc['mag'], 99)
    delta_mag = max_mag - median_mag
    
    # Convert delta_mag to fractional depth (delta = 1 - 10^(-dm/2.5))
    fractional_depth = 1.0 - 10.0**(-delta_mag / 2.5)
    
    # 3. Duration: Approximate by finding continuous blocks of points below threshold
    # Threshold = baseline + 0.3 * delta_mag
    thresh = median_mag + 0.3 * delta_mag
    dipping = df_lc['mag'] > thresh
    
    # Find the longest continuous duration
    df_lc['group'] = (dipping != dipping.shift()).cumsum()
    dip_groups = df_lc[dipping].groupby('group')
    
    max_duration = 0
    if len(dip_groups) > 0:
        for name, group in dip_groups:
            if len(group) > 1:
                if 'jd' in group:
                    dur = group['jd'].max() - group['jd'].min()
                else:
                    dur = group['JD'].max() - group['JD'].min()
                if dur > max_duration:
                    max_duration = dur
                    
    results.append({
        'asas_sn_id': asas_sn_id,
        'delta_mag': delta_mag,
        'fractional_depth': fractional_depth,
        'duration_days': max_duration
    })

print("\n--- Dipper Parameter Estimates (Sample) ---")
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
