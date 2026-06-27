import sqlite3
import json
import pandas as pd
from pathlib import Path

def main():
    # File paths
    csv_path = 'output/runs/runs_march18_bundle_all/results/march18_review_cmd_dustmaps_full.csv'
    db_path = 'output/runs/runs_march18_bundle_all/review/review.taxonomy_filled.db'
    out_path = 'output/runs/runs_march18_bundle_all/results/dipper_candidates_table.csv'
    
    # 1. Read the CSV for G, BP-RP, and Distance
    df_csv = pd.read_csv(csv_path, dtype={'source_id': str, 'candidate_id': str})
    
    # Filter only dippers
    df_dippers = df_csv[df_csv['event_class'] == 'dipper'].copy()
    
    # Select columns we need from CSV
    csv_cols = ['candidate_id', 'source_id', 'phot_g_mean_mag', 'bp_rp', 'distance_gspphot']
    df_dippers = df_dippers[csv_cols]
    
    # 2. Extract properties from DB
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Get all candidate_ids that are dippers
    dipper_cids = df_dippers['candidate_id'].astype(str).tolist()
    
    c.execute('SELECT candidate_id, payload_json FROM candidates')
    
    db_rows = []
    for cid, p_json in c.fetchall():
        if str(cid) in dipper_cids:
            p = json.loads(p_json)
            ext = p.get('external_stats', {})
            lc = p.get('lc_stats', {})
            
            db_rows.append({
                'candidate_id': cid,
                'ra': ext.get('ra_deg'),
                'dec': ext.get('dec_deg'),
                'tdip': lc.get('dip_best_t0'),
                'n_dips': lc.get('dipper_n_dips')
            })
            
    df_db = pd.DataFrame(db_rows)
    
    # Merge CSV and DB data
    # Convert candidate_id to strings to ensure merge works
    df_dippers['candidate_id'] = df_dippers['candidate_id'].astype(str)
    df_db['candidate_id'] = df_db['candidate_id'].astype(str)
    
    df_final = pd.merge(df_dippers, df_db, on='candidate_id', how='left')
    
    # 3. Format the table
    # Distance in CSV is usually in pc. To convert to kpc, divide by 1000
    if 'distance_gspphot' in df_final.columns:
        df_final['Distance (kpc)'] = df_final['distance_gspphot'] / 1000.0
    else:
        df_final['Distance (kpc)'] = pd.NA
        
    # Rename columns to match desired table
    df_final = df_final.rename(columns={
        'source_id': 'Gaia DR3 ID',
        'ra': 'R.A.',
        'dec': 'Decl.',
        'phot_g_mean_mag': 'G (mag)',
        'bp_rp': 'GBP - GRP (mag)',
        'tdip': 'tdip (MJD)',
        'n_dips': 'NDips'
    })
    
    # Select and order final columns
    final_cols = [
        'Gaia DR3 ID', 
        'R.A.', 
        'Decl.', 
        'G (mag)', 
        'GBP - GRP (mag)', 
        'Distance (kpc)', 
        'tdip (MJD)', 
        'NDips'
    ]
    
    # Round columns to look nice
    df_final['R.A.'] = df_final['R.A.'].round(5)
    df_final['Decl.'] = df_final['Decl.'].round(5)
    df_final['G (mag)'] = df_final['G (mag)'].round(3)
    df_final['GBP - GRP (mag)'] = df_final['GBP - GRP (mag)'].round(3)
    df_final['Distance (kpc)'] = df_final['Distance (kpc)'].round(3)
    df_final['tdip (MJD)'] = df_final['tdip (MJD)'].round(3)
    
    # Write out to CSV
    df_final[final_cols].to_csv(out_path, index=False)
    print(f"Successfully generated table for {len(df_final)} dippers.")
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
