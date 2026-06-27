import sqlite3
import pandas as pd
from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord
from pathlib import Path
import warnings
import json

def update_dipper_photometry(db_path: str):
    db_path = Path(db_path).expanduser()
    print(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # 1. Get dipper IDs from CSV
    csv_path = 'output/runs/runs_march18_bundle_all/results/march18_review_cmd_dustmaps_full.csv'
    df_csv = pd.read_csv(csv_path)
    dipper_ids = df_csv.loc[df_csv['event_class'] == 'dipper', 'candidate_id'].astype(str).tolist()
    
    # 2. Get RA and Dec from characterized.parquet
    char_df = pd.read_parquet('output/runs/runs_march18_bundle_all/results/characterized.parquet')
    if 'candidate_id' in char_df.columns:
        id_col = 'candidate_id'
    elif 'asas_sn_id' in char_df.columns:
        id_col = 'asas_sn_id'
        
    char_df[id_col] = char_df[id_col].astype(str)
    coords_df = char_df[char_df[id_col].isin(dipper_ids)].set_index(id_col)
    
    # 3. Get payload_json from database
    query = f"""
    SELECT candidate_id, payload_json
    FROM candidates
    WHERE candidate_id IN ({','.join(['?']*len(dipper_ids))})
    """
    df_db = pd.read_sql(query, conn, params=dipper_ids)
    print(f"Found {len(df_db)} Dippers in DB.")
    
    # Query Vizier
    vphas_vizier = Vizier(columns=["r_mag", "i_mag", "Ha_mag"], timeout=60, row_limit=1)
    allwise_vizier = Vizier(columns=["W1mag", "W2mag"], timeout=60, row_limit=1)
    tmass_vizier = Vizier(columns=["Hmag", "Kmag"], timeout=60, row_limit=1)
    
    updates = []
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i, row in df_db.iterrows():
            cid = str(row['candidate_id'])
            p_json = row['payload_json']
            p = json.loads(p_json)
            ext = p.get('external_stats', {})
            
            if cid not in coords_df.index:
                continue
                
            ra = coords_df.loc[cid, 'ra']
            dec = coords_df.loc[cid, 'dec']
            
            if pd.isna(ra) or pd.isna(dec):
                continue
                
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            
            # VPHAS+
            try:
                v_res = vphas_vizier.query_region(coord, radius=2*u.arcsec, catalog="II/341/vphasp")
                if len(v_res) > 0 and len(v_res[0]) > 0:
                    match = v_res[0].to_pandas().iloc[0]
                    if pd.notna(match.get('r_mag')) and pd.notna(match.get('Ha_mag')):
                        ext['vphas_r_ha'] = float(match['r_mag'] - match['Ha_mag'])
                    if pd.notna(match.get('r_mag')) and pd.notna(match.get('i_mag')):
                        ext['vphas_r_i'] = float(match['r_mag'] - match['i_mag'])
            except Exception as e:
                pass
                
            # AllWISE
            try:
                a_res = allwise_vizier.query_region(coord, radius=3*u.arcsec, catalog="II/328/allwise")
                if len(a_res) > 0 and len(a_res[0]) > 0:
                    match = a_res[0].to_pandas().iloc[0]
                    w1 = match.get('W1mag', None)
                    w2 = match.get('W2mag', None)
                    if pd.notna(w1): ext['w1'] = float(w1)
                    if pd.notna(w2): ext['w2'] = float(w2)
                    if pd.notna(w1) and pd.notna(w2):
                        ext['w1_w2'] = float(w1 - w2)
            except Exception as e:
                pass
            
            # 2MASS
            try:
                t_res = tmass_vizier.query_region(coord, radius=3*u.arcsec, catalog="II/246/out")
                if len(t_res) > 0 and len(t_res[0]) > 0:
                    match = t_res[0].to_pandas().iloc[0]
                    hmag = match.get('Hmag', None)
                    kmag = match.get('Kmag', None)
                    if pd.notna(hmag) and pd.notna(kmag):
                        ext['H_K'] = float(hmag - kmag)
            except Exception as e:
                pass

            p['external_stats'] = ext
            updates.append((json.dumps(p), cid))
    
    # Apply updates
    cursor = conn.cursor()
    updated_count = 0
    for new_json, cid in updates:
        cursor.execute("UPDATE candidates SET payload_json = ? WHERE candidate_id = ?", (new_json, cid))
        updated_count += 1
            
    conn.commit()
    conn.close()
    print(f"Updated payload_json for {updated_count} rows in candidates.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    args = parser.parse_args()
    update_dipper_photometry(args.db)
