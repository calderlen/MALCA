import re

with open("malca/ltv/neowise.py", "r") as f:
    code = f.read()

old_code = """    # Write to memory buffer in IPAC format
    f_str = io.StringIO()
    t.write(f_str, format='ipac')
    f_bytes = io.BytesIO(f_str.getvalue().encode('utf-8'))
    
    query = f\"\"\"
    SELECT 
        db.mjd AS mjd, db.w1mpro AS w1mpro, db.w1sigmpro AS w1sigmpro, db.w1snr AS w1snr, 
        db.w2mpro AS w2mpro, db.w2sigmpro AS w2sigmpro, db.w2snr AS w2snr, 
        db.qual_frame AS qual_frame, db.cc_flags AS cc_flags, 
        my_table.target_id AS target_id
    FROM neowiser_p1bs_psd AS db, TAP_UPLOAD.my_table AS my_table
    WHERE CONTAINS(POINT(db.ra, db.dec), CIRCLE(my_table.ra, my_table.dec, {match_radius_arcsec / 3600.0})) = 1
    \"\"\"

    files = {'table.tbl': f_bytes}
    data = {
        'UPLOAD': 'my_table,param:table.tbl',
        'FORMAT': 'VOTABLE',
        'QUERY': query
    }

    try:
        if verbose:
            print(f"  Sending TAP query for {len(df)} targets...")
        
        response = requests.post('https://irsa.ipac.caltech.edu/TAP/sync', files=files, data=data, timeout=600)
        
        if response.status_code == 200:
            if b"ERROR" in response.content:
                 if verbose:
                     print(f"NEOWISE query error: {response.content.decode('utf-8')[:200]}")
                 return pd.DataFrame()
            else:
                 try:
                     result_table = Table.read(io.BytesIO(response.content), format='votable')
                     res_df = result_table.to_pandas()
                     if len(res_df.columns) == 10:
                         res_df.columns = [
                             'mjd', 'w1mpro', 'w1sigmpro', 'w1snr',
                             'w2mpro', 'w2sigmpro', 'w2snr',
                             'qual_frame', 'cc_flags', 'target_id'
                         ]
                     
                     # Filter bad data (same logic as before)
                     if "qual_frame" in res_df.columns:
                         res_df = res_df[res_df["qual_frame"].isin([0, 1])]
                     if "cc_flags" in res_df.columns:
                         # Need to handle bytes in astropy votable pandas conversion
                         if len(res_df) > 0 and res_df["cc_flags"].dtype == object and isinstance(res_df["cc_flags"].iloc[0], bytes):
                             res_df["cc_flags"] = res_df["cc_flags"].str.decode("utf-8")
                         res_df = res_df[~res_df["cc_flags"].str.contains("[^0]", regex=True, na=False)]
                     if "w1snr" in res_df.columns:
                         res_df = res_df[res_df["w1snr"] >= MIN_SNR]
                     if "w2snr" in res_df.columns:
                         res_df = res_df[res_df["w2snr"] >= MIN_SNR]
                         
                     # Rename target_id back to original id_col
                     res_df = res_df.rename(columns={"target_id": id_col})
                     return res_df.reset_index(drop=True)
                     
                 except Exception as e:
                     if verbose:
                         print(f"NEOWISE table parse error: {e}")
                     return pd.DataFrame()
        else:
            if verbose:
                print(f"NEOWISE query HTTP {response.status_code}: {response.content.decode('utf-8')[:200]}")
            return pd.DataFrame()
            
    except Exception as e:
        if verbose:
            print(f"NEOWISE request error: {e}")
        return pd.DataFrame()"""

new_code = """    query = f\"\"\"
    SELECT 
        db.mjd AS mjd, db.w1mpro AS w1mpro, db.w1sigmpro AS w1sigmpro, db.w1snr AS w1snr, 
        db.w2mpro AS w2mpro, db.w2sigmpro AS w2sigmpro, db.w2snr AS w2snr, 
        db.qual_frame AS qual_frame, db.cc_flags AS cc_flags, 
        my_table.target_id AS target_id
    FROM neowiser_p1bs_psd AS db, TAP_UPLOAD.my_table AS my_table
    WHERE CONTAINS(POINT(db.ra, db.dec), CIRCLE(my_table.ra, my_table.dec, {match_radius_arcsec / 3600.0})) = 1
    \"\"\"

    try:
        import pyvo
        tap = pyvo.dal.TAPService('https://irsa.ipac.caltech.edu/TAP')
        if verbose:
            print(f"  Sending async TAP query for {len(df)} targets to IRSA...")
        
        job = tap.run_async(query, uploads={"my_table": t})
        res_df = job.to_table().to_pandas()
        
        if res_df.empty:
            return res_df
            
        if len(res_df.columns) == 10:
            res_df.columns = [
                'mjd', 'w1mpro', 'w1sigmpro', 'w1snr',
                'w2mpro', 'w2sigmpro', 'w2snr',
                'qual_frame', 'cc_flags', 'target_id'
            ]
        
        # Filter bad data
        if "qual_frame" in res_df.columns:
            res_df = res_df[res_df["qual_frame"].isin([0, 1])]
        if "cc_flags" in res_df.columns:
            if len(res_df) > 0 and res_df["cc_flags"].dtype == object and isinstance(res_df["cc_flags"].iloc[0], bytes):
                res_df["cc_flags"] = res_df["cc_flags"].str.decode("utf-8")
            res_df = res_df[~res_df["cc_flags"].str.contains("[^0]", regex=True, na=False)]
        if "w1snr" in res_df.columns:
            res_df = res_df[res_df["w1snr"] >= MIN_SNR]
        if "w2snr" in res_df.columns:
            res_df = res_df[res_df["w2snr"] >= MIN_SNR]
            
        # Rename target_id back to original id_col
        res_df = res_df.rename(columns={"target_id": id_col})
        return res_df.reset_index(drop=True)
            
    except Exception as e:
        if verbose:
            print(f"NEOWISE async request error: {e}")
        return pd.DataFrame()"""

if old_code in code:
    with open("malca/ltv/neowise.py", "w") as f:
        f.write(code.replace(old_code, new_code))
    print("Replaced successfully.")
else:
    print("Old code not found.")
