
columns_raw = ['cam#',
               'median',
               '1siglow', 
               '1sighigh', 
               '90percentlow',
               '90percenthigh']
                # all are mags
columns_dat=["JD",
                "mag",
                'error', 
                'good/bad', 
                'camera#', 
                'band', 
                'camera name'] #1=good, 0 =bad #1=V, 0=g
                # this is equivalent to asassn_columns in vsx_crossmatch.py

''' these steps come after 
   (1) crossmatching candidates to vsx, filtering out undesirable labels 
   (2) searching for the dippers and LTVs

   so only after then we should do the following 
        - get the single highest and lowest median of all 8 cameras at the time of collecting this light curve
   
'''