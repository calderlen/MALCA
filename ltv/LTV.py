import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from astropy.stats import mad_std
from scipy import stats
from astropy.table import Table
import os
import warnings
import time
import csv
from tqdm import tqdm
import argparse

# Flags set by cli arguments
def parse_args():
    parser = argparse.ArgumentParser(prog='LTvar.py')

    parser.add_argument("--root", default="/data/poohbah/1/assassin/rowan.90/lcsv2/", type=str, help="Root folder containing mag-bin subdirectories")
    parser.add_argument("--mag-bin", default="13_13.5", type=str, help="Magnitude bin subdirectory, e.g. 13_13.5")
    parser.add_argument("--output", default="LTvar13_13.5.csv", type=str, help="Output CSV filename")
    parser.add_argument("--dir-start", type=int, default=0, help="First lcXX_cal index (inclusive)")
    parser.add_argument("--dir-end", type=int, default=30, help="Last lcXX_cal index (inclusive)")
    return parser.parse_args()

args = parse_args()
ROOT = args.root
MAG_BIN = args.mag_bin
OUTPUT = args.output
DIR_START = args.dir_start
DIR_END = args.dir_end

columns = ["jd", "mag", 'error', 'good/bad', 'camera', 'v/g?', 'saturated/unsaturated', 'camera,field']

directories = list(map(str, range(DIR_START, DIR_END + 1)))

# Light curve columns

ltv_path = f"LTvar{MAG_BIN.replace('_','-')}.csv" ######WIILL NEED TO CHANGE THIS IF YOU DON'T NAME IT MAG_BIN######
ltv = Table.read(ltv_path)

id = np.array(ltv['ASAS-SN ID']).tolist()
mag = np.array(ltv['Pstarss gmag']).tolist()
median = np.array(ltv['Median']).tolist()
median_err = np.array(ltv['Median_err']).tolist()
dispersion = np.array(ltv['Dispersion']).tolist()
slope = np.array(ltv['Slope']).tolist()
quad_slope = np.array(ltv['Quad Slope']).tolist()
coeff1 = np.array(ltv['coeff1']).tolist()
coeff2 = np.array(ltv['coeff2']).tolist()
Diff = np.array(ltv['max diff']).tolist()

for x in directories:
    start_time = time.time()
    print(f'Starting{MAG_BIN}' + x + ' directory')

    ID = pd.read_table(f'{ROOT}{MAG_BIN}/index' + x + '.csv', sep=r'\,|\t', engine='python')
    directory = f'{ROOT}{MAG_BIN}/lc' + x + '_cal/'

    files = [f for f in os.listdir(directory) if f.endswith('.dat')]
    
    for file in tdqm(files, desc=f'Processing lc{x}_cal', unit='file'):
        path = os.path.join(directory, file)
        target = [file.split('.')[0]]
        Target = [int(i) for i in target]
        Target = Target[0]

        ra = np.where(ID['asas_sn_id'] == Target)
        
        df = pd.read_table(path, sep="\s+", names=columns)
        df = df[df['good/bad'] == 1] # ltv: good=1; dippers, may need both
        df = df[df['v/g?'] == 0]  # g-band: 0, V-band: 1

        df['JD'] = df['JD'] + 2450000

        if Target == 17181160895:
            df.drop(df[df['good/bad'] < 1].index, inplace = True)
            df.drop(df[df['jd'] < 2.458e+06].index, inplace = True)
            
        RA = ID['ra_deg'].iloc[ra]

        #pstarr mag
        p_mag = float(np.array(ID['pstarrs_g_mag'].iloc[ra]))
        
        lc_median = np.median(df['mag'])
        lc_mad = mad_std(df['mag'])  # median absolute deviation, robust std dev
        lc_dispersion = np.ptp(df['mag'])  # peak to peak
        
        # computing seasonal medians
        indices = np.arange(0, DIR_END + 1) # Sydney had this as not inclusive of last number, why?
        dspring = 2460023.5

        Mid = [] 

        for n in indices:
            date1 = dspring + 365.25*(RA-12.0)/24.0 #target overhead at midnight at date1
            date2 = date1 +  365.25/2.0 +365.25 #same RA as the sun at date2
            mid = date2 - n*365.25 #seasonal gaps (where sun is blocking target)
            Mid.append(mid)

        mid = np.array(Mid)

        if len(df.JD) == 0:
            continue
        
        mid = [x for x in mid if x < max(df.JD) and x > min(df.JD)]
        mid_length = len(mid)

        if mid_length == 1:
            continue
    
        lens = []
        lens.append(mid_length)

         if len(mid) > 11:
            lc1 = [x for x in df['jd'] if x < mid[-1]]
            lc2 = [x for x in df['jd'] if x > mid[-1] and x < mid[-2]]
            lc3 = [x for x in df['jd'] if x > mid[-2] and x < mid[-3]]
            lc4 = [x for x in df['jd'] if x > mid[-3] and x < mid[-4]]
            lc5 = [x for x in df['jd'] if x > mid[-4] and x < mid[-5]]
            lc6 = [x for x in df['jd'] if x > mid[-5] and x < mid[-6]]
            lc7 = [x for x in df['jd'] if x > mid[-6] and x < mid[-7]]
            lc8 = [x for x in df['jd'] if x > mid[-7] and x < mid[-8]]
            lc9 = [x for x in df['jd'] if x > mid[-8] and x < mid[-9]]
            lc10 = [x for x in df['jd'] if x > mid[-9] and x < mid[-10]]
            lc11 = [x for x in df['jd'] if x > mid[-10] and x < mid[-11]]
            lc12 = [x for x in df['jd'] if x > mid[-11]]

        elif len(mid) > 10:
            lc1 = [x for x in df['jd'] if x < mid[-1]]
            lc2 = [x for x in df['jd'] if x > mid[-1] and x < mid[-2]]
            lc3 = [x for x in df['jd'] if x > mid[-2] and x < mid[-3]]
            lc4 = [x for x in df['jd'] if x > mid[-3] and x < mid[-4]]
            lc5 = [x for x in df['jd'] if x > mid[-4] and x < mid[-5]]
            lc6 = [x for x in df['jd'] if x > mid[-5] and x < mid[-6]]
            lc7 = [x for x in df['jd'] if x > mid[-6] and x < mid[-7]]
            lc8 = [x for x in df['jd'] if x > mid[-7] and x < mid[-8]]
            lc9 = [x for x in df['jd'] if x > mid[-8] and x < mid[-9]]
            lc10 = [x for x in df['jd'] if x > mid[-9] and x < mid[-10]]
            lc11 = [x for x in df['jd'] if x > mid[-10]]
            lc12 = []

        elif len(mid) <= 2:
            print(target)
            try: 
                lc1 = [x for x in df['jd'] if x < mid[-1]]
                lc2 = [x for x in df['jd'] if x > mid[-1] and x < mid[-2]]
                lc3 = [x for x in df['jd'] if x > mid[-2]]
                lc4 = []
                lc5 = []
                lc6 = []
                lc7 = []
                lc8 = []
                lc9 = []
                lc10 = []
                lc11 = []
                lc12 = []
            except (IndexError):
                continue
    
        elif len (mid) <= 3:
            lc1 = [x for x in df['jd'] if x < mid[-1]]
            lc2 = [x for x in df['jd'] if x > mid[-1] and x < mid[-2]]
            lc3 = [x for x in df['jd'] if x > mid[-2] and x < mid[-3]]
            lc4 = [x for x in df['jd'] if x > mid[-3]]
            lc5 = []
            lc6 = []
            lc7 = []
            lc8 = []
            lc9 = []
            lc10 = []
            lc11 = []
            lc12 = []
            
        elif len (mid) <= 4:
            lc1 = [x for x in df['jd'] if x < mid[-1]]
            lc2 = [x for x in df['jd'] if x > mid[-1] and x < mid[-2]]
            lc3 = [x for x in df['jd'] if x > mid[-2] and x < mid[-3]]
            lc4 = [x for x in df['jd'] if x > mid[-3] and x < mid[-4]]
            lc5 = [x for x in df['jd'] if x > mid[-4]]
            lc6 = []
            lc7 = []
            lc8 = []
            lc9 = []
            lc10 = []
            lc11 = []
            lc12 = []
            
        elif len (mid) <= 5:
            lc1 = [x for x in df['jd'] if x < mid[-1]]
            lc2 = [x for x in df['jd'] if x > mid[-1] and x < mid[-2]]
            lc3 = [x for x in df['jd'] if x > mid[-2] and x < mid[-3]]
            lc4 = [x for x in df['jd'] if x > mid[-3] and x < mid[-4]]
            lc5 = [x for x in df['jd'] if x > mid[-4] and x < mid[-5]]
            lc6 = [x for x in df['jd'] if x > mid[-5]]
            lc7 = []
            lc8 = []
            lc9 = []
            lc10 = []
            lc11 = []
            lc12 = []
            
        elif len (mid) <= 6:
            lc1 = [x for x in df['jd'] if x < mid[-1]]
            lc2 = [x for x in df['jd'] if x > mid[-1] and x < mid[-2]]
            lc3 = [x for x in df['jd'] if x > mid[-2] and x < mid[-3]]
            lc4 = [x for x in df['jd'] if x > mid[-3] and x < mid[-4]]
            lc5 = [x for x in df['jd'] if x > mid[-4] and x < mid[-5]]
            lc6 = [x for x in df['jd'] if x > mid[-5] and x < mid[-6]]
            lc7 = [x for x in df['jd'] if x > mid[-6]]
            lc8 = []
            lc9 = []
            lc10 = []
            lc11 = []
            lc12 = []

        elif len (mid) <= 7:
            lc1 = [x for x in df['jd'] if x < mid[-1]]
            lc2 = [x for x in df['jd'] if x > mid[-1] and x < mid[-2]]
            lc3 = [x for x in df['jd'] if x > mid[-2] and x < mid[-3]]
            lc4 = [x for x in df['jd'] if x > mid[-3] and x < mid[-4]]
            lc5 = [x for x in df['jd'] if x > mid[-4] and x < mid[-5]]
            lc6 = [x for x in df['jd'] if x > mid[-5] and x < mid[-6]]
            lc7 = [x for x in df['jd'] if x > mid[-6] and x < mid[-7]]
            lc8 = [x for x in df['jd'] if x > mid[-7]]
            lc9 = []
            lc10 = []
            lc11 = []
            lc12 = []

        elif len (mid) <= 8:
            lc1 = [x for x in df['jd'] if x < mid[-1]]
            lc2 = [x for x in df['jd'] if x > mid[-1] and x < mid[-2]]
            lc3 = [x for x in df['jd'] if x > mid[-2] and x < mid[-3]]
            lc4 = [x for x in df['jd'] if x > mid[-3] and x < mid[-4]]
            lc5 = [x for x in df['jd'] if x > mid[-4] and x < mid[-5]]
            lc6 = [x for x in df['jd'] if x > mid[-5] and x < mid[-6]]
            lc7 = [x for x in df['jd'] if x > mid[-6] and x < mid[-7]]
            lc8 = [x for x in df['jd'] if x > mid[-7] and x < mid[-8]]
            lc9 = [x for x in df['jd'] if x > mid[-8]]
            lc10 = []
            lc11 = []
            lc12 = []
            
        elif len(mid) <= 9:
            lc1 = [x for x in df['jd'] if x < mid[-1]]
            lc2 = [x for x in df['jd'] if x > mid[-1] and x < mid[-2]]
            lc3 = [x for x in df['jd'] if x > mid[-2] and x < mid[-3]]
            lc4 = [x for x in df['jd'] if x > mid[-3] and x < mid[-4]]
            lc5 = [x for x in df['jd'] if x > mid[-4] and x < mid[-5]]
            lc6 = [x for x in df['jd'] if x > mid[-5] and x < mid[-6]]
            lc7 = [x for x in df['jd'] if x > mid[-6] and x < mid[-7]]
            lc8 = [x for x in df['jd'] if x > mid[-7] and x < mid[-8]]
            lc9 = [x for x in df['jd'] if x > mid[-8] and x < mid[-9]]
            lc10 = [x for x in df['jd'] if x > mid[-9]] 
            lc11 = []
            lc12 = []

        elif len(mid) <= 10:
            lc1 = [x for x in df['jd'] if x < mid[-1]]
            lc2 = [x for x in df['jd'] if x > mid[-1] and x < mid[-2]]
            lc3 = [x for x in df['jd'] if x > mid[-2] and x < mid[-3]]
            lc4 = [x for x in df['jd'] if x > mid[-3] and x < mid[-4]]
            lc5 = [x for x in df['jd'] if x > mid[-4] and x < mid[-5]]
            lc6 = [x for x in df['jd'] if x > mid[-5] and x < mid[-6]]
            lc7 = [x for x in df['jd'] if x > mid[-6] and x < mid[-7]]
            lc8 = [x for x in df['jd'] if x > mid[-7] and x < mid[-8]]
            lc9 = [x for x in df['jd'] if x > mid[-8] and x < mid[-9]]
            lc10 = [x for x in df['jd'] if x > mid[-9] and x < mid[-10]]
            lc11 = [x for x in df['jd'] if x > mid[-10]]
            lc12 = []

        #print(len(lc1),len(lc2),len(lc3),len(lc4),len(lc5),len(lc6),len(lc7),len(lc8),len(lc9),len(lc10),len(lc11),len(lc12))

        #breakpoint()
        
        if (len(lc2) == 0 and len(lc3) == 0 and len(lc4) == 0 and len(lc5) > 0):
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df5 = pd.read_table(path, sep="\s+", names=column_names)
            df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 > max(lc5)].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 < min(lc5)].index, inplace = True)

            df6 = pd.read_table(path, sep="\s+", names=column_names)
            df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 > max(lc6)].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 < min(lc6)].index, inplace = True)

            df7 = pd.read_table(path, sep="\s+", names=column_names)
            df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 > max(lc7)].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 < min(lc7)].index, inplace = True)

            df8 = pd.read_table(path, sep="\s+", names=column_names)
            df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 > max(lc8)].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 < min(lc8)].index, inplace = True)

            df9 = pd.read_table(path, sep="\s+", names=column_names)
            df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
            df9.drop(df9[df9['jd']+ 2450000 > max(lc9)].index, inplace = True)
            df9.drop(df9[df9['jd']+ 2450000 < min(lc9)].index, inplace = True)

            df10 = pd.read_table(path, sep="\s+", names=column_names)
            df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
        #         df9.drop(df9[df9['jd'] > max(lc9)].index, inplace = True)
            df10.drop(df10[df10['jd']+ 2450000 < min(lc9)].index, inplace = True)
            
            meds = [np.median(df1.mag) ,np.median(df5.mag),
            np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
            
            indexes = np.array([1,5,6,7,8,9,10])
            
        elif(len(lc1) > 0 and len(lc2) == 0 and len(lc3) == 0 and len(lc4) > 0 and len(lc5) > 0 and len(lc9) == 0 and len(lc10) == 0):
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df4 = pd.read_table(path, sep="\s+", names=column_names)
            df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 > max(lc4)].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 < min(lc4)].index, inplace = True)

            df5 = pd.read_table(path, sep="\s+", names=column_names)
            df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 > max(lc5)].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 < min(lc5)].index, inplace = True)

            df6 = pd.read_table(path, sep="\s+", names=column_names)
            df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 > max(lc6)].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 < min(lc6)].index, inplace = True)

            df7 = pd.read_table(path, sep="\s+", names=column_names)
            df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 > max(lc7)].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 < min(lc7)].index, inplace = True)

            df8 = pd.read_table(path, sep="\s+", names=column_names)
            df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
            #   df8.drop(df8[df8['jd'] > max(lc8)].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 < min(lc8)].index, inplace = True)

            meds = [np.median(df1.mag),
            np.median(df4.mag), np.median(df5.mag), np.median(df6.mag), np.median(df7.mag), np.median(df8.mag)]

            meds_err = [mad_std(df1.mag),
            mad_std(df4.mag), mad_std(df5.mag), mad_std(df6.mag), mad_std(df7.mag), mad_std(df8.mag)]

            indexes = np.array([1,4,5,6,7,8])

        elif (len(lc1) >  0 and len(lc2) == 0 and len(lc3) == 0 and len(lc9) == 0 and len(lc8) > 0):
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df4 = pd.read_table(path, sep="\s+", names=column_names)
            df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 > max(lc4)].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 < min(lc4)].index, inplace = True)

            df5 = pd.read_table(path, sep="\s+", names=column_names)
            df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 > max(lc5)].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 < min(lc5)].index, inplace = True)

            df6 = pd.read_table(path, sep="\s+", names=column_names)
            df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 > max(lc6)].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 < min(lc6)].index, inplace = True)

            df7 = pd.read_table(path, sep="\s+", names=column_names)
            df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 > max(lc7)].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 < min(lc7)].index, inplace = True)

            df8 = pd.read_table(path, sep="\s+", names=column_names)
            df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
        #         df9.drop(df9[df9['jd'] > max(lc9)].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 < min(lc8)].index, inplace = True)

            meds = [np.median(df1.mag),
            np.median(df4.mag), np.median(df5.mag), np.median(df6.mag), np.median(df7.mag), np.median(df8.mag)]

            indexes = np.array([1,4,5,6,7,8])


        elif (len(lc1)>0 and len(lc2) == 0 and len(lc3) == 0 and len(lc4) == 0 and len(lc5) == 0 and len(lc6) > 0):
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df6 = pd.read_table(path, sep="\s+", names=column_names)
            df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 > max(lc6)].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 < min(lc6)].index, inplace = True)

            df7 = pd.read_table(path, sep="\s+", names=column_names)
            df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 > max(lc7)].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 < min(lc7)].index, inplace = True)

            df8 = pd.read_table(path, sep="\s+", names=column_names)
            df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 > max(lc8)].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 < min(lc8)].index, inplace = True)

            df9 = pd.read_table(path, sep="\s+", names=column_names)
            df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
            df9.drop(df9[df9['jd']+ 2450000 > max(lc9)].index, inplace = True)
            df9.drop(df9[df9['jd']+ 2450000 < min(lc9)].index, inplace = True)

            df10 = pd.read_table(path, sep="\s+", names=column_names)
            df10.drop(df8[df8['good/bad'] < 1].index, inplace = True)
            #df8.drop(df8[df8['jd'] > max(lc8)].index, inplace = True)
            df10.drop(df10[df10['jd']+ 2450000 < min(lc9)].index, inplace = True)

            meds = [np.median(df1.mag),
            np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]

            indexes = np.array([1,6,7,8,9,10])



        elif (len(lc2) == 0 and len(lc9) == 0 and len(lc10) == 0 and len(lc8) > 0):
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] <1].index, inplace=True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace=True)

            df3 = pd.read_table(path, sep="\s+", names=column_names)
            df3.drop(df1[df1['good/bad'] < 1].index, inplace=True)
            df3.drop(df3[df3['jd']+ 2450000 > max(lc3)].index, inplace=True)
            df3.drop(df3[df3['jd']+ 2450000 < min(lc3)].index, inplace=True)

            df4 = pd.read_table(path, sep="\s+", names=column_names)
            df4.drop(df4[df4['good/bad'] < 1].index, inplace=True)
            df4.drop(df4[df4['jd']+ 2450000 > max(lc4)].index, inplace=True)
            df4.drop(df4[df4['jd']+ 2450000 < min(lc4)].index, inplace=True)

            df5 = pd.read_table(path, sep="\s+", names=column_names)
            df5.drop(df5[df5['good/bad'] <1 ].index, inplace=True)
            df5.drop(df5[df5['jd']+ 2450000 > max(lc5)].index, inplace=True)
            df5.drop(df5[df5['jd']+ 2450000 < min(lc5)].index, inplace=True)

            df6 = pd.read_table(path, sep="\s+", names=column_names)
            df6.drop(df6[df6['good/bad'] < 1].index, inplace=True)
            df6.drop(df6[df6['jd']+ 2450000 > max(lc6)].index, inplace=True)
            df6.drop(df6[df6['jd']+ 2450000 < min (lc6)].index, inplace=True)

            df7 = pd.read_table(path, sep="\s+", names=column_names)
            df7.drop(df7[df7['good/bad'] <1].index, inplace=True)
            df7.drop(df7[df7['jd']+ 2450000 > max(lc7)].index, inplace=True)
            df7.drop(df7[df7['jd']+ 2450000 < min (lc7)].index, inplace=True)

            df8 = pd.read_table(path, sep="\s+", names=column_names)
            df8.drop(df8[df8['good/bad'] < 1].index, inplace=True)
            df8.drop(df8[df8['jd']+ 2450000 < min(lc8)].index, inplace=True)

            meds = [np.median(df1.mag), np.median(df3.mag), np.median(df4.mag), np.median(df5.mag), np.median(df6.mag), np.median(df7.mag), np.median(df8.mag)]
            indexes = np.array([1,3,4,5,6,7,8])
            
        elif (len(lc1) > 0 and len(lc2) == 0 and len(lc3) > 0 and len(lc8) == 0):
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df3 = pd.read_table(path, sep="\s+", names=column_names)
            df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 > max(lc3)].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 < min(lc3)].index, inplace = True)

            df4 = pd.read_table(path, sep="\s+", names=column_names)
            df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 > max(lc4)].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 < min(lc4)].index, inplace = True)

            df5 = pd.read_table(path, sep="\s+", names=column_names)
            df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 > max(lc5)].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 < min(lc5)].index, inplace = True)

            df6 = pd.read_table(path, sep="\s+", names=column_names)
            df6.drop(df4[df6['good/bad'] < 1].index, inplace = True)
            df6.drop(df4[df6['jd']+ 2450000 > max(lc6)].index, inplace = True)
            df6.drop(df4[df6['jd']+ 2450000 < min(lc6)].index, inplace = True)

            df7 = pd.read_table(path, sep="\s+", names=column_names)
            df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 < min(lc7)].index, inplace = True)

            meds = [np.median(df1.mag) ,np.median(df3.mag),
            np.median(df4.mag), np.median(df5.mag), np.median(df6.mag), np.median(df7.mag)]
            
            indexes = np.array([1,3,4,5,6,7])

        elif (len(lc3) == 0 and len(lc4) == 0 and len(lc5) == 0 and len(lc6) > 0):
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df2 = pd.read_table(path, sep="\s+", names=column_names)
            df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 > max(lc2)].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 < min(lc2)].index, inplace = True)

            df6 = pd.read_table(path, sep="\s+", names=column_names)
            df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 > max(lc6)].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 < min(lc6)].index, inplace = True)

            df7 = pd.read_table(path, sep="\s+", names=column_names)
            df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 > max(lc7)].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 < min(lc7)].index, inplace = True)

            df8 = pd.read_table(path, sep="\s+", names=column_names)
            df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 > max(lc8)].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 < min(lc8)].index, inplace = True)

            df9 = pd.read_table(path, sep="\s+", names=column_names)
            df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
            df9.drop(df9[df9['jd']+ 2450000 > max(lc9)].index, inplace = True)
            df9.drop(df9[df9['jd']+ 2450000 < min(lc9)].index, inplace = True)

            df10 = pd.read_table(path, sep="\s+", names=column_names)
            df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
        #         df9.drop(df9[df9['jd'] > max(lc9)].index, inplace = Tre)
            df10.drop(df10[df10['jd']+ 2450000 < min(lc9)].index, inplace = True)
            
            meds = [np.median(df1.mag) ,np.median(df5.mag),
            np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
            
            indexes = np.array([1,5,6,7,8,9,10])

        elif (len(lc2) == 0 and len(lc3) == 0 and len(lc4) > 0):
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df4 = pd.read_table(path, sep="\s+", names=column_names)
            df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 > max(lc4)].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 < min(lc4)].index, inplace = True)

            df5 = pd.read_table(path, sep="\s+", names=column_names)
            df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 > max(lc5)].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 < min(lc5)].index, inplace = True)

            df6 = pd.read_table(path, sep="\s+", names=column_names)
            df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 > max(lc6)].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 < min(lc6)].index, inplace = True)

            df7 = pd.read_table(path, sep="\s+", names=column_names)
            df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 > max(lc7)].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 < min(lc7)].index, inplace = True)

            df8 = pd.read_table(path, sep="\s+", names=column_names)
            df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 > max(lc8)].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 < min(lc8)].index, inplace = True)

            df9 = pd.read_table(path, sep="\s+", names=column_names)
            df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
            df9.drop(df9[df9['jd']+ 2450000 > max(lc9)].index, inplace = True)
            df9.drop(df9[df9['jd']+ 2450000 < min(lc9)].index, inplace = True)

            df10 = pd.read_table(path, sep="\s+", names=column_names)
            df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
        #         df9.drop(df9[df9['jd'] > max(lc9)].index, inplace = True)
            df10.drop(df10[df10['jd']+ 2450000 < min(lc9)].index, inplace = True)
            
            meds = [np.median(df1.mag),np.median(df4.mag),np.median(df5.mag),
            np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
            
            indexes = np.array([1,4,5,6,7,8,9,10])


        elif (len(lc3) == 0 and len(lc4) == 0 and len(lc5) > 0):
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df2 = pd.read_table(path, sep="\s+", names=column_names)
            df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 > max(lc2)].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 < min(lc2)].index, inplace = True)

            df5 = pd.read_table(path, sep="\s+", names=column_names)
            df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 > max(lc5)].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 < min(lc5)].index, inplace = True)

            df6 = pd.read_table(path, sep="\s+", names=column_names)
            df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 > max(lc6)].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 < min(lc6)].index, inplace = True)

            df7 = pd.read_table(path, sep="\s+", names=column_names)
            df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 > max(lc7)].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 < min(lc7)].index, inplace = True)

            df8 = pd.read_table(path, sep="\s+", names=column_names)
            df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 > max(lc8)].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 < min(lc8)].index, inplace = True)

            df9 = pd.read_table(path, sep="\s+", names=column_names)
            df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
            df9.drop(df9[df9['jd']+ 2450000 > max(lc9)].index, inplace = True)
            df9.drop(df9[df9['jd']+ 2450000 < min(lc9)].index, inplace = True)

            df10 = pd.read_table(path, sep="\s+", names=column_names)
            df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
        #         df9.drop(df9[df9['jd'] > max(lc9)].index, inplace = True)
            df10.drop(df10[df10['jd']+ 2450000 < min(lc9)].index, inplace = True)
            
            meds = [np.median(df1.mag), np.median(df2.mag),np.median(df5.mag),
            np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
            
            indexes = np.array([1,2,5,6,7,8,9,10])

        elif (len(lc3) == 0 and len(lc4) > 0):
            try:
                print(Target)
                df1 = pd.read_table(path, sep="\s+", names=column_names)
                df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
                df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

                df2 = pd.read_table(path, sep="\s+", names=column_names)
                df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
                df2.drop(df2[df2['jd']+ 2450000 > max(lc2)].index, inplace = True)
                df2.drop(df2[df2['jd']+ 2450000 < min(lc2)].index, inplace = True)

                df4 = pd.read_table(path, sep="\s+", names=column_names)
                df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
                df4.drop(df4[df4['jd']+ 2450000 > max(lc4)].index, inplace = True)
                df4.drop(df4[df4['jd']+ 2450000 < min(lc4)].index, inplace = True)

                df5 = pd.read_table(path, sep="\s+", names=column_names)
                df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
                df5.drop(df5[df5['jd']+ 2450000 > max(lc5)].index, inplace = True)
                df5.drop(df5[df5['jd']+ 2450000 < min(lc5)].index, inplace = True)

                df6 = pd.read_table(path, sep="\s+", names=column_names)
                df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
                df6.drop(df6[df6['jd']+ 2450000 > max(lc6)].index, inplace = True)
                df6.drop(df6[df6['jd']+ 2450000 < min(lc6)].index, inplace = True)

                df7 = pd.read_table(path, sep="\s+", names=column_names)
                df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
                df7.drop(df7[df7['jd']+ 2450000 > max(lc7)].index, inplace = True)
                df7.drop(df7[df7['jd']+ 2450000 < min(lc7)].index, inplace = True)

                df8 = pd.read_table(path, sep="\s+", names=column_names)
                df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
                df8.drop(df8[df8['jd']+ 2450000 > max(lc8)].index, inplace = True)
                df8.drop(df8[df8['jd']+ 2450000 < min(lc8)].index, inplace = True)

                df9 = pd.read_table(path, sep="\s+", names=column_names)
                df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
                df9.drop(df9[df9['jd']+ 2450000 > max(lc9)].index, inplace = True)
                df9.drop(df9[df9['jd']+ 2450000 < min(lc9)].index, inplace = True)

                df10 = pd.read_table(path, sep="\s+", names=column_names)
                df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
        #         df9.drop(df9[df9['jd'] > max(lc9)].index, inplace = True)
                df10.drop(df10[df10['jd']+ 2450000 < min(lc9)].index, inplace = True)
            
                meds = [np.median(df1.mag), np.median(df2.mag),np.median(df4.mag),np.median(df5.mag),
                np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
            
                indexes = np.array([1,2,4,5,6,7,8,9,10])
            except (ValueError):
                continue

        elif (len(lc2) == 0 and len(lc3) > 0):
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df3 = pd.read_table(path, sep="\s+", names=column_names)
            df3.drop(df2[df2['good/bad'] < 1].index, inplace = True)
            df3.drop(df2[df2['jd']+ 2450000 > max(lc3)].index, inplace = True)
            df3.drop(df2[df2['jd']+ 2450000 < min(lc3)].index, inplace = True)

            df4 = pd.read_table(path, sep="\s+", names=column_names)
            df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 > max(lc4)].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 < min(lc4)].index, inplace = True)

            df5 = pd.read_table(path, sep="\s+", names=column_names)
            df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 > max(lc5)].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 < min(lc5)].index, inplace = True)

            df6 = pd.read_table(path, sep="\s+", names=column_names)
            df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 > max(lc6)].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 < min(lc6)].index, inplace = True)

            df7 = pd.read_table(path, sep="\s+", names=column_names)
            df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 > max(lc7)].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 < min(lc7)].index, inplace = True)

            df8 = pd.read_table(path, sep="\s+", names=column_names)
            df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 > max(lc8)].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 < min(lc8)].index, inplace = True)

            df9 = pd.read_table(path, sep="\s+", names=column_names)
            df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
            df9.drop(df9[df9['jd']+ 2450000 > max(lc9)].index, inplace = True)
            df9.drop(df9[df9['jd']+ 2450000 < min(lc9)].index, inplace = True)

            df10 = pd.read_table(path, sep="\s+", names=column_names)
            df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
        #         df9.drop(df9[df9['jd'] > max(lc9)].index, inplace = True)
            df10.drop(df10[df10['jd']+ 2450000 < min(lc9)].index, inplace = True)
            
            meds = [np.median(df1.mag), np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
            np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]
            
            indexes = np.array([1,3,4,5,6,7,8,9,10])

        elif len(lc4) < 1:
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df2 = pd.read_table(path, sep="\s+", names=column_names)
            df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 > max(lc2)].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 < min(lc2)].index, inplace = True)

            df3 = pd.read_table(path, sep="\s+", names=column_names)
            df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
        #     df3.drop(df3[df3['jd'] > max(lc3)].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 < min(lc3)].index, inplace = True)

            meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag)]

            indexes = np.array([1,2,3])

        elif len(lc5) < 1:
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df2 = pd.read_table(path, sep="\s+", names=column_names)
            df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 > max(lc2)].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 < min(lc2)].index, inplace = True)

            df3 = pd.read_table(path, sep="\s+", names=column_names)
            df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 > max(lc3)].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 < min(lc3)].index, inplace = True)

            df4 = pd.read_table(path, sep="\s+", names=column_names)
            df4.drop(df3[df3['good/bad'] < 1].index, inplace = True)
        #     df3.drop(df3[df3['jd'] > max(lc3)].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 < min(lc4)].index, inplace = True)

            meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag)]
            
            indexes = np.array([1,2,3,4])

        elif len(lc6) < 1:
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df2 = pd.read_table(path, sep="\s+", names=column_names)
            df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 > max(lc2)].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 < min(lc2)].index, inplace = True)

            df3 = pd.read_table(path, sep="\s+", names=column_names)
            df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 > max(lc3)].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 < min(lc3)].index, inplace = True)

            df4 = pd.read_table(path, sep="\s+", names=column_names)
            df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 > max(lc4)].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 < min(lc4)].index, inplace = True)

            df5 = pd.read_table(path, sep="\s+", names=column_names)
            df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
        #    df5.drop(df5[df5['jd'] > max(lc5)].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 < min(lc5)].index, inplace = True)

            meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag)]

            indexes = np.array([1,2,3,4,5])

        elif len(lc7) < 1:
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df2 = pd.read_table(path, sep="\s+", names=column_names)
            df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 > max(lc2)].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 < min(lc2)].index, inplace = True)

            df3 = pd.read_table(path, sep="\s+", names=column_names)
            df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 > max(lc3)].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 < min(lc3)].index, inplace = True)

            df4 = pd.read_table(path, sep="\s+", names=column_names)
            df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 > max(lc4)].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 < min(lc4)].index, inplace = True)

            df5 = pd.read_table(path, sep="\s+", names=column_names)
            df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 > max(lc5)].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 < min(lc5)].index, inplace = True)

            df6 = pd.read_table(path, sep="\s+", names=column_names)
            df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
        #         df6.drop(df6[df6['jd'] > max(lc6)].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 < min(lc6)].index, inplace = True)

            meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                    np.median(df6.mag)]

            indexes = np.array([1,2,3,4,5,6])

        elif len(lc8) < 1:

            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df2 = pd.read_table(path, sep="\s+", names=column_names)
            df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 > max(lc2)].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 < min(lc2)].index, inplace = True)

            df3 = pd.read_table(path, sep="\s+", names=column_names)
            df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 > max(lc3)].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 < min(lc3)].index, inplace = True)

            df4 = pd.read_table(path, sep="\s+", names=column_names)
            df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 > max(lc4)].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 < min(lc4)].index, inplace = True)

            df5 = pd.read_table(path, sep="\s+", names=column_names)
            df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 > max(lc5)].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 < min(lc5)].index, inplace = True)

            df6 = pd.read_table(path, sep="\s+", names=column_names)
            df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 > max(lc6)].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 < min(lc6)].index, inplace = True)

            df7 = pd.read_table(path, sep="\s+", names=column_names)
            df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
        #         df6.drop(df6[df6['jd'] > max(lc6)].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 < min(lc7)].index, inplace = True)

            meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                    np.median(df6.mag), np.median(df7.mag)]

            indexes = np.array([1,2,3,4,5,6,7])

        elif len(lc9) < 1:
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df2 = pd.read_table(path, sep="\s+", names=column_names)
            df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 > max(lc2)].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 < min(lc2)].index, inplace = True)

            df3 = pd.read_table(path, sep="\s+", names=column_names)
            df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 > max(lc3)].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 < min(lc3)].index, inplace = True)

            df4 = pd.read_table(path, sep="\s+", names=column_names)
            df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 > max(lc4)].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 < min(lc4)].index, inplace = True)

            df5 = pd.read_table(path, sep="\s+", names=column_names)
            df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 > max(lc5)].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 < min(lc5)].index, inplace = True)

            df6 = pd.read_table(path, sep="\s+", names=column_names)
            df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 > max(lc6)].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 < min(lc6)].index, inplace = True)

            df7 = pd.read_table(path, sep="\s+", names=column_names)
            df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 > max(lc7)].index, inplace = True)                       
            df7.drop(df7[df7['jd']+ 2450000 < min(lc7)].index, inplace = True)

            df8 = pd.read_table(path, sep="\s+", names=column_names)
            df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
        #         df6.drop(df6[df6['jd'] > max(lc6)].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 < min(lc8)].index, inplace = True)

            meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                    np.median(df6.mag), np.median(df7.mag), np.median(df8.mag)]

            indexes = np.array([1,2,3,4,5,6,7,8])

        elif len(lc10) <= 1:
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df2 = pd.read_table(path, sep="\s+", names=column_names)
            df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 > max(lc2)].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 < min(lc2)].index, inplace = True)

            df3 = pd.read_table(path, sep="\s+", names=column_names)
            df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 > max(lc3)].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 < min(lc3)].index, inplace = True)

            df4 = pd.read_table(path, sep="\s+", names=column_names)
            df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 > max(lc4)].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 < min(lc4)].index, inplace = True)

            df5 = pd.read_table(path, sep="\s+", names=column_names)
            df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 > max(lc5)].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 < min(lc5)].index, inplace = True)

            df6 = pd.read_table(path, sep="\s+", names=column_names)
            df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 > max(lc6)].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 < min(lc6)].index, inplace = True)

            df7 = pd.read_table(path, sep="\s+", names=column_names)
            df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 > max(lc7)].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 < min(lc7)].index, inplace = True)

            df8 = pd.read_table(path, sep="\s+", names=column_names)
            df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 > max(lc8)].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 < min(lc8)].index, inplace = True)

            df9 = pd.read_table(path, sep="\s+", names=column_names)
            df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
        #       
            df9.drop(df9[df9['jd']+ 2450000 < min(lc9)].index, inplace = True)

            meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                    np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag)]

            indexes = np.array([1,2,3,4,5,6,7,8,9])

        elif len(lc11) <= 1:
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df2 = pd.read_table(path, sep="\s+", names=column_names)
            df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 > max(lc2)].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 < min(lc2)].index, inplace = True)

            df3 = pd.read_table(path, sep="\s+", names=column_names)
            df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 > max(lc3)].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 < min(lc3)].index, inplace = True)

            df4 = pd.read_table(path, sep="\s+", names=column_names)
            df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 > max(lc4)].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 < min(lc4)].index, inplace = True)

            df5 = pd.read_table(path, sep="\s+", names=column_names)
            df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 > max(lc5)].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 < min(lc5)].index, inplace = True)

            df6 = pd.read_table(path, sep="\s+", names=column_names)
            df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 > max(lc6)].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 < min(lc6)].index, inplace = True)

            df7 = pd.read_table(path, sep="\s+", names=column_names)
            df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 > max(lc7)].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 < min(lc7)].index, inplace = True)

            df8 = pd.read_table(path, sep="\s+", names=column_names)
            df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 > max(lc8)].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 < min(lc8)].index, inplace = True)

            df9 = pd.read_table(path, sep="\s+", names=column_names)
            df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
            df9.drop(df9[df9['jd']+ 2450000 > max(lc9)].index, inplace = True)    
            df9.drop(df9[df9['jd']+ 2450000 < min(lc9)].index, inplace = True)

            df10 = pd.read_table(path, sep="\s+", names=column_names)
            df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
            df10.drop(df10[df10['jd']+ 2450000 < min(lc10)].index, inplace = True)


            meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                    np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag)]

            indexes = np.array([1,2,3,4,5,6,7,8,9,10])

        elif len(lc12) <= 1:
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df2 = pd.read_table(path, sep="\s+", names=column_names)
            df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 > max(lc2)].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 < min(lc2)].index, inplace = True)

            df3 = pd.read_table(path, sep="\s+", names=column_names)
            df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 > max(lc3)].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 < min(lc3)].index, inplace = True)

            df4 = pd.read_table(path, sep="\s+", names=column_names)
            df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 > max(lc4)].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 < min(lc4)].index, inplace = True)

            df5 = pd.read_table(path, sep="\s+", names=column_names)
            df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 > max(lc5)].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 < min(lc5)].index, inplace = True)

            df6 = pd.read_table(path, sep="\s+", names=column_names)
            df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 > max(lc6)].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 < min(lc6)].index, inplace = True)

            df7 = pd.read_table(path, sep="\s+", names=column_names)
            df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 > max(lc7)].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 < min(lc7)].index, inplace = True)

            df8 = pd.read_table(path, sep="\s+", names=column_names)
            df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 > max(lc8)].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 < min(lc8)].index, inplace = True)

            df9 = pd.read_table(path, sep="\s+", names=column_names)
            df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
            df9.drop(df9[df9['jd']+ 2450000 > max(lc9)].index, inplace = True)
            df9.drop(df9[df9['jd']+ 2450000 < min(lc9)].index, inplace = True)

            df10 = pd.read_table(path, sep="\s+", names=column_names)
            df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
            df10.drop(df10[df10['jd']+ 2450000 > max(lc10)].index, inplace = True)
            df10.drop(df10[df10['jd']+ 2450000 < min(lc10)].index, inplace = True)

            df11 = pd.read_table(path, sep="\s+", names=column_names)
            df11.drop(df11[df11['good/bad'] < 1].index, inplace = True)
            df11.drop(df11[df11['jd']+ 2450000 < min(lc11)].index, inplace = True)


            meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                    np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag), np.median(df11.mag)]

            indexes = np.array([1,2,3,4,5,6,7,8,9,10,11])


        elif len(lc12) > 1:
            df1 = pd.read_table(path, sep="\s+", names=column_names)
            df1.drop(df1[df1['good/bad'] < 1].index, inplace = True)
            df1.drop(df1[df1['jd']+ 2450000 > max(lc1)].index, inplace = True)

            df2 = pd.read_table(path, sep="\s+", names=column_names)
            df2.drop(df2[df2['good/bad'] < 1].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 > max(lc2)].index, inplace = True)
            df2.drop(df2[df2['jd']+ 2450000 < min(lc2)].index, inplace = True)

            df3 = pd.read_table(path, sep="\s+", names=column_names)
            df3.drop(df3[df3['good/bad'] < 1].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 > max(lc3)].index, inplace = True)
            df3.drop(df3[df3['jd']+ 2450000 < min(lc3)].index, inplace = True)

            df4 = pd.read_table(path, sep="\s+", names=column_names)
            df4.drop(df4[df4['good/bad'] < 1].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 > max(lc4)].index, inplace = True)
            df4.drop(df4[df4['jd']+ 2450000 < min(lc4)].index, inplace = True)

            df5 = pd.read_table(path, sep="\s+", names=column_names)
            df5.drop(df5[df5['good/bad'] < 1].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 > max(lc5)].index, inplace = True)
            df5.drop(df5[df5['jd']+ 2450000 < min(lc5)].index, inplace = True)
            df6 = pd.read_table(path, sep="\s+", names=column_names)
            df6.drop(df6[df6['good/bad'] < 1].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 > max(lc6)].index, inplace = True)
            df6.drop(df6[df6['jd']+ 2450000 < min(lc6)].index, inplace = True)

            df7 = pd.read_table(path, sep="\s+", names=column_names)
            df7.drop(df7[df7['good/bad'] < 1].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 > max(lc7)].index, inplace = True)
            df7.drop(df7[df7['jd']+ 2450000 < min(lc7)].index, inplace = True)

            df8 = pd.read_table(path, sep="\s+", names=column_names)
            df8.drop(df8[df8['good/bad'] < 1].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 > max(lc8)].index, inplace = True)
            df8.drop(df8[df8['jd']+ 2450000 < min(lc8)].index, inplace = True)

            df9 = pd.read_table(path, sep="\s+", names=column_names)
            df9.drop(df9[df9['good/bad'] < 1].index, inplace = True)
            df9.drop(df9[df9['jd']+ 2450000 > max(lc9)].index, inplace = True)
            df9.drop(df9[df9['jd']+ 2450000 < min(lc9)].index, inplace = True)

            df10 = pd.read_table(path, sep="\s+", names=column_names)
            df10.drop(df10[df10['good/bad'] < 1].index, inplace = True)
            df10.drop(df10[df10['jd']+ 2450000 > max(lc10)].index, inplace = True)
            df10.drop(df10[df10['jd']+ 2450000 < min(lc10)].index, inplace = True)

            df11 = pd.read_table(path, sep="\s+", names=column_names)
            df11.drop(df11[df11['good/bad'] < 1].index, inplace = True)
            df11.drop(df11[df11['jd']+ 2450000 > max(lc11)].index, inplace = True)
            df11.drop(df11[df11['jd']+ 2450000 < min(lc11)].index, inplace = True)

            df12 = pd.read_table(path, sep="\s+", names=column_names)
            df12.drop(df12[df12['good/bad'] < 1].index, inplace = True)
            df12.drop(df12[df12['jd']+ 2450000 < min(lc12)].index, inplace = True)


            meds = [np.median(df1.mag), np.median(df2.mag),np.median(df3.mag),np.median(df4.mag),np.median(df5.mag),
                    np.median(df6.mag), np.median(df7.mag), np.median(df8.mag), np.median(df9.mag), np.median(df10.mag), np.median(df11.mag), np.median(df12.mag)]

            indexes = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

        coeffs1 = np.polyfit(indexes, meds, 1)
        slope = coeffs1[0]

        N = len(meds)

        start = meds[0]
        end = meds[-1]

        #polyfit

        degree = 2

        coeffs = np.polyfit(indexes, meds, degree)
        poly_function = np.poly1d(coeffs)

        # fitted data
        fitted_mag = poly_function(indexes)
        quadratic_slope = coeffs[-3]
        c1 = coeffs[-2]
        c2 = coeffs[-1]

        te = -c2/(2*quadratic_slope)
        me = c1-(c2**2)/(4*quadratic_slope)

        m0 = c1 + c2*indexes[0] + quadratic_slope*indexes[0]**2
        m1 = c1 + c2*indexes[-1] + quadratic_slope*indexes[-1]**2

        if te > indexes[0] and te < indexes[-1]:
            m1m0 = np.abs(m1-m0)
            m1me = np.abs(m1-me)
            m0me = np.abs(m0-me)

            mags = [m1m0,m1me,m0me]

            diff = max(mags)

        else:
            diff = np.abs(m1-m0)

        id.append(Target)
        median.append(lc_median)
        mag.append(p_mag)
        median_err.append(lc_mad)
        dispersion.append(lc_dispersion)
        slope.append(slope)
        quad_slope.append(quadratic_slope)
        coeff1.append(c1)
        coeff2.append(c2)
        Diff.append(diff)

    ltv_table = Table([id, mag, median, median_err, dispersion, slope, quad_slope, coeff1, coeff2, Diff],
    names=('ASAS-SN ID', 'Pstarss gmag', 'Median', 'Median_err', 'Dispersion', 'Slope', 'Quad Slope', 'coeff1', 'coeff2', 'max diff'),
    meta={'name': 'ltv_table'})

    # update column names

    csv_file_path = MAG_BIN + '/new/' + x + '.csv'

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(ltv_table)
    
    print('Ending'+x)
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))