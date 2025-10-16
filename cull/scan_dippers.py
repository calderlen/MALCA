import numpy as np
import pandas as pd
import scipy
import math
import os
import glob


from utils import naive_peak_search, read_lc_dat
from baseline import per_camera_baseline



# readd notes and todo back into the github directory


# for LCs by iterating thru indices that are in the asassn lc filtered NOT the vsx crossmatched (i think, now im sort of confused why we did the vsx crossmatch in the first place)
# you now have the fwf lc as a dataframe
# KEEP data points marked as bad
#df_baseline = per_camera_baseline(df, days=30., min_points=10)
# now given the baseline (for each data point?), subtract off the mag from that baseline
# search for the upward and downward dips within the file that you accidentally deleted, empirically set dep fraction until you recover all of the same candidates that brayden does

# compare the results of your fitting procedure to the simple one that brayden uses




df_g, df_v = read_lc_dat(asassn_id, path)

for df in df_g, df_v:
	df = per_camera_baseline(df, days=30)

# 

naive_peak_search

naive_peak_search(df, prominence=0.17, distance=25, height=0.3, width=2)
