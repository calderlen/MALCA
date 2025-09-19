import os
import glob
from pathlib import Path as p
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from astropy.coordinates import SkyCoord
from astropy import units as u
from datetime import datetime

# file paths
lc_dir = '/data/poohbah/1/assassin/rowan.90/lcsv2'
lc_dir_masked = "/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked"
vsx_file = '/data/poohbah/1/assassin/lenhart/code/calder/vsxcat.090525'

# ----- define bins once -----
MAG_BINS = ['12_12.5','12.5_13','13_13.5','13.5_14','14_14.5','14.5_15']

# LCs live here (for lc*_cal/*.dat)
lc_bins = [f"{lc_dir}/{b}" for b in MAG_BINS]

# masked index CSVs live here (for index*_masked.csv)
masked_bins = [f"{lc_dir_masked}/{b}" for b in MAG_BINS]

lc_12_12_5 = lc_dir_masked + '/12_12.5'
lc_12_5_13 = lc_dir_masked + '/12.5_13'
lc_13_13_5 = lc_dir_masked + '/13_13.5'
lc_13_5_14 = lc_dir_masked + '/13.5_14'
lc_14_14_5 = lc_dir_masked + '/14_14.5'
lc_14_5_15 = lc_dir_masked + '/14.5_15'

dirs = [lc_12_12_5, lc_12_5_13, lc_13_13_5, lc_13_5_14, lc_14_14_5, lc_14_5_15]

# each file's column headers
tqdm.pandas(desc="Filtering VSX by class")

vsx_columns = ["id_vsx", 
                "name", 
                "UNKNOWN_FLAG", 
                "ra", 
                "dec", 
                "class", 
                "mag", 
                "mag_band", 
                "amp_flag", 
                "amp", 
                "amp_band",
                "epoch", 
                "period", 
                "spectral_type"]

'''
- vsx column notes
    - first column is the vsx id, dtype should be int?
    - second column is the name of the object, which has instances of being separated by a single space
    - third column is some flag: when any survey: 0, except for NSV=1, 
'''

asassn_columns=["JD",
                "mag",
                'error', 
                'good/bad', 
                'camera#', 
                'band', 
                'camera name'] #1=good, 0 =bad #1=V, 0=g

asassn_index_columns = ['asassn_id',
                        'ra_deg',
                        'dec_deg',
                        'refcat_id',
                        'gaia_id', 
                        'hip_id',
                        'tyc_id',
                        'tmass_id',
                        'sdss_id',
                        'allwise_id',
                        'tic_id',
                        'plx',
                        'plx_d',
                        'pm_ra',
                        'pm_ra_d',
                        'pm_dec',
                        'pm_dec_d',
                        'gaia_mag',
                        'gaia_mag_d',
                        'gaia_b_mag',
                        'gaia_b_mag_d',
                        'gaia_r_mag',
                        'gaia_r_mag_d',
                        'gaia_eff_temp',
                        'gaia_g_extinc',
                        'gaia_var',
                        'sfd_g_extinc',
                        'rp_00_1',
                        'rp_01',
                        'rp_10',
                        'pstarrs_g_mag',
                        'pstarrs_g_mag_d',
                        'pstarrs_g_mag_chi',
                        'pstarrs_g_mag_contrib',
                        'pstarrs_r_mag',
                        'pstarrs_r_mag_d',
                        'pstarrs_r_mag_chi',
                        'pstarrs_r_mag_contrib',
                        'pstarrs_i_mag',
                        'pstarrs_i_mag_d',
                        'pstarrs_i_mag_chi',
                        'pstarrs_i_mag_contrib',
                        'pstarrs_z_mag',
                        'pstarrs_z_mag_d',
                        'pstarrs_z_mag_chi',
                        'pstarrs_z_mag_contrib',
                        'nstat']

df_vsx = pd.read_fwf(
    vsx_file,
    names=vsx_columns,
    header=None,
    dtype=str,          
    on_bad_lines="skip",
    colspecs="infer",
    infer_nrows=20000,
)



def parse_censored_mag(s):
    '''
    Since vsx mag columns has upper limits in some cases, we initially process the mag columns as strings. This function takes those entries and splits the mag column into two additional columns: magX_val and magX_censor, with the former containing a float and the latter a string with "lt" or "gt"
    '''
    if pd.isna(s): return pd.NA, pd.NA
    t = str(s).strip()
    if not t: return pd.NA, pd.NA
    if t[0] in "<>":
        sign = "lt" if t[0] == "<" else "gt"
        try: val = float(t[1:].strip())
        except ValueError: return pd.NA, sign
        return val, sign
    try: return float(t), pd.NA
    except ValueError: return pd.NA, pd.NA

# floats
for c in ["ra","dec","epoch","period"]:
    df_vsx[c] = pd.to_numeric(df_vsx[c], errors="coerce")

# nullable ints (only true integers survive; non-integers -> NA)
def to_nullable_int(s, dtype="Int64"):
    x = pd.to_numeric(s, errors="coerce")
    x = x.where(x.isna() | (np.floor(x) == x))
    return x.astype(dtype)

df_vsx["id_vsx"]       = to_nullable_int(df_vsx["id_vsx"], "Int64")
df_vsx["UNKNOWN_FLAG"] = to_nullable_int(df_vsx["UNKNOWN_FLAG"], "Int8")

# appending two more columns in the df_vsx that split the mag column into a float and a string "lt"/"gt" (if the latter is necessary)
df_vsx[["amp","amp_censor"]] = df_vsx["amp"].apply(parse_censored_mag).apply(pd.Series)

# vsx variability classes
EXCLUDE = set([
    # Eclipsing binaries (geometric, periodic dips)
    "E","EA","EB","E-DO","EP","EW","EC","ED","ESD", #eclipsing variables
    "AR","BD","D","DM","DS","DW","EL","GS","HW","K","KE","KW", #subtypes of eclipsing variables

    # White dwarf / compact single stars
    "PN","SD","WD",

    # Eruptive / flaring (brightenings, not dusty dimmings)
    "EXOR","FUOR","GCAS","UV","UVN","FF",
    "I","IA","IB","IS","ISB",

    # Cataclysmic variables (interacting binaries, novae, magnetic systems)
    "AM","CBSS","CBSS/V","DQ","DQ/AE","IBWD",
    "N","NA","NB","NC","NL","NL/VY","NR",

    # Supernovae and explosive transients (all spectroscopic subclasses)
    "SN","SN I","SN Ia","SN Iax",
    "SN Ia-00cx-like","SN Ia-02es-like","SN Ia-06gz-like",
    "SN Ia-86G-like","SN Ia-91bg-like","SN Ia-91T-like","SN Ia-99aa-like",
    "SN Ia-Ca-rich","SN Ia-CSM",
    "SN Ib","SN Ic","SN Icn","SN Ic-BL","SN Idn","SN Ien",
    "SN II","SN IIa","SN IIb","SN IId","SN II-L","SN IIn","SN II-P",
    "SN-pec",
    "SLSN","SLSN-I","SLSN-II",
    "V838MON","LFBOT",

    # High-energy X-ray/gamma transients
    "XB","XN","GRB","Transient",

    # Gravitational microlensing events
    "Microlens",

    # Non-stellar extragalactic sources
    "AGN","BLLAC","QSO",

    # Cataclysmic dwarf novae (subtypes of UG)
    "UG","UGER","UGSS","UGSU","UGWZ","UGZ","UGZ/IW",

    # Rotational variables (spotted stars, ellipsoidal, binaries, pulsars)
    "ACV","BY","CTTS/ROT","ELL","FKCOM","HB","LERI",
    "PSR","R","ROT","RS","SXARI","TTS/ROT","WTTS/ROT",
    "NSIN ELL","ROT (TTS subtype)",

    # Pulsating variables (radial/non-radial, classical pulsators, WDs, hot stars)
    "ACEP","ACEP(B)","ACEPS","ACYG","BCEP","BCEPS",
    "BLAP","BXCIR","CEP","CW","CWA","CWB","CWB(B)","CWBS",
    "DCEP","DCEP(B)","DCEPS","DCEPS(B)","DSCT","DSCTC",
    "DWLYN","GDOR","HADS","HADS(B)",
    "L","LB","LC","M","ORG",
    "PPN","PVTEL","PVTELI","PVTELII","PVTELIII",
    "roAm","roAp",
    "RR","RRAB","RRC","RRD",
    "RV","RVA","RVB",
    "SPB","SPBe",
    "SR","SRA","SRB","SRC","SRD","SRS",
    "SXPHE","SXPHE(B)",
    "V361HYA","V1093HER",
    "ZZ","ZZA","ZZB","ZZ/GWLIB","ZZO","ZZLep",
    "LPV","CW-FO","CW-FU","DCEP-FO","DCEP-FU","DSCTr",
    "PULS","(B)","BL","GWLIB",

    # Miscellaneous compact/X-ray systems
    "WDP","XP",

    # Survey or uncertain classifications (catch-all codes)
    "NSIN","PER","SIN","VBD",
    
    # Evolved dust-formers (deep, long dimmings)
    "RCB", "DYPer",

    # young stars; many show dipping episodes
    "YSO", "TTS", "CTTS", "WTTS",  
    # Irregular/inauspicious YSO subtypes (often dust-related dips show up here)
    "IN","INA","INB","INS","INSA","INSB","INST","INT","ISA","INAT",
])

KEEP = set([
    # Explicit
    "DIP",           # VSX “dippers”
    "UXOR",          # UX Ori-type (disk occultations in YSOs)
    # aperiodic variables
    "APER",
    # Catch-alls that often hide dippers
    "VAR","MISC","*",  # useful to *not* discard a priori
])

def filter_vsx_classes(var_string):
    '''
    screens out unwanted vsx variability classes
    '''
    # currently we are keeping any vsx entry if there is no variability class given; make note of this and adjust if necessary
    if pd.isna(var_string):
        return False
    parts = var_string.split("|")
    return any(p in EXCLUDE for p in parts)

cont_var_mask = df_vsx["class"].progress_apply(filter_vsx_classes)
df_vsx_filt = df_vsx[~cont_var_mask].copy()

# iterate over all dats in all lc_cal subdirs in all magnitude dirs, collect IDs of light curve
present_ids = set()

all_files = [f for d in lc_bins for f in glob.glob(f"{d}/lc*_cal/*.dat")]
for f in tqdm(all_files, desc="Collecting IDs", unit="file", unit_scale=True, dynamic_ncols=True):
    present_ids.add(p(f).stem)

# this probably ins't necessary to repeat since we did it once. remove after confirming
#for d in tqdm(dirs, desc="Creating masked index files"):
#    mask_index_dir(d)

# gather masked CSVs and read them
masked_files = []
for d in masked_bins:
    masked_files.extend(glob.glob(f"{d}/index*_masked.csv"))

dfs = []
for fpath in tqdm(masked_files, desc="Reading masked CSVs"):
    dfs.append(pd.read_csv(fpath, dtype={"asassn_id": "string"}))

# outputting concatenated lightcurve index csv; come back to this because I don't think it's necessary
df_all = pd.concat(dfs, ignore_index=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_csv = p.cwd() / f"asassn_index_masked_concat_{stamp}.csv"
df_all.to_csv(out_csv, index=False)

# coerce ASAS-SN coords and drop NaNs BEFORE SkyCoord so indices match search results
df_all["ra_deg"]  = pd.to_numeric(df_all["ra_deg"], errors="coerce")
df_all["dec_deg"] = pd.to_numeric(df_all["dec_deg"], errors="coerce")
df_all_clean = df_all.dropna(subset=["ra_deg","dec_deg"]).reset_index(drop=True)
df_vsx_filt_clean = df_vsx_filt.dropna(subset=["ra","dec"]).reset_index(drop=True)

# extract RA and Dec from asas-sn and vsx
c_asassn = SkyCoord(ra=df_all_clean['ra_deg'].values*u.deg,
                    dec=df_all_clean['dec_deg'].values*u.deg)
c_vsx    = SkyCoord(ra=df_vsx_filt_clean["ra"].values*u.deg,
                    dec=df_vsx_filt_clean["dec"].values*u.deg)

# nearest neighbor in VSX for each ASAS-SN target
match_radius = 3 * u.arcsec  # 3 arcsec
idx_targ, idx_vsx, sep2d, _ = c_asassn.search_around_sky(c_vsx, match_radius)

# create df with asassn index and vsx index and their separation
df_pairs = pd.DataFrame({
    "targ_idx": idx_targ,
    "vsx_idx":  idx_vsx,
    "sep_arcsec": sep2d.to(u.arcsec).value
})

# merge metadata
out = (df_pairs
       .merge(df_all_clean, left_on="targ_idx", right_index=True, how="left")
       .merge(df_vsx_filt_clean, left_on="vsx_idx", right_index=True, how="left",
              suffixes=("_targ","_vsx")))

# outputting timestamped crossmatched csv
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out2 = p.cwd() / f"asassn_x_vsx_matches_{stamp}.csv"
out.to_csv(out2, index=False)