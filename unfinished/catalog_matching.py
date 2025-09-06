import numpy as np
import pandas as pd
import scipy
import math
import os
import glob

from tqdm.auto import tqdm

from astropy.coordinates import SkyCoord
from astropy import units as u

from pathlib import Path as p

# each file's column headings

tqdm.pandas(desc="Filtering VSX by class")

vsx_columns = set(["id_vsx", "name", "UNKNOWN_FLAG", "ra", "dec", "variability_class", "mag", "band_mag", "UNKNOWN_FLAG_2", "amplitude?", "amplitude/mag_diff", "band_amplitude/mag_diff_amplitude", "epoch", "period", "spectral_type"])

asassn_columns=set(["JD","mag",'error', 'good/bad', 'camera#', 'band', 'camera name']) #1=good, 0 =bad #1=V, 0=g

asassn_index_columns = set(['asas_sn_id','ra_deg','dec_deg','refcat_id','gaia_id',  'hip_id','tyc_id','tmass_id','sdss_id','allwise_id','tic_id','plx','plx_d','pm_ra','pm_ra_d','pm_dec','pm_dec_d','gaia_mag','gaia_mag_d','gaia_b_mag','gaia_b_mag_d','gaia_r_mag','gaia_r_mag_d','gaia_eff_temp','gaia_g_extinc','gaia_var','sfd_g_extinc','rp_00_1','rp_01','rp_10','pstarrs_g_mag','pstarrs_g_mag_d','pstarrs_g_mag_chi','pstarrs_g_mag_contrib','pstarrs_r_mag','pstarrs_r_mag_d','pstarrs_r_mag_chi','pstarrs_r_mag_contrib','pstarrs_i_mag','pstarrs_i_mag_d','pstarrs_i_mag_chi','pstarrs_i_mag_contrib','pstarrs_z_mag','pstarrs_z_mag_d','pstarrs_z_mag_chi','pstarrs_z_mag_contrib','nstat'])

# loading in files
vsx_dir = '/home/lenhart.106/Downloads/vsxcat.090525'

df_vsx = pd.read_csv(vsx_dir, delim_whitespace=True, header=None, names=vsx_columns, dtype=str)

asassn_dir = '/data/poohbah/1/assassin/rowan.90/lcsv2'

lc_12_12_5 = asassn_dir + '/12_12.5'
lc_12_5_13 = asassn_dir + '/12.5_13'
lc_13_13_5 = asassn_dir + '/13_13.5'
lc_13_5_14 = asassn_dir + '/13.5_14'
lc_14_14_5 = asassn_dir + '/14_14.5'
lc_14_5_15 = asassn_dir + '/14.5_15'

dirs = [lc_12_12_5, lc_12_5_13, lc_13_13_5, lc_13_5_14, lc_14_14_5, lc_14_5_15]



# variability class filters
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


def should_drop_cont_var(var_string):
    # Handle NaN
    if pd.isna(var_string):
        return False
    parts = var_string.split("|")
    return any(p in EXCLUDE for p in parts)

cont_var_mask = df_vsx["variability_class"].progress_apply(should_drop_cont_var)
df_vsx_filt = df_vsx[~cont_var_mask].copy()


# need to check for [asassn_id].dat missing from lc[num]_cal create a new index[num]_masked.csv by filtering out all entries IN index[num].csv and NOT IN lc[num]_cal

present_ids = set()
for folder in tqdm(dirs, desc="Scanning bins"):
    files = glob.glob(f"{folder}/lc*_cal/*.dat")
    for f in tqdm(files, desc=f"Collecting IDs in {os.path.basename(folder)}", leave=False):
        present_ids.add(p(f).stem)t

def mask_index_dir(bin_dir):
    idx_paths = glob.glob(f"{bin_dir}/index[0-9]*.csv")
    for idx_path in tqdm(idx_paths, desc=f"Masking {os.path.basename(bin_dir)}", leave=False):
        df_idx = pd.read_csv(idx_path)
        id_col = "asas_sn_id"
        m = df_idx[id_col].astype(str).isin(present_ids)
        df_masked = df_idx[m].copy()
        out = p(idx_path).with_name(p(idx_path).stem + "_masked.csv")
        df_masked.to_csv(out, index=False)


for d in tqdm(dirs, desc="Creating masked index files"):
    mask_index_dir(d)

masked_files = []
for d in dirs:
    masked_files.extend(glob.glob(f"{d}/index*_masked.csv"))

dfs = []
for fpath in tqdm(masked_files, desc="Reading masked CSVs"):
    dfs.append(pd.read_csv(fpath))

df_all = pd.concat(dfs, ignore_index=True)

# extract RA and Dec from asas-sn and vsx
c_asassn = SkyCoord(ra=df_all['ra_deg'].values*u.deg, dec=df_all['dec_deg'].values*u.deg)
c_vsx  = SkyCoord(ra=df_vsx_filt["ra"].values*u.deg, dec=df_vsx_filt["dec"].values*u.deg)

# nearest neighbor in VSX for each ASAS-SN target
idx_vsx_near, sep2d_near, _ = c_asassn.match_to_catalog_sky(c_vsx)

match_radius = 3 * u.arcsec  # 3 arcsec
ok = sep2d_near < match_radius

df_match = pd.DataFrame({
    "targ_idx": np.arange(len(df_all))[ok],
    "vsx_idx":  idx_vsx_near[ok],
    "sep_arcsec": sep2d_near[ok].to(u.arcsec).value,
})

# merge metadata
out = (df_match
       .merge(df_all.reset_index(drop=True), left_on="targ_idx", right_index=True, how="left")
       .merge(df_vsx_filt.reset_index(drop=True), left_on="vsx_idx",  right_index=True, how="left",
              suffixes=("_targ","_vsx")))

keep_cols = [
    "asas_sn_id", "ra_deg", "dec_deg",      # from ASAS-SN index files
    "id_vsx", "name", "variability_class",  # from VSX
    "mag", "period", "sep_arcsec"
]

out = out[[c for c in keep_cols if c in out.columns]]

out.to_csv("asassn_x_vsx_matches.csv", index=False)

