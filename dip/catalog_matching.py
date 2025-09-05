import numpy as np
import pandas as pd
import scipy
import math
import os

from tqdm import tqdm

from astropy.timeseries import LombScargle as ls
from astropy.coordinates import SkyCoord
from astropy import units as u

from astroquery import Gaia

fname = '/home/lenhart.106/Downloads/vsxcat.090525'

columns = ["id_vsx", "name", "UNKNOWN_FLAG", "ra", "dec", "variability_class", "mag", "band_mag", "UNKNOWN_FLAG_2", "amplitude?", "amplitude/mag_diff", "band_amplitude/mag_diff_amplitude", "epoch", "period", "spectral_type"]

# variability class lists
continuous = [
    # Rotating
    "ACV","BY","CTTS/ROT","ELL","FKCOM","HB","LERI","PSR","R","ROT","RS",
    "SXARI","TTS/ROT","WTTS/ROT","NSIN ELL","ROT (TTS subtype)",
    # Pulsating
    "ACEP","ACEP(B)","ACEPS","ACYG","BCEP","BCEPS","BLAP","BXCIR","CEP",
    "CW","CWA","CWB","CWB(B)","CWBS","DCEP","DCEP(B)","DCEPS","DCEPS(B)",
    "DSCT","DSCTC","DWLYN","GDOR","HADS","HADS(B)","L","LB","LC","M","ORG",
    "PPN","PVTEL","PVTELI","PVTELII","PVTELIII","roAm","roAp","RR","RRAB",
    "RRC","RRD","RV","RVA","RVB","SPB","SPBe","SR","SRA","SRB","SRC","SRD",
    "SRS","SXPHE","SXPHE(B)","V361HYA","V1093HER","ZZ","ZZA","ZZB","ZZ/GWLIB",
    "ZZO","ZZLep","LPV","CW-FO","CW-FU","DCEP-FO","DCEP-FU","DSCTr","PULS",
    "(B)","BL","GWLIB",
    # Cataclysmic / X-ray
    "WDP","XP",
    # Other objects
    "AGN","BLLAC","QSO","VBD","APER","NSIN","PER","SIN"
]

# Non-continuous (event-dominated) variability classes
non_continuous = [
    # Eclipsing
    "E","EA","EB","E-DO","EP","EW","EC","ED","ESD","AR","BD","D","DM","DS",
    "DW","EL","GS","HW","K","KE","KW","PN","SD","WD",
    # Eruptive
    "DYPer","EXOR","FF","FUOR","GCAS","I","IA","IB","IS","ISB","UV","UVN",
    "DIP","Microlens",
    # Cataclysmic / explosive
    "AM","CBSS","CBSS/V","DQ","DQ/AE","IBWD","LFBOT","N","NA","NB","NC",
    "NL","NL/VY","NR","SN","SN I","SN Ia","SN Iax","SN Ia-00cx-like",
    "SN Ia-02es-like","SN Ia-06gz-like","SN Ia-86G-like","SN Ia-91bg-like",
    "SN Ia-91T-like","SN Ia-99aa-like","SN Ia-Ca-rich","SN Ia-CSM","SN Ib",
    "SN Ic","SN Icn","SN Ic-BL","SN Idn","SN Ien","SN II","SN IIa","SN IIb",
    "SN IId","SN II-L","SN IIn","SN II-P","SN-pec","SLSN","SLSN-I","SLSN-II",
    "UG","UGER","UGSS","UGSU","UGWZ","UGZ","UGZ/IW","V838MON",
    # X-ray bursts
    "XB","XN","GRB","Transient"
]

# Mixed variability classes (both continuous & discrete modes)
mixed = [
    # Rotating / pulsating hybrids
    "SXARI/E","PSR (binary subtype)","RS (as subtype E/ELL)",
    # Eruptive
    "cPNB[e]","CTTS","DPV","FSCMa","IN","INA","INAT","INB","INS","INSA",
    "INSB","INST","INT","ISA","RCB","SDOR","TTS","UXOR","WR (eruptive)",
    "YHG","ZZA/O","YSO","(YY)",
    # Cataclysmic / symbiotic
    "ZAND",
    # X-ray
    "HMXB","IMXB","LMXB","X","BHXB","XJ","XPR","XBR",
    # Other
    "*","S","MISC","VAR"
]

df = pd.read_csv(fname, delim_whitespace=True, header=None, dtype=str)

def should_drop_cont_var(var_string):
    # Handle NaN
    if pd.isna(var_string):
        return False
    parts = var_string.split("|")
    return any(p in continuous for p in parts)

cont_var_mask = df["VarType"].apply(should_drop_cont_var)
df = df[~cont_var_mask]

asassn_data_columns = [
    "asas_sn_id",
    "ra_deg",
    "dec_deg",
    "refcat_id",
    "gaia_id",
    "hip_id",
    "tyc_id",
    "tmass_id",
    "sdss_id",
    "allwise_id",
    "tic_id",
    "plx",
    "plx_d",
    "pm_ra",
    "pm_ra_d",
    "pm_dec",
    "pm_dec_d",
    "gaia_mag",
    "gaia_mag_d",
    "gaia_b_mag",
    "gaia_b_mag_d",
    "gaia_r_mag",
    "gaia_r_mag_d",
    "gaia_eff_temp",
    "gaia_g_extinc",
    "gaia_var",
    "sfd_g_extinc",
    "rp_00_1",
    "rp_01",
    "rp_10",
    "pstarrs_g_mag",
    "pstarrs_g_mag_d",
    "pstarrs_g_mag_chi",
    "pstarrs_g_mag_contrib",
    "pstarrs_r_mag",
    "pstarrs_r_mag_d",
    "pstarrs_r_mag_chi",
    "pstarrs_r_mag_contrib",
    "pstarrs_i_mag",
    "pstarrs_i_mag_d",
    "pstarrs_i_mag_chi",
    "pstarrs_i_mag_contrib",
    "pstarrs_z_mag",
    "pstarrs_z_mag_d",
    "pstarrs_z_mag_chi",
    "pstarrs_z_mag_contrib",
    "nstat"
]

asassn_dir = '/data/poohbah/1/assassin/rowan.90/lcsv2'

lc_12_12_5 = asassn_dir + '/12_12.5'
lc_12_5_13 = asassn_dir + '/12.5_13'
lc_13_13_5 = asassn_dir + '/13_13.5'
lc_13_5_14 = asassn_dir + '/13.5_14'
lc_14_5_15 = asassn_dir + '/14.5_15'

test_index = lc_12_12_5 + 'index0.csv'

df_asassn_idx = pd.read_csv(test_index)
