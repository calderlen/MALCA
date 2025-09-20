import glob
import re
from pathlib import Path as p
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

# file paths
lc_dir = '/data/poohbah/1/assassin/rowan.90/lcsv2'
lc_dir_masked = "/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked"
vsx_file = '/data/poohbah/1/assassin/lenhart/code/calder/vsxcat.090525'

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


colspecs = [
    (0, 8),    # 1-8   OID
    (9, 39),   # 10-39 Name
    (40, 41),  # 41    V (variability flag)
    (42, 51),  # 43-51 RAdeg
    (52, 61),  # 53-61 DEdeg
    (62, 92),  # 63-92 Type
    (93, 94),  # 94    l_max
    (95, 102), # 96-102 max
    (103,104), # 104   u_max
    (105,112), # 106-112 n_max
    (113,114), # 114   f_min
    (115,116), # 116   l_min
    (117,124), # 118-124 min
    (125,126), # 126   u_min
    (127,135), # 128-135 n_min
    (136,150), # 137-150 Epoch
    (151,152), # 152   u_Epoch
    (153,154), # 154   l_Period
    (155,174), # 156-174 Period
    (175,176), # 176   u_Period
    (177,206), # 178-206 Sp
]

vsx_columns = ["id_vsx",
               "name",
               "var_flag", #0 = Variable, 1 = Suspected variable, 2 = Constant or non-existing, 3 = Possible duplicate
               "ra",
               "dec",
               "class", #GCVS catalog Variability type
                "l_max", 
                "mag_max",
                "u_max",
                "mag_band_max",
#     U B V R I   = Johnson broad-band
#     J H K L M   = Johnson infra-red (1.2, 1.6, 2.2, 3.5, 5um)
#     Rc Ic       = Cousins' red and infra-red
#     u v b y     = Stroemgren intermediate-band
#     u'g'r'i'z'  = Sloan (SDSS)
#     pg pv bj rf = photographic blue (pg, bj) visual (pv), red (rf)
#     w C CR CV   = white (clear); R or V used for comparison star.
#     R1          = ROTSE-I (450-1000nm)
#     Hp T        = Hipparcos and Tycho (Cat. I/239)
#     NUV         = near-UV (Galex)
#     H1A H1B     = STEREO mission filter (essentially 600-800nm)
                "f_min",
                "l_min",
                "mag_min",
                "u_min",
                "mag_band_min",
                "epoch",
                "u_epoch",
                "l_period",
                "period",
#     The value of the period may be followed by the following symbols:
#     : = uncertainty flag
#     ) = value of the mean cycle for U Gem or recurrent nova
#         (the corresponding opening bracket exists in l_Period)
#     *N = the real period is likely a multiple of the quoted period
#     /N = the period quoted is likely a multiple of the real period
                "u_period",
                "spectral_type"
]


# each file's column headers
tqdm.pandas(desc="Filtering VSX by class")

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
    colspecs=colspecs,
    names=vsx_columns,
    dtype=str
)

# coercions
for c in ["ra","dec","mag_max","mag_min","epoch","period"]:
    df_vsx[c] = pd.to_numeric(df_vsx[c], errors="coerce")

for c in ["id_vsx","var_flag","l_max","u_max","f_min", "l_min","u_min","u_epoch","l_period","u_period"]:
        df_vsx[c] = pd.to_numeric(df_vsx[c], errors="coerce").astype("Int64")


# vsx variability classes
EXCLUDE = set([
    # Eclipsing binaries (geometric, periodic dips)
    "E","EA","EB","E-DO","EP","EW","EC","ED","ESD", #eclipsing variables
    "AR","BD","D","DM","DS","DW","EL","GS","HW","K","KE","KW", #subtypes of eclipsing variables

    # White dwarf / compact single stars
    "PN","SD","WD",

    # Eruptive / flaring (brightenings, not dusty dimmings)
    "EXOR","FUOR","GCAS","UV","UVN","FF",
    "I","IA","IB","IS","ISB", "ZAND", "BE", "WR", "FSCMA",

    # Cataclysmic variables (interacting binaries, novae, magnetic systems)
    "AM","CBSS","CBSS/V","DQ","DQ/AE","IBWD",
    "N","NA","NB","NC","NL","NL/VY","NR", "CV", "ZAMD",

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
    "XB","XN","GRB","Transient","HMXB","LMXB",

    # Gravitational microlensing events
    "Microlens",

    # Non-stellar extragalactic sources
    "AGN","BLLAC","QSO","GALAXY",

    # Cataclysmic dwarf novae (subtypes of UG)
    "UG","UGER","UGSS","UGSU","UGWZ","UGZ","UGZ/IW",

    # Rotational variables (spotted stars, ellipsoidal, binaries, pulsars)
    "ACV", "ACV:","BY","CTTS/ROT","ELL","FKCOM","HB","LERI",
    "PSR","R","ROT","RS","SXARI","TTS/ROT","WTTS/ROT",
    "NSIN ELL","ROT (TTS subtype)",
    "r", # Assuming "r"=="R" for now
    "r'", # Assuming "r'"=="R" for now

    # Pulsating variables (radial/non-radial, classical pulsators, WDs, hot stars)
    "ACEP","ACEP:","ACEP(B)","ACEPS", "ACEPS:","ACYG","BCEP","BCEPS",
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

    # Unstudied variable stars with rapid light changes
    "S",

    # Miscellaneous subtypes
    "V",    # V Sge subtype of the CBSS variables
    "EA/SD", # β Persei-type (Algol) eclipsing systems, semi-detached EBs
    "",
    "Minor planet",

])



KEEP = set([
    # Explicit
    "DIP",              # VSX “dippers”
    "UXOR",             # UX Ori-type (disk occultations in YSOs)
    "APER",             # aperiodic variables
    "CST", #            # "Non-variable stars (constant), formerly suspected to be variable and hastily designated. Further observations have not confirmed their variability."
    "VAR","MISC","*",   # useful to not discard a priori
])

_SPLIT_RE = re.compile(r"[+|,]")  # split on + or | or ,; not "/" because those are baked into the types already
def _normalize_token(tok: str) -> str:
    t = tok.strip().upper()
    if not t:
        return ""
    # drop uncertainty and stray punctuation at ends
    t = t.replace(":", "")
    # drop any bracketed/parenthetical annotations (e.g., (YY), [E])
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"\[.*?\]", "", t)
    t = t.strip(" '\"")
    # small alias fixes
    if t == "PUL":  # sometimes appears as PUL in combos
        t = "PULS"
    return t

def tokenize_classes(s):
    if pd.isna(s):
        return []
    raw = [r for r in _SPLIT_RE.split(s) if r]
    out = []
    for r in raw:
        t = _normalize_token(r)
        if t:
            out.append(t)
    return out

def filter_vsx_classes(var_string):
    """
    screens out unwanted vsx variability classes
    Returns True if row should be excluded.
    KEEP classes override EXCLUDE if both present (e.g., UXOR/ROT => keep).
    """
    parts = set(tokenize_classes(var_string))
    if not parts:
        return False
    return bool(parts & EXCLUDE)


cont_var_mask = df_vsx["class"].progress_apply(filter_vsx_classes)
df_vsx_filt = df_vsx[~cont_var_mask].copy()

### BEGIN DEBUG OUTPUTS ###
print("Total rows:", len(df_vsx))
print("Excluded rows:", cont_var_mask.sum())
print("Kept rows:", (~cont_var_mask).sum())

print(df_vsx[["id_vsx","name","var_flag","ra","dec","class"]].head(5))
print("Class NaN rate:", df_vsx["class"].isna().mean())

# See top classes surviving/excluded
kept   = df_vsx[~cont_var_mask]
excl   = df_vsx[ cont_var_mask]
print("Kept count:", len(kept), "Excluded count:", len(excl))
print("Top kept classes:", kept["class"].value_counts().head(10))
print("Top excluded classes:", excl["class"].value_counts().head(10))

print(f"Shape of df_vsx_filt after filtering: {df_vsx_filt.shape}") # DEBUG
### END DEBUG OUTPUTS ###

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

df_all = pd.concat(dfs, ignore_index=True)

print(f"Shape of df_all after concat: {df_all.shape}")                       # DEBUG

# coerce ASAS-SN coords and drop NaNs BEFORE SkyCoord so indices match search results
df_all["ra_deg"]  = pd.to_numeric(df_all["ra_deg"], errors="coerce")
df_all["dec_deg"] = pd.to_numeric(df_all["dec_deg"], errors="coerce")
df_all_clean = df_all.dropna(subset=["ra_deg","dec_deg"]).reset_index(drop=True)
df_vsx_filt_clean = df_vsx_filt.dropna(subset=["ra","dec"]).reset_index(drop=True)

print(f"Shape of df_all_clean after dropna: {df_all_clean.shape}")           # DEBUG
print(f"Shape of df_vsx_filt_clean after dropna: {df_vsx_filt_clean.shape}") # DEBUG

# outputting cleaned concatenated asas-sn lightcurve index csv, and cleaned vsx csv
stamp = datetime.now().strftime("%Y%m%d_%H%M")

out_csv = p.cwd() / f"output/asassn_index_masked_concat_cleaned_{stamp}.csv"
df_all_clean.to_csv(out_csv, index=False)

out_csv = p.cwd() / f"output/vsx_cleaned_{stamp}.csv"
df_vsx_filt_clean.to_csv(out_csv, index=False)