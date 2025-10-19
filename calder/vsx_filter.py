from __future__ import annotations

import glob
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

MAG_BINS = ("12_12.5", "12.5_13", "13_13.5", "13.5_14", "14_14.5", "14.5_15")

DEFAULT_LC_DIR = Path("/data/poohbah/1/assassin/rowan.90/lcsv2")
DEFAULT_LC_DIR_MASKED = Path("/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked")
DEFAULT_VSX_FILE = Path("/data/poohbah/1/assassin/lenhart/code/calder/vsxcat.090525")
DEFAULT_OUTPUT_DIR = Path("/data/poohbah/1/assassin/lenhart/code/calder/calder/output")


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
    "AR","BD","D","DM","DS","DW","EL","GS","HW","K","KE","KW", #subtypes
    "E/DS","E/DW","E/GS","E/KE","E/KW","E/RS","E/SD",
    "EA/AR","EA/BD","EA/D","EA/DM","EA/DS","EA/EL","EA/GS","EA/HW",
    "EA/KE","EA/KW","EA/PN","EA/WD","EA/WR",
    "EB/AR","EB/D","EB/DM","EB/DW","EB/GS","EB/K","EB/KE","EB/KW",
    "EB/RS","EB/SD","EB/WR",
    "ELL/PN","ELL/RS/BY",
    "EW/D","EW/DM","EW/DW","EW/K","EW/KE","EW/KW","EW/RS",

    # White dwarf / compact single stars
    "PN","SD","WD",

    # Eruptive / flaring (brightenings, not dusty dimmings)
    "EXOR","FUOR","GCAS","UV","UVN","FF",
    "I","IA","IB","IS","ISB", "ZAND", "BE", "WR", "FSCMA","CPNB",
    "DPV/EA","DPV/EB","DPV/ELL",
    "EXOR/ROT","SDOR",

    # Cataclysmic variables (interacting binaries, novae, magnetic systems)
    "AM","CBSS","CBSS/V","DQ","DQ/AE","IBWD",
    "N","NA","NB","NC","NL","NL/VY","NR", "CV", "ZAMD",
    "UG","UGER","UGSS","UGSU","UGWZ","UGZ","UGZ/IW",
    "UG/DQ","UGSS/DQ","UGSU/IBWD","UGZ/DQ","NON-CV",

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
    "HMXB/XP",

    # Gravitational microlensing events
    "Microlens",

    # Non-stellar extragalactic sources
    "AGN","BLLAC","QSO","GALAXY",

    # Rotational variables (spotted stars, ellipsoidal, binaries, pulsars)
    "ACV", "ACV:","BY","BY:","CTTS/ROT","CTTS/DIP","ELL","FKCOM","HB","LERI",
    "PSR","R","ROT","RS","SXARI","TTS/ROT","WTTS/ROT",
    "NSIN ELL","ROT (TTS subtype)",
    "ROT/DIP","TTS/ROT/DIP","UXOR/ROT",
    "ROT/WD",
    "r","r'",

    # Pulsating variables (radial/non-radial, classical pulsators, WDs, hot stars)
    "ACEP","ACEP:","ACEP(B)","ACEPS","ACEPS:","ACYG","BCEP","BCEPS",
    "BLAP","BXCIR","CEP","CW","CWA","CWB","CWB(B)","CWBS",
    "DCEP","DCEP(B)","DCEPS","DCEPS(B)","DSCT","DSCTC",
    "DWLYN","GDOR","HADS","HADS(B)",
    "L","LB","LC","M","ORG",
    "PPN","PVTEL","PVTELI","PVTELII","PVTELIII",
    "roAm","roAp",
    "RR","RRAB","RRAB/BL","RRC","RRD","RRC/BL",
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
    "RCB","DYPer","VY",

    # Young stars; many show dipping episodes
    "YSO","TTS","CTTS","WTTS","YSO/DIP",
    "IN","INA","INB","INS","INSA","INSB","INST","INT","ISA","INAT",

    # Unstudied variable stars with rapid light changes
    "S",

    # Miscellaneous subtypes
    "V",   # V Sge subtype of CBSS
    "EA/SD","EA/RS",
    "",
    "Minor planet",
    "R/PN",
    "UXOR",             # UX Ori-type (disk occultations in YSOs)
    "APER",             # aperiodic variables
    "CST", #            # "Non-variable stars (constant), formerly suspected to be variable and hastily designated. Further observations have not confirmed their variability."
    "DIP", "DIP:",      # VSX “dippers”
])


KEEP = set([
    "VAR","MISC","*",
])

# Normalize policy sets to uppercase to match tokenization
EXCLUDE = set(s.upper() for s in EXCLUDE)
KEEP = set(s.upper() for s in KEEP)

_SPLIT_RE = re.compile(r"[+|/,]")  # split on + or | or , or /

def normalize_token(tok: str) -> str:
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
        t = normalize_token(r)
        if t:
            out.append(t)
    return out

def filter_vsx_classes(var_string):
    """
    Screens out unwanted VSX variability classes.
    Discard precedence: if any EXCLUDE token present -> exclude (return True).
    """
    parts = set(tokenize_classes(var_string))
    if not parts:
        return False
    return bool(parts & EXCLUDE)


def load_vsx_catalog(path: Path | str = DEFAULT_VSX_FILE) -> pd.DataFrame:
    """Load the raw VSX catalog and coerce numeric columns."""
    df = pd.read_fwf(path, colspecs=colspecs, names=vsx_columns, dtype=str)
    for col in ["ra", "dec", "mag_max", "mag_min", "epoch", "period"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["id_vsx", "var_flag", "l_max", "u_max", "f_min", "l_min", "u_min", "u_epoch", "l_period", "u_period"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def filter_vsx(df_vsx: pd.DataFrame) -> pd.DataFrame:
    """Return a VSX subset excluding unwanted variability classes."""
    mask = df_vsx["class"].progress_apply(filter_vsx_classes)
    df = df_vsx[~mask].copy()
    return df.dropna(subset=["ra", "dec"]).reset_index(drop=True)


def load_masked_indexes(masked_root: Path | str = DEFAULT_LC_DIR_MASKED) -> pd.DataFrame:
    """Concatenate all masked index CSVs into a single dataframe."""
    masked_root = Path(masked_root)
    masked_files = [
        f
        for mag_bin in MAG_BINS
        for f in glob.glob(str(masked_root / mag_bin / "index*_masked.csv"))
    ]
    dfs = [
        pd.read_csv(fpath, dtype={"asassn_id": "string"})
        for fpath in tqdm(masked_files, desc="Reading masked CSVs")
    ]
    df = pd.concat(dfs, ignore_index=True)
    df["ra_deg"] = pd.to_numeric(df["ra_deg"], errors="coerce")
    df["dec_deg"] = pd.to_numeric(df["dec_deg"], errors="coerce")
    return df.dropna(subset=["ra_deg", "dec_deg"]).reset_index(drop=True)


def collect_present_ids(lc_root: Path | str = DEFAULT_LC_DIR) -> set[str]:
    """Return the set of ASAS-SN IDs with .dat files present."""
    lc_root = Path(lc_root)
    present_ids = {
        Path(f).stem
        for mag_bin in MAG_BINS
        for f in glob.glob(str(lc_root / mag_bin / "lc*_cal/*.dat"))
    }
    return present_ids


def write_clean_outputs(
    df_asassn: pd.DataFrame,
    df_vsx: pd.DataFrame,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    stamp: str | None = None,
) -> tuple[Path, Path]:
    """Write cleaned ASAS-SN index and VSX CSVs, returning their paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M")
    asas_out = output_dir / f"asassn_index_masked_concat_cleaned_{stamp}.csv"
    vsx_out = output_dir / f"vsx_cleaned_{stamp}.csv"
    df_asassn.to_csv(asas_out, index=False)
    df_vsx.to_csv(vsx_out, index=False)
    return asas_out, vsx_out


def main(
    vsx_file: Path | str = DEFAULT_VSX_FILE,
    lc_dir: Path | str = DEFAULT_LC_DIR,
    masked_dir: Path | str = DEFAULT_LC_DIR_MASKED,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Run VSX filtering and ASAS-SN index cleaning and write cleaned CSVs."""
    df_vsx_raw = load_vsx_catalog(vsx_file)
    df_vsx_clean = filter_vsx(df_vsx_raw)
    df_asassn_clean = load_masked_indexes(masked_dir)
    _ = collect_present_ids(lc_dir)  # presently unused but retained for parity
    return write_clean_outputs(df_asassn_clean, df_vsx_clean, output_dir=output_dir)


if __name__ == "__main__":
    asas_path, vsx_path = main()
    print(f"Wrote ASAS-SN cleaned index to {asas_path}")
    print(f"Wrote VSX cleaned catalog to {vsx_path}")
