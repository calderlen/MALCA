from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import glob
import re

from tqdm.auto import tqdm
import pandas as pd

from malca.config import LCV2_ROOT, LCV2_MASKED_ROOT, VSX_RAW_CATALOG_PATH
from malca.config import MAG_BINS




DEFAULT_LC_DIR = LCV2_ROOT
DEFAULT_LC_DIR_MASKED = LCV2_MASKED_ROOT
DEFAULT_VSX_FILE = VSX_RAW_CATALOG_PATH
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "input" / "vsx"


colspecs = [
    (0, 8),               
    (9, 39),               
    (40, 41),                              
    (42, 51),               
    (52, 61),               
    (62, 92),              
    (93, 94),               
    (95, 102),             
    (103,104),              
    (105,112),                
    (113,114),              
    (115,116),              
    (117,124),              
    (125,126),              
    (127,135),                
    (136,150),                
    (151,152),                
    (153,154),                 
    (155,174),                 
    (175,176),                 
    (177,206),             
]

vsx_columns = ["id_vsx",
               "name",
               "var_flag",                                                                                            
               "ra",
               "dec",
               "class",                               
                "l_max", 
                "mag_max",
                "u_max",
                "mag_band_max",
                                      
                                                               
                                              
                                                
                                
                                                                    
                                                                   
                                        
                                                    
                                   
                                                                 
                "f_min",
                "l_min",
                "mag_min",
                "u_min",
                "mag_band_min",
                "epoch",
                "u_epoch",
                "l_period",
                "period",
                                                                       
                          
                                                             
                                                                
                                                                    
                                                                    
                "u_period",
                "spectral_type"
]


                            

tqdm.pandas(desc="Filtering VSX by class")


EXCLUDE = set([
                                                   
    "E","EA","EB","E-DO","EP","EW","EC","ED","ESD",                     
    "AR","BD","D","DM","DS","DW","EL","GS","HW","K","KE","KW",          
    "E/DS","E/DW","E/GS","E/KE","E/KW","E/RS","E/SD",
    "EA/AR","EA/BD","EA/D","EA/DM","EA/DS","EA/EL","EA/GS","EA/HW",
    "EA/KE","EA/KW","EA/PN","EA/WD","EA/WR",
    "EB/AR","EB/D","EB/DM","EB/DW","EB/GS","EB/K","EB/KE","EB/KW",
    "EB/RS","EB/SD","EB/WR",
    "ELL/PN","ELL/RS/BY",
    "EW/D","EW/DM","EW/DW","EW/K","EW/KE","EW/KW","EW/RS",

                                        
    "PN","SD","WD",

                                                           
    "EXOR","FUOR","GCAS","UV","UVN","FF",
    "I","IA","IB","IS","ISB", "ZAND", "BE", "WR", "FSCMA","CPNB",
    "DPV/EA","DPV/EB","DPV/ELL",
    "EXOR/ROT","SDOR",

                                                                           
    "AM","CBSS","CBSS/V","DQ","DQ/AE","IBWD",
    "N","NA","NB","NC","NL","NL/VY","NR", "CV", "ZAMD",
    "UG","UGER","UGSS","UGSU","UGWZ","UGZ","UGZ/IW",
    "UG/DQ","UGSS/DQ","UGSU/IBWD","UGZ/DQ","NON-CV",

                                                                        
    "SN","SN I","SN Ia","SN Iax",
    "SN Ia-00cx-like","SN Ia-02es-like","SN Ia-06gz-like",
    "SN Ia-86G-like","SN Ia-91bg-like","SN Ia-91T-like","SN Ia-99aa-like",
    "SN Ia-Ca-rich","SN Ia-CSM",
    "SN Ib","SN Ic","SN Icn","SN Ic-BL","SN Idn","SN Ien",
    "SN II","SN IIa","SN IIb","SN IId","SN II-L","SN IIn","SN II-P",
    "SN-pec",
    "SLSN","SLSN-I","SLSN-II",
    "V838MON","LFBOT",

                                        
    "XB","XN","GRB","Transient","HMXB","LMXB",
    "HMXB/XP",

                                       
    "Microlens",

                                       
    "AGN","BLLAC","QSO","GALAXY",

                                                                          
    "ACV", "ACV:","BY","BY:","CTTS/ROT","CTTS/DIP","ELL","FKCOM","HB","LERI",
    "PSR","R","ROT","RS","SXARI","TTS/ROT","WTTS/ROT",
    "NSIN ELL","ROT (TTS subtype)",
    "ROT/DIP","TTS/ROT/DIP","UXOR/ROT",
    "ROT/WD",
    "r","r'",

                                                                                  
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

                                         
    "WDP","XP",

                                                           
    "NSIN","PER","SIN","VBD",

                                                
    "RCB","DYPer","VY",

                                             
    "YSO","TTS","CTTS","WTTS","YSO/DIP",
    "IN","INA","INB","INS","INSA","INSB","INST","INT","ISA","INAT",

                                                       
    "S",

                            
    "V",                          
    "EA/SD","EA/RS",
    "",
    "Minor planet",
    "R/PN",
    "UXOR",                                                      
    "APER",                                  
    "CST",                                                                                                                                                                     
    "DIP", "DIP:",                     
])


KEEP = set([
    "VAR","MISC","*",
])

                                                          
EXCLUDE = set(s.upper() for s in EXCLUDE)
KEEP = set(s.upper() for s in KEEP)

_SPLIT_RE = re.compile(r"[+|/,]")                             

def normalize_token(tok: str) -> str:
    t = tok.strip().upper()
    if not t:
        return ""
                                                    
    t = t.replace(":", "")
                                                                    
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"\[.*?\]", "", t)
    t = t.strip(" '\"")
                       
    if t == "PUL":                                      
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
    """Write cleaned ASAS-SN index and VSX Parquets, returning their paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stamp:
        asas_out = output_dir / f"asassn_index_masked_concat_cleaned_{stamp}.parquet"
        vsx_out = output_dir / f"vsx_cleaned_{stamp}.parquet"
    else:
        # Use simple default names matching tag.py expectations
        asas_out = output_dir / "asassn_catalog.parquet"
        vsx_out = output_dir / "vsx_cleaned.parquet"

    df_asassn.to_parquet(asas_out, index=False)
    df_vsx.to_parquet(vsx_out, index=False)
    return asas_out, vsx_out


def main(
    vsx_file: Path | str = DEFAULT_VSX_FILE,
    masked_dir: Path | str = DEFAULT_LC_DIR_MASKED,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    stamp: str | None = None,
) -> tuple[Path, Path]:
    """Run VSX filtering and ASAS-SN index cleaning and write cleaned Parquets."""
    df_vsx_raw = load_vsx_catalog(vsx_file)
    df_vsx_clean = filter_vsx(df_vsx_raw)
    df_asassn_clean = load_masked_indexes(masked_dir)
    return write_clean_outputs(df_asassn_clean, df_vsx_clean, output_dir=output_dir, stamp=stamp)


def cli():
    """CLI entry point for ``malca vsx-filter``."""


    parser = argparse.ArgumentParser(
        description="Build cleaned ASAS-SN index and filtered VSX catalog."
    )
    parser.add_argument("--vsx-file", type=Path, default=DEFAULT_VSX_FILE,
                        help=f"Raw VSX catalog (default: {DEFAULT_VSX_FILE})")
    parser.add_argument("--masked-dir", type=Path, default=DEFAULT_LC_DIR_MASKED,
                        help=f"Masked light-curve index root (default: {DEFAULT_LC_DIR_MASKED})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--stamp", type=str, default=None,
                        help="Timestamp suffix for output filenames (default: no stamp)")
    args = parser.parse_args()

    asas_path, vsx_path = main(
        vsx_file=args.vsx_file,
        masked_dir=args.masked_dir,
        output_dir=args.output_dir,
        stamp=args.stamp,
    )
    print(f"Wrote ASAS-SN cleaned index to {asas_path}")
    print(f"Wrote VSX cleaned catalog to {vsx_path}")


if __name__ == "__main__":
    cli()
