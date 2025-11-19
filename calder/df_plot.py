import matplotlib.pyplot as pl
import matplotlib.ticker as tick
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import scipy.signal
from pathlib import Path
from astropy.time import Time
from df_utils import jd_to_year, year_to_jd

# these are all of the non-derived columns we have to work with -- consider joining them together here as necessary


asassn_columns=["JD",
                "mag",
                'error', 
                'good_bad', #1=good, 0 =bad
                'camera#', 
                'v_g_band', #1=V, 0=g
                'saturated',
                'cam_field']
  
asassn_raw_columns = [
                'cam#',
                'median',
                '1siglow', 
               '1sighigh', 
               '90percentlow',
               '90percenthigh']

PLOT_OUTPUT_DIR = Path("/data/poohbah/1/assassin/lenhart/asassn-variability/calder/lc_plots")
DETECTION_RESULTS_FILE = Path("calder/detection_results.csv")
DEFAULT_DAT_PATHS = [
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13_13.5/lc18_cal/377957522430.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13_13.5/lc22_cal/42950993887.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13_13.5/lc21_cal/223339338105.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13_13.5/lc13_cal/601296043597.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13_13.5/lc13_cal/472447294641.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13.5_14/lc6_cal/455267102087.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13_13.5/lc4_cal/266288137752.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13.5_14/lc17_cal/532576686103.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/12_12.5/lc8_cal/352187470767.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/12.5_13/lc19_cal/609886184506.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13_13.5/lc22_cal/68720274411.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13.5_14/lc9_cal/377958261591.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13.5_14/lc14_cal/515397118400.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13_13.5/lc29_cal/326417831663.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/12_12.5/lc11_cal/644245387906.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13.5_14/lc28_cal/661425129485.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13.5_14/lc18_cal/438086977939.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13_13.5/lc5_cal/360777377116.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13_13.5/lc10_cal/635655234580.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13.5_14/lc35_cal/412317159120.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13.5_14/lc36_cal/438086901547.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13_13.5/lc23_cal/463856535113.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13_13.5/lc2_cal/120259184943.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13.5_14/lc20_cal/25770019815.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13.5_14/lc2_cal/515396514761.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/14_14.5/lc51_cal/231929175915.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/14.5_15/lc2_cal/335007754417.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/12.5_13/lc10_cal/60130040391.dat",
    "/data/poohbah/1/assassin/rowan.90/lcsv2/13_13.5/lc6_cal/317827964025.dat",
]

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


# stats you have to work with, this is everything you've derived from the above data and the file structure
import matplotlib.pyplot as pl
import matplotlib.ticker as tick
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from pathlib import Path

# --- CONFIG ---
PLOT_OUTPUT_DIR = Path("/data/poohbah/1/assassin/lenhart/asassn-variability/calder/lc_plots")
DETECTION_RESULTS_FILE = Path("calder/detection_results.csv")

def read_asassn_dat(dat_path):
    """
    Read an ASAS-SN .dat file using whitespace separation.
    """
    # Using sep='\s+' is generally more robust for these files than fixed-width
    df = pd.read_csv(
        dat_path,
        sep=r'\s+',
        names=asassn_columns,
        dtype={
            "JD": float, "mag": float, "error": float, 
            "good_bad": int, "camera#": int, 
            "v_g_band": int, "saturated": int, "cam_field": str,
        },
        comment='#'
    )
    return df

def load_detection_results(csv_path=DETECTION_RESULTS_FILE):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"detection_results file not found: {csv_path}")
    df = pd.read_csv(
        csv_path,
        dtype={"Source_ID": "string", "Source": "string", "DAT_Path": "string", "Category": "string"},
        keep_default_na=False,
    )
    df["DAT_Path"] = df["DAT_Path"].astype(str).str.strip()
    df["Source_ID"] = df["DAT_Path"].apply(lambda p: Path(p).stem if p else "")
    df["Source"] = df["Source"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()
    return df

def lookup_source_metadata(asassn_id=None, *, source_name=None, dat_path=None, csv_path=DETECTION_RESULTS_FILE):
    df = load_detection_results(csv_path)
    mask = pd.Series(True, index=df.index)
    if asassn_id is not None:
        mask &= df["Source_ID"].astype(str).str.strip() == str(asassn_id).strip()
    if source_name is not None:
        mask &= df["Source"].astype(str).str.strip().str.lower() == str(source_name).strip().lower()
    if dat_path is not None:
        mask &= df["DAT_Path"].astype(str).str.strip() == str(dat_path).strip()
    matches = df.loc[mask]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return {
        "dat_path": row.get("DAT_Path"),
        "source": row.get("Source"),
        "source_id": str(row.get("Source_ID")),
        "category": row.get("Category"),
    }

def plot_one_lc(
    dat_path,
    *,
    out_path=None,
    out_format="pdf",
    title=None,
    source_name=None,
    figsize=(10, 6),
    show=False,
    **kwargs
):
    dat_path = Path(dat_path)
    metadata = None
    try:
        metadata = lookup_source_metadata(asassn_id=dat_path.stem, dat_path=str(dat_path))
    except FileNotFoundError:
        metadata = None

    df = read_asassn_dat(dat_path)

    # Cleaning
    mask = df["JD"].notna() & df["mag"].notna()
    mask &= df["error"].between(0, 1, inclusive="neither")
    mask &= df["saturated"] == 0
    mask &= df["good_bad"] == 1
    df = df.loc[mask].copy()
    
    if df.empty:
        print(f"Warning: No valid rows found in {dat_path}")
        return None

    bands_present = [band for band in (0, 1) if (df["v_g_band"] == band).any()]
    if not bands_present:
        print(f"Warning: No g or V band data in {dat_path}")
        return None

    ax_count = len(bands_present)
    fig, axes = pl.subplots(ax_count, 1, figsize=figsize, constrained_layout=True, sharex=True)
    if ax_count == 1:
        axes = [axes]

    camera_ids = sorted(df["camera#"].unique())
    cmap = pl.get_cmap("tab20", max(len(camera_ids), 1))
    camera_colors = {cam: cmap(i % cmap.N) for i, cam in enumerate(camera_ids)}
    band_labels = {0: "g band", 1: "V band"}
    band_markers = {0: "o", 1: "s"}

    for ax, band in zip(axes, bands_present):
        ax.invert_yaxis()
        band_df = df[df["v_g_band"] == band]
        legend_handles = {}

        for cam in camera_ids:
            subset = band_df[band_df["camera#"] == cam]
            if subset.empty:
                continue
            color = camera_colors[cam]
            marker = band_markers.get(band, "o")
            ax.errorbar(
                subset["JD"],
                subset["mag"],
                yerr=subset["error"],
                fmt=marker,
                ms=4,
                color=color,
                alpha=0.8,
                ecolor=color,
                elinewidth=0.8,
                capsize=2,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )
            if cam not in legend_handles:
                legend_handles[cam] = Line2D([], [], color=color, marker='o', linestyle="", markeredgecolor="black", markeredgewidth=0.5, label=f"Camera {cam}")

        ax.set_ylabel(f"{band_labels.get(band, f'band {band}')} mag")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        if legend_handles:
            ax.legend(handles=list(legend_handles.values()), title="Cameras", loc="best", fontsize="small")

    axes[-1].set_xlabel("JD (raw)")

    asassn_id = dat_path.stem
    category = metadata.get("category") if metadata else None
    jd_start = float(df["JD"].min())
    jd_end = float(df["JD"].max())
    jd_label = f"JD {jd_start:.0f}-{jd_end:.0f}"
    
    if source_name is None and metadata:
        source_name = metadata.get("source")
        
    label = f"{source_name} ({asassn_id})" if source_name else asassn_id
    parts = [label]
    if category: parts.append(category)
    parts.append(jd_label)
    
    fig_title = title or f"{' – '.join(parts)} light curve"
    axes[0].set_title(fig_title)

    if out_path is None:
        ext = f".{out_format.lstrip('.')}" if out_format else ".pdf"
        out_path = PLOT_OUTPUT_DIR / f"{dat_path.stem}{ext}"
        
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path, dpi=400)
    else:
        fig.savefig(out_path)
        
    if show: pl.show()
    else: pl.close(fig)

    return str(out_path)

def plot_many_lc(
    dat_paths,
    *,
    out_dir=None,
    out_format="pdf",
    source_names=None,
    figsize=(10, 6),
    show=False,
    # Removed jd_offset here too
):
    outputs = []
    dat_paths = list(dat_paths)
    
    if source_names is None: lookup = {}
    elif isinstance(source_names, dict): lookup = source_names
    else: lookup = {Path(p).stem: name for p, name in zip(dat_paths, source_names)}

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for dat_path in dat_paths:
        dat_path = Path(dat_path)
        stem = dat_path.stem
        name = lookup.get(stem)
        
        out_path = (out_dir / f"{stem}.{out_format.lstrip('.')}") if out_dir else None

        saved = plot_one_lc(
            dat_path,
            out_path=out_path,
            out_format=out_format,
            source_name=name,
            figsize=figsize,
            show=show,
        )
        if saved: outputs.append(saved)

    return outputs
