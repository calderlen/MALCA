import matplotlib.pyplot as pl
import matplotlib.ticker as tick
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from pathlib import Path
import time

from lc_baseline import (
    global_mean_baseline,
    global_median_baseline,
    global_rolling_mean_baseline,
    global_rolling_median_baseline,
    per_camera_mean_baseline,
    per_camera_median_baseline,
    per_camera_trend_baseline,
)


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

PLOT_OUTPUT_DIR = Path("/lc_plots")
DETECTION_RESULTS_FILE = Path("calder/detection_results.csv")
JD_OFFSET = 2458000.0
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

SKYPATROL_CSV_PATHS = [
    "calder/lc_data_skypatrol/120259184943-light-curves.csv",
    "calder/lc_data_skypatrol/223339338105-light-curves.csv",
    "calder/lc_data_skypatrol/231929175915-light-curves.csv",
    "calder/lc_data_skypatrol/25770019815-light-curves.csv",
    "calder/lc_data_skypatrol/266288137752-light-curves.csv",
    "calder/lc_data_skypatrol/317827964025-light-curves.csv",
    "calder/lc_data_skypatrol/326417831663-light-curves.csv",
    "calder/lc_data_skypatrol/335007754417-light-curves.csv",
    "calder/lc_data_skypatrol/352187470767-light-curves.csv",
    "calder/lc_data_skypatrol/360777377116-light-curves.csv",
    "calder/lc_data_skypatrol/377957522430-light-curves.csv",
    "calder/lc_data_skypatrol/377958261591-light-curves.csv",
    "calder/lc_data_skypatrol/412317159120-light-curves.csv",
    "calder/lc_data_skypatrol/438086901547-light-curves.csv",
    "calder/lc_data_skypatrol/438086977939-light-curves.csv",
    "calder/lc_data_skypatrol/455267102087-light-curves.csv",
    "calder/lc_data_skypatrol/463856535113-light-curves.csv",
    "calder/lc_data_skypatrol/472447294641-light-curves.csv",
    "calder/lc_data_skypatrol/515396514761-light-curves.csv",
    "calder/lc_data_skypatrol/515397118400-light-curves.csv",
    "calder/lc_data_skypatrol/532576686103-light-curves.csv",
    "calder/lc_data_skypatrol/601296043597-light-curves.csv",
    "calder/lc_data_skypatrol/60130040391-light-curves.csv",
    "calder/lc_data_skypatrol/609886184506-light-curves.csv",
    "calder/lc_data_skypatrol/635655234580-light-curves.csv",
    "calder/lc_data_skypatrol/644245387906-light-curves.csv",
    "calder/lc_data_skypatrol/661425129485-light-curves.csv",
    "calder/lc_data_skypatrol/68720274411-light-curves.csv",
]

DEFAULT_LC_PATHS = DEFAULT_DAT_PATHS + SKYPATROL_CSV_PATHS

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


def read_asassn_dat(dat_path):
    """
    Read an ASAS-SN .dat file using whitespace separation.
    """
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

def read_skypatrol_csv(csv_path):
    """
    Read a SkyPatrol CSV, remapping columns to the ASAS-SN schema.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(
        csv_path,
        comment="#",
        dtype={
            "JD": float,
            "Flux": float,
            "Flux Error": float,
            "Mag": float,
            "Mag Error": float,
            "Limit": float,
            "FWHM": float,
            "Filter": "string",
            "Quality": "string",
            "Camera": "string",
        },
    )
    rename_map = {
        "Flux": "flux",
        "Flux Error": "flux_error",
        "Mag": "mag",
        "Mag Error": "error",
        "Limit": "limit",
        "FWHM": "fwhm",
        "Filter": "filter_band",
        "Quality": "quality_flag",
        "Camera": "camera",
    }
    df = df.rename(columns=rename_map)
    df["JD"] = pd.to_numeric(df["JD"], errors="coerce")
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    df["error"] = pd.to_numeric(df["error"], errors="coerce")
    df["flux"] = pd.to_numeric(df.get("flux"), errors="coerce")
    df["flux_error"] = pd.to_numeric(df.get("flux_error"), errors="coerce")
    df["camera"] = df["camera"].astype(str).str.strip()
    df["camera#"] = df["camera"]
    df["cam_field"] = df["camera#"]
    df["quality_flag"] = df["quality_flag"].astype(str).str.strip().str.upper()
    df["good_bad"] = (df["quality_flag"] == "G").astype(int)
    df["saturated"] = 0

    filt = df["filter_band"].astype(str).str.strip().str.lower()
    band_map = {"v": 1, "g": 0}
    df["v_g_band"] = filt.map(band_map)
    df = df[df["v_g_band"].notna()].copy()
    df["v_g_band"] = df["v_g_band"].astype(int)

    df = df[pd.notna(df["JD"]) & pd.notna(df["mag"])]
    df = df.sort_values("JD").reset_index(drop=True)
    return df

def _load_lightcurve_df(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_skypatrol_csv(path)
    return read_asassn_dat(path)

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

def _lookup_metadata_for_path(path: Path):
    """
    Try to resolve metadata for a given light curve path, falling back to ID stems.
    """
    metadata = None
    stem = path.stem
    try:
        metadata = lookup_source_metadata(asassn_id=stem, dat_path=str(path))
    except FileNotFoundError:
        metadata = None

    if metadata:
        return metadata

    base_id = stem.split("-", 1)[0]
    if base_id and base_id != stem:
        try:
            metadata = lookup_source_metadata(asassn_id=base_id)
        except FileNotFoundError:
            metadata = None
    return metadata


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
    metadata = _lookup_metadata_for_path(dat_path)

    df = _load_lightcurve_df(dat_path)

    mask = df["JD"].notna() & df["mag"].notna()
    mask &= df["error"].between(0, 1, inclusive="neither")
    mask &= df["saturated"] == 0
    mask &= df["good_bad"] == 1
    df = df.loc[mask].copy()
    if df.empty:
        print(f"Warning: No valid rows found in {dat_path}")
        return None

    df["JD_plot"] = df["JD"] - JD_OFFSET
    
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
                subset["JD_plot"],
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

    axes[-1].set_xlabel(f"JD - {int(JD_OFFSET)} [d]")

    asassn_id = dat_path.stem
    category = metadata.get("category") if metadata else None
    jd_start = float(df["JD_plot"].min())
    jd_end = float(df["JD_plot"].max())
    jd_label = f"JD - {int(JD_OFFSET)} [{jd_start:.0f}:{jd_end:.0f}]"
    
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
    dat_paths=None,
    *,
    out_dir=None,
    out_format="pdf",
    source_names=None,
    figsize=(10, 6),
    show=False,
):
    outputs = []
    if dat_paths is None:
        dat_paths = DEFAULT_LC_PATHS
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


def _plot_lc_with_residuals_df(
    df,
    *,
    out_path,
    out_format="pdf",
    title=None,
    source_name=None,
    figsize=(12, 8),
    show=False,
    metadata=None,
):
    required_cols = {"JD", "mag", "error", "v_g_band", "camera#", "resid"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"plot_lc_with_residuals requires columns: {missing}")

    data = df.copy()
    data = data[np.isfinite(data["JD"]) & np.isfinite(data["mag"])]
    if data.empty:
        raise ValueError("No finite JD/mag values available for plotting.")
    data["JD_plot"] = data["JD"] - JD_OFFSET

    preferred_order = [1, 0]
    bands = [band for band in preferred_order if (data["v_g_band"] == band).any()]
    if not bands:
        bands = sorted(data["v_g_band"].dropna().unique())
    if not bands:
        raise ValueError("No band information available (v_g_band missing).")

    n_cols = len(bands)
    fig, axes = pl.subplots(
        2,
        n_cols,
        figsize=figsize,
        constrained_layout=True,
        sharex="col",
    )
    
    if n_cols == 1:
        axes = np.array(axes).reshape(2, 1)

    camera_ids = sorted(data["camera#"].dropna().unique())
    cmap = pl.get_cmap("tab20", max(len(camera_ids), 1))
    camera_colors = {cam: cmap(i % cmap.N) for i, cam in enumerate(camera_ids)}
    band_labels = {0: "g band", 1: "V band"}
    band_markers = {0: "o", 1: "s"}

    for col_idx, band in enumerate(bands):
        band_mask = data["v_g_band"] == band
        band_df = data[band_mask]
        if band_df.empty:
            continue

        raw_ax = axes[0, col_idx]
        resid_ax = axes[1, col_idx]

        raw_ax.invert_yaxis()
        raw_ax.grid(True, which="both", linestyle="--", alpha=0.3)
        raw_ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        raw_ax.set_xlabel(f"JD - {int(JD_OFFSET)} [d]")
        raw_ax.xaxis.set_label_position("top")

        resid_ax.grid(True, which="both", linestyle="--", alpha=0.3)
        resid_ax.axhline(0.0, color="black", linestyle="--", alpha=0.4, zorder=1)
        
        resid_ax.invert_yaxis()

        resid_ax.axhline(0.3, color="black", linestyle="-", linewidth=0.8, zorder=1)
        resid_ax.axhline(-0.3, color="black", linestyle="-", linewidth=0.8, zorder=1)
        
        resid_ax.fill_between(
            [band_df["JD_plot"].min(), band_df["JD_plot"].max()],
            0.3, 100, # Fill "down" (visually) for dips > 0.3
            color="lightgrey", alpha=0.5, zorder=0
        )
        resid_ax.fill_between(
            [band_df["JD_plot"].min(), band_df["JD_plot"].max()],
            -0.3, -100, # Fill "up" (visually) for negative deviations < -0.3
            color="lightgrey", alpha=0.45, zorder=0
        )

        legend_handles = {}
        
        for cam in camera_ids:
            cam_subset = band_df[band_df["camera#"] == cam]
            if cam_subset.empty:
                continue
            
            color = camera_colors[cam]
            marker = band_markers.get(band, "o")
            
            raw_ax.errorbar(
                cam_subset["JD_plot"],
                cam_subset["mag"],
                yerr=cam_subset["error"],
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
            
            resid_ax.scatter(
                cam_subset["JD_plot"],
                cam_subset["resid"],
                s=10,
                color=color,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.3,
                marker=marker,
                zorder=3 
            )
            
            if cam not in legend_handles:
                legend_handles[cam] = Line2D(
                    [], [],
                    color=color, marker="o", linestyle="",
                    markeredgecolor="black", markeredgewidth=0.5,
                    label=f"Camera {cam}",
                )

        band_name = band_labels.get(band, f'band {band}')
        raw_ax.set_ylabel(f"{band_name} mag")
        resid_ax.set_ylabel(f"Residual {band_name}")
        resid_ax.set_xlabel("JD")

        resid_min = band_df["resid"].min()
        resid_max = band_df["resid"].max()
        pad = (resid_max - resid_min) * 0.1 if resid_max != resid_min else 0.1
        plot_min = min(resid_min - pad, -0.35) 
        plot_max = max(resid_max + pad, 0.35)
        resid_ax.set_ylim(plot_max, plot_min)

        if legend_handles:
            raw_ax.legend(
                handles=list(legend_handles.values()),
                title="Cameras",
                loc="best",
                fontsize="small",
            )

    source_id = metadata.get("source_id") if metadata else None
    category = metadata.get("category") if metadata else None
    src_name = source_name or (metadata.get("source") if metadata else None)
    
    if src_name and source_id:
        label = f"{src_name} ({source_id})"
    else:
        label = src_name or source_id or ""
        
    jd_start = float(data["JD"].min())
    jd_end = float(data["JD"].max())
    jd_label = f"JD {jd_start:.0f}-{jd_end:.0f}"
    
    title_parts = []
    if label: title_parts.append(label)
    if category: title_parts.append(category)
    title_parts.append(jd_label)
    
    fig.suptitle(title or " – ".join(title_parts), fontsize="large")

    if out_path is None:
        base = source_id or src_name or "lc"
        ext = f".{out_format.lstrip('.')}" if out_format else ".pdf"
        out_path = PLOT_OUTPUT_DIR / f"{base}_residuals{ext}"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path, dpi=400)
    else:
        fig.savefig(out_path)

    if show:
        pl.show()
    else:
        pl.close(fig)

    return str(out_path)

def plot_lc_with_residuals(
    df=None,
    *,
    dat_paths=None,
    baseline_func=None,
    baseline_kwargs=None,
    out_path=None,
    out_format="pdf",
    title=None,
    source_name=None,
    figsize=(12, 8),
    show=False,
    metadata=None,
    baseline_tag=None,
    timestamp_output=False,
):
    """
    Wrapper to handle either a DataFrame or a list of file paths.
    """
    if baseline_func is None:
        baseline_func = per_camera_trend_baseline
        
    baseline_kwargs = baseline_kwargs or {}
    if baseline_tag is None and baseline_func is not None:
        baseline_tag = getattr(baseline_func, "__name__", "baseline")
    results: list[str] = []

    if df is not None:
        dest = out_path
        if timestamp_output and dest is not None:
            dest = Path(dest)
            stamp = Path(time.strftime("%Y%m%d_%H%M%S"))
            dest = dest.with_name(f"{dest.stem}_{stamp}{dest.suffix}")
        return _plot_lc_with_residuals_df(
            df,
            out_path=dest,
            out_format=out_format,
            title=title,
            source_name=source_name,
            figsize=figsize,
            show=show,
            metadata=metadata,
        )

    if dat_paths is None:
        dat_paths = DEFAULT_LC_PATHS
    dat_paths = list(dat_paths)

    multi = len(dat_paths) > 1
    out_dir = None
    if multi:
        if out_path is not None:
            out_dir = Path(out_path)
        else:
            out_dir = PLOT_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

    for path in dat_paths:
        path = Path(path)
        try:
            df_raw = _load_lightcurve_df(path)
        except Exception as exc:
            print(f"[warn] Failed to read {path}: {exc}")
            continue

        if baseline_func is not None:
            try:
                df_base = baseline_func(df_raw, **baseline_kwargs)
            except Exception as exc:
                print(f"[warn] baseline_func failed for {path}: {exc}")
                continue
        else:
            df_base = df_raw

        if "resid" not in df_base.columns:
            print(f"[warn] No 'resid' column for {path}; skipping.")
            continue

        meta = metadata
        if meta is None:
            meta = _lookup_metadata_for_path(path)

        if multi:
            ext = f".{out_format.lstrip('.')}" if out_format else ".pdf"
            base_name = f"{path.stem}_{baseline_tag}" if baseline_tag else f"{path.stem}"
            if timestamp_output:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                base_name = f"{base_name}_{stamp}"
            dest = (out_dir / f"{base_name}_residuals{ext}") if out_dir else None
        else:
            dest = out_path
            if dest is not None:
                dest = Path(dest)
                base_name = dest.stem
                if baseline_tag:
                    base_name = f"{base_name}_{baseline_tag}"
                if timestamp_output:
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    base_name = f"{base_name}_{stamp}"
                dest = dest.with_name(f"{base_name}{dest.suffix}")

        result = _plot_lc_with_residuals_df(
            df_base,
            out_path=dest,
            out_format=out_format,
            title=title,
            source_name=source_name,
            figsize=figsize,
            show=show,
            metadata=meta,
        )
        results.append(result)

    if not results:
        return None
    return results if multi else results[0]
