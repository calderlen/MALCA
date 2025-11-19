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


# in lc_dips.process_record_naive
    #   mag_bin
    #   asas_sn_id
    #   index_num
    #   index_csv
    #   lc_dir
    #   dat_path
    #   raw_path
    #   g_n_peaks
    #   g_mean_mag
    #   g_peaks_idx
    #   g_peaks_jd
    #   v_n_peaks
    #   v_mean_mag
    #   v_peaks_idx
    #   v_peaks_jd
    #   jd_first
    #   jd_last
    #   n_rows_g
    #   n_rows_v
    
# in lc_dips.naive_dip_finder
    #    n_dip_runs,
    #    n_jump_runs,
    #    n_dip_points,
    #    n_jump_points,
    #    most_recent_dip,
    #    most_recent_jump,
    #    max_depth,
    #    max_height,
    #    max_dip_duration,
    #    max_jump_duration,
    #    dip_fraction
    #    jump_fraction


def read_asassn_dat(dat_path):
    """
    Read an ASAS-SN .dat file (fixed-width format) into a pandas DataFrame.
    """
    df = pd.read_fwf(
        dat_path,
        names=asassn_columns,
        dtype={
            "JD": float,
            "mag": float,
            "error": float,
            "good_bad": int,
            "camera#": int,
            "v_g_band": int,
            "saturated": int,
            "cam_field": str,
        },
    )
    return df


def load_detection_results(csv_path=DETECTION_RESULTS_FILE):
    """
    Load detection_results.csv into a DataFrame.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"detection_results file not found: {csv_path}")
    return pd.read_csv(csv_path)


def lookup_source_metadata(asassn_id=None, *, source_name=None, csv_path=DETECTION_RESULTS_FILE):
    """
    Fetch DAT_Path, Source, and Source_ID for a given ASAS-SN ID or J-name.
    """
    df = load_detection_results(csv_path)
    mask = pd.Series(True, index=df.index)
    if asassn_id is not None:
        mask &= df["Source_ID"].astype(str).str.strip() == str(asassn_id).strip()
    if source_name is not None:
        mask &= df["Source"].astype(str).str.strip().str.lower() == str(source_name).strip().lower()
    matches = df.loc[mask]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return {
        "dat_path": row.get("DAT_Path"),
        "source": row.get("Source"),
        "source_id": str(row.get("Source_ID")),
    }


def plot_one_lc(
    dat_path,
    *,
    out_path=None,
    out_format="pdf",
    title=None,
    source_name=None,
    jd_offset=0.0,
    figsize=(10, 6),
    show=False,
):
    """
    Plot an ASAS-SN .dat light curve separated by band (g vs. V) and save
    to the requested location.

    Args:
        dat_path (str | Path):
            Path to the .dat file.
        out_path (str | Path | None):
            Destination path. When None, the figure is saved to
            /data/poohbah/1/assassin/lenhart/asassn-variability/calder/lc_plots/<basename>.<format>.
        out_format (str):
            File format/extension to use when out_path is not provided. Defaults to 'pdf'.
        title (str | None):
            Figure title; defaults to "<basename> light curve".
        source_name (str | None):
            Optional J-name or alias to append next to the ASAS-SN ID in the title.
        jd_offset (float):
            Subtracted from the JD axis to improve readability.
        figsize (tuple):
            Matplotlib figure size (in inches).
        show (bool):
            If True, display the plot interactively; always saved to disk.
    """
    dat_path = Path(dat_path)
    metadata = None
    try:
        metadata = lookup_source_metadata(asassn_id=dat_path.stem)
    except FileNotFoundError:
        metadata = None

    df = read_asassn_dat(dat_path)

    # Basic cleaning similar to lc_utils.clean_lc
    mask = df["JD"].notna() & df["mag"].notna()
    mask &= df["error"].between(0, 1, inclusive="neither")
    mask &= df["saturated"] == 0
    mask &= df["good_bad"] == 1
    df = df.loc[mask].copy()
    if df.empty:
        raise ValueError(f"No valid rows found in {dat_path}")

    df["JD_plot"] = df["JD"] - float(jd_offset)

    fig, ax = pl.subplots(figsize=figsize, constrained_layout=True)
    ax.invert_yaxis()  # magnitudes: brighter lower

    camera_ids = sorted(df["camera#"].unique())
    cmap = pl.get_cmap("tab20", max(len(camera_ids), 1))
    camera_colors = {cam: cmap(i % cmap.N) for i, cam in enumerate(camera_ids)}
    band_markers = {0: "o", 1: "s"}
    camera_handles = {}

    for cam in camera_ids:
        cam_subset = df[df["camera#"] == cam]
        for band in (0, 1):
            subset = cam_subset[cam_subset["v_g_band"] == band]
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
            if cam not in camera_handles:
                camera_handles[cam] = Line2D(
                    [],
                    [],
                    color=color,
                    marker="o",
                    linestyle="",
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    label=f"Camera {cam}",
                )

    ax.set_xlabel(f"JD - {jd_offset:g}" if jd_offset else "JD")
    ax.set_ylabel("Magnitude")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    if camera_handles:
        ax.legend(
            handles=list(camera_handles.values()),
            title="Cameras",
            loc="best",
            fontsize="small",
            title_fontsize="small",
        )

    asassn_id = dat_path.stem
    if source_name is None and metadata:
        source_name = metadata.get("source")
    label = f"{source_name} ({asassn_id})" if source_name else asassn_id
    fig_title = title or f"{label} light curve"
    ax.set_title(fig_title)

    if out_path is None:
        ext = f".{out_format.lstrip('.')}" if out_format else ".pdf"
        out_path = PLOT_OUTPUT_DIR / f"{dat_path.stem}{ext}"
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


def plot_many_lc(
    dat_paths,
    *,
    out_dir=None,
    out_format="pdf",
    source_names=None,
    jd_offset=0.0,
    figsize=(10, 6),
    show=False,
):
    """
    Convenience wrapper that plots multiple .dat light curves in sequence.

    Args:
        dat_paths (Sequence[str | Path]): Iterable of .dat file paths.
        out_dir (str | Path | None): If provided, override the output directory
            for all plots. Filenames still follow <basename>.<format>.
        out_format (str): Desired file extension when out_dir is used.
        source_names (Mapping[str, str] | Sequence[str] | None): Optional mapping
            from ASAS-SN ID (file stem) to human-readable source name, or a list
            matching dat_paths.
        jd_offset (float): Pass-through to plot_one_lc.
        figsize (tuple): Pass-through to plot_one_lc.
        show (bool): Pass-through to plot_one_lc.

    Returns:
        list[str]: Paths of the generated plots.
    """
    outputs = []
    dat_paths = list(dat_paths)
    if source_names is None:
        lookup = {}
    elif isinstance(source_names, dict):
        lookup = source_names
    else:
        lookup = {Path(p).stem: name for p, name in zip(dat_paths, source_names)}

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for dat_path in dat_paths:
        dat_path = Path(dat_path)
        stem = dat_path.stem
        name = lookup.get(stem)
        if out_dir is not None:
            out_path = out_dir / f"{stem}.{out_format.lstrip('.')}"
        else:
            out_path = None

        saved = plot_one_lc(
            dat_path,
            out_path=out_path,
            out_format=out_format,
            source_name=name,
            jd_offset=jd_offset,
            figsize=figsize,
            show=show,
        )
        outputs.append(saved)

    return outputs
