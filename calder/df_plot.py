import matplotlib.pyplot as pl
import matplotlib.ticker as tick
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


def plot_dat_lightcurve(
    dat_path,
    *,
    out_path='/data/poohbah/1/assassin/lenhart/asassn-variability/calder/lc_plots',
    title=None,
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
            Where to store the resulting image. When None, a PNG is written
            next to the .dat file.
        title (str | None):
            Figure title; defaults to "<basename> light curve".
        jd_offset (float):
            Subtracted from the JD axis to improve readability.
        figsize (tuple):
            Matplotlib figure size (in inches).
        show (bool):
            If True, display the plot interactively; always saved to disk.
    """
    dat_path = Path(dat_path)
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

    colors = {0: "#1f77b4", 1: "#d62728"}  # g-band, V-band
    labels = {0: "g band", 1: "V band"}

    for band in (0, 1):
        subset = df[df["v_g_band"] == band]
        if subset.empty:
            continue
        ax.errorbar(
            subset["JD_plot"],
            subset["mag"],
            yerr=subset["error"],
            fmt="o",
            ms=3,
            color=colors[band],
            alpha=0.7,
            ecolor=colors[band],
            elinewidth=0.8,
            capsize=2,
            label=f"{labels[band]} (N={len(subset)})",
        )

    ax.set_xlabel(f"JD - {jd_offset:g}" if jd_offset else "JD")
    ax.set_ylabel("Magnitude")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend()

    fig_title = title or f"{dat_path.stem} light curve"
    ax.set_title(fig_title)

    if out_path is None:
        out_dir = Path("lc_plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{dat_path.stem}.png"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=200)
    if show:
        pl.show()
    else:
        pl.close(fig)

    return str(out_path)
