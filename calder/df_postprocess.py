import numpy as np
import pandas as pd

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
