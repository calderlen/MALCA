#from vsx_crossmatch import mask_index_dir

dir_12_12_5 = "/data/poohbah/1/assassin/rowan.90/lcsv2/12_12.5"
dir_12_5_13 = "/data/poohbah/1/assassin/rowan.90/lcsv2/12.5_13"
dir_13_13_5 = "/data/poohbah/1/assassin/rowan.90/lcsv2/13_13.5"
dir_13_5_14 = "/data/poohbah/1/assassin/rowan.90/lcsv2/13.5_14"
dir_14_14_5 = "/data/poohbah/1/assassin/rowan.90/lcsv2/14_14.5"
dir_14_5_15 = "/data/poohbah/1/assassin/rowan.90/lcsv2/14.5_15"
dir_15_15_5 = "/data/poohbah/1/assassin/rowan.90/lcsv2/15_15.5"

#mask_index_dir(dir_12_12_5)
#mask_index_dir(dir_12_5_13)
#mask_index_dir(dir_13_13_5)
#mask_index_dir(dir_13_5_14)
#mask_index_dir(dir_14_14_5)
#mask_index_dir(dir_14_5_15)
#mask_index_dir(dir_15_15_5)


# per_camera_zero_point() -- this probably isn't necessary since handled by asassn?
# drop_bad_epochs()
# robust_baseline()
# residuals = lc.mag-baseline
# find_excursions()
# multi_cam() -- require dip to appear in >= 2 cameras to be real
# final_confirmation() -- 

# once catalog matched, now need to fit lightcurves to find those with delta mag > 0.3
    # need a better dispersion limit that's hopefully physically motivated, instead of a simple 0.15 sigma dispersion limit

# once lightcurves have been fit, classify the lightcurves. this should probably just be done by eye, so you will need some plotting code to take in the lightcurves and make a plot of them; name the function something like plot_candidates()

# once you've plotted the lightcurves to classify them, you need to make plots characterizing the distribution of new candidates in
    # CMD with new candidates and their categories overplotted
    # SED of significant IR excess dippers
    # interesting (follow-up?) spectra of some of these candidates to further characterize them
    # collect source name, ra, dec, search method, mean g mag, mean G mag, G_BP-G_RP, RV_amp, RUWE, distance of the candidates, search method, number of dips, maximum duration, start of most recent dip, maxiumum depth




# (1) load in csv output by vsx_crossmatch
# (2) create df from csv output
# (3) use csv IDs that are from the index to match to the real lightcurve dat files
# (4) detect dimming events relative to the median and explicitly check for brightening symmetry; ordinary variables show both up and down outliers, while our targets are all downward
    # (4.1) get a baseline and residuals
        # (4.1.1) build a per-camera trend-removed light curve
        # (4.1.2) compute residuals (r_i = m_i - med_smooth(t_i)) between the mag and the smoothed mag and calculate a scatter using mean absolute deivation (mad)
    # (4.2) count the upward and downward features
        # (4.2.1) count downward dips where r_i< -k * MAD for >= L_min consecutive points, allowing small gaps, and measure dip depth/area
        # (4.2.2) count upward sequences where r_i > k*MAD
        # (4.2.3) keep the light curve only if N_down >> N_up AND one contiguous dip meets depth and duration
    # (4.3) mutli-camera confirmation 
        # (4.3.1) require that the same dip window is seen in >= 2 cameras within some timing tolerance
# better bad-data detection before candidate search
    # per-camera zero-point normalization
    # sigma-clip per-night (or per small time bin) usinbg a  weighted approach to drop outliers
    # remove points too close to survey's faint-end SNR limit or bright-end saturation
    # airmass/seeing/background cuts
    # filter with asassn quality flags
    # outlier rejection within a night -- if one night is > 5 sigma from nightly median throw it out
    # drop epochs with
        # unrealistic uncertainties
# lower delta g threshold and expand magnitude range
# periodicity screen?
    # run Lomb-Scargle



from lc_stats import compute_stats, print_summary

#df, summary = compute_stats("42949755092",f"{dir_14_5_15}/lc50_cal")
#print_summary(summary, max_rows=100)

#df, summary = compute_stats("352188462946",f"{dir_14_5_15}/lc50_cal") # NO DAT AT END
#print_summary(summary, max_rows=100)

from lc_process import dip_finder, dip_finder_by_bin, dip_finder_streaming

#df_peaks = dip_finder_by_bin()
#df_peaks = dip_finder_by_bin()
#dip_finder_streaming()


# save df_peaks to csv
#df_peaks.to_csv("/data/poohbah/1/assassin/lenhart/code/calder/calder/output/dip_finder_peaks.csv", index=False)