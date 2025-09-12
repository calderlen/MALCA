
# need to get the vsx catalog
# get the asassn id of each surviving lc in the vsx catalog
# load the asassn_id's into a a list
# then for all the asassn_id's, append the v and g mag lightcurves to them
# probably need to rewrite read_lightcurve to account for this

df_v, df_g = read_lightcurve(asassn_id, path)

# calculate robust_baseline for each of the lightcurves


def robust_baseline(df_g):
