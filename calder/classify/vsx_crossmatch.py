from pathlib import Path as p
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from astropy.time import Time
from astropy.coordinates import SkyCoord, Distance
from astropy import units as u
from datetime import datetime

df_all_clean = pd.read_csv("/data/poohbah/1/assassin/lenhart/code/calder/calder/output/asassn_index_masked_concat_cleaned_20250919_154524.csv", low_memory=False)
df_vsx_filt_clean = pd.read_csv("/data/poohbah/1/assassin/lenhart/code/calder/calder/output/vsx_cleaned_20250919_154524.csv", low_memory=False)

# coerce numerics
for c in ["ra_deg","dec_deg","pm_ra","pm_dec"]:
    df_all_clean[c] = pd.to_numeric(df_all_clean[c], errors="coerce")
for c in ["ra","dec"]:
    df_vsx_filt_clean[c] = pd.to_numeric(df_vsx_filt_clean[c], errors="coerce")

# debug
pm_ok = np.isfinite(df_all_clean["pm_ra"].to_numpy()) & np.isfinite(df_all_clean["pm_dec"].to_numpy())
if not pm_ok.all():
    bad = (~pm_ok).sum()
    sample = df_all_clean.loc[~pm_ok, ["asas_sn_id","gaia_id","pm_ra","pm_dec"]].head(10)
    raise ValueError(f"{bad} row(s) missing/invalid proper motion.\nSample:\n{sample}")

# epochs
t_gaia  = Time(2016.0, format="jyear")   # Gaia DR3 ref epoch
t_j2000 = Time(2000.0, format="jyear")   # VSX (J2000)

ra_deg_arr  = df_all_clean['ra_deg'].to_numpy(dtype=float)
dec_deg_arr = df_all_clean['dec_deg'].to_numpy(dtype=float)
pm_ra_arr   = df_all_clean['pm_ra'].to_numpy(dtype=float)
pm_dec_arr  = df_all_clean['pm_dec'].to_numpy(dtype=float)

c_asassn = SkyCoord(
                    ra=ra_deg_arr * u.deg,
                    dec=dec_deg_arr * u.deg,
                    pm_ra_cosdec=pm_ra_arr * u.mas/u.yr,
                    pm_dec=pm_dec_arr * u.mas/u.yr,
                    obstime=Time(2016.0, format="jyear"),
).apply_space_motion(new_obstime=Time(2000.0, format="jyear"))

vsx_ra_arr  = df_vsx_filt_clean["ra"].to_numpy(dtype=float)
vsx_dec_arr = df_vsx_filt_clean["dec"].to_numpy(dtype=float)

c_vsx = SkyCoord(ra=vsx_ra_arr * u.deg,
                dec=vsx_dec_arr * u.deg)

# nearest neighbor in VSX for each ASAS-SN target
idx_vsx, sep2d, _ = c_asassn.match_to_catalog_sky(c_vsx)

match_radius = 3 * u.arcsec  # 3 arcsec
mask = sep2d < match_radius

targ_idx = np.where(mask)[0]                # indices into df_all_clean
vsx_idx  = idx_vsx[mask]                    # indices into df_vsx_filt_clean
sep_arc  = sep2d[mask].to(u.arcsec).value

df_pairs = pd.DataFrame({
    "targ_idx": targ_idx,
    "vsx_idx":  vsx_idx,
    "sep_arcsec": sep_arc,
})

out = (df_pairs
       .merge(df_all_clean, left_on="targ_idx", right_index=True, how="left")
       .merge(df_vsx_filt_clean, left_on="vsx_idx", right_index=True, how="left",
              suffixes=("_targ","_vsx")))

stamp = datetime.now().strftime("%Y%m%d_%H%M")

# outputting timestamped crossmatched csv
out2 = p.cwd() / f"/data/poohbah/1/assassin/lenhart/code/calder/calder/output/asassn_x_vsx_matches_{stamp}.csv"
out.to_csv(out2, index=False)
