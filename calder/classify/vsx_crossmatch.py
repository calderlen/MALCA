from pathlib import Path as p
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from datetime import datetime

df_all_clean = pd.read_csv("/data/poohbah/1/assassin/lenhart/code/calder/calder/output/asassn_index_masked_concat_cleaned_20250919_154524.csv", low_memory=False)
df_vsx_filt_clean = pd.read_csv("/data/poohbah/1/assassin/lenhart/code/calder/calder/output/vsx_cleaned_20250919_154524.csv", low_memory=False)


# coerce numerics
for c in ["ra_deg","dec_deg","pm_ra","pm_dec"]:
    df_all_clean[c] = pd.to_numeric(df_all_clean[c], errors="coerce")
for c in ["ra","dec"]:
    df_vsx_filt_clean[c] = pd.to_numeric(df_vsx_filt_clean[c], errors="coerce")

pm_ok = df_all_clean["pm_ra"].notna() & df_all_clean["pm_dec"].notna()
pm_ok &= np.isfinite(df_all_clean["pm_ra"].values) & np.isfinite(df_all_clean["pm_dec"].values)

if not pm_ok.all():
    bad = (~pm_ok).sum()
    # show a few offending rows to fix upstream
    sample = df_all_clean.loc[~pm_ok, ["asassn_id","gaia_id","pm_ra","pm_dec"]].head(10)
    raise ValueError(f"{bad} row(s) missing/invalid proper motion.\nSample:\n{sample}")

# epochs
t_gaia  = Time(2016.0, format="jyear")  # Gaia DR3 ref
t_j2000 = Time(2000.0, format="jyear")  # VSX ref

# extract RA and Dec from asas-sn and vsx
c_asassn = SkyCoord(ra=df_all_clean['ra_deg'].values*u.deg,
                    dec=df_all_clean['dec_deg'].values*u.deg,
                    pm_ra_cosdec=df_all_clean['pm_ra'].values*u.mas/u.yr,
                    pm_dec=df_all_clean["pm_dec"].values*u.mas/u.yr,
                    obstime=t_gaia.apply_space_motion(new_obstime=t_j2000)
                    ).applyspacemotion(newtime=t_j2000)

c_vsx    = SkyCoord(ra=df_vsx_filt_clean["ra"].values*u.deg,
                    dec=df_vsx_filt_clean["dec"].values*u.deg)

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