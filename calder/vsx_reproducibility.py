import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

import glob
import re
from pathlib import Path as p
from tqdm.auto import tqdm
from datetime import datetime

vsx_crossmatch   = "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/asassn_x_vsx_matches_20250920_1415.csv"
already_found     = "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/bj_objects.csv"
asassn_catalog    = "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/asassn_index_masked_concat_cleaned_20250920_1351.csv"
VSX_CLEAN         = "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/vsx_cleaned_20250920_1351.csv"

OUT_MATCH_VSX     = "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/bj_objects_matched.csv"
OUT_UNMATCH_VSX   = "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/bj_objects_unmatched.csv"
OUT_MATCH_ASASSN  = "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/bj_objects_matched_asassn.csv"
OUT_MATCH_VSXCLEAN= "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/bj_objects_x_vsxclean.csv"

TOL_ARCSEC = 2.0

existing = pd.read_csv(already_found)
vsx_cm   = pd.read_csv(vsx_crossmatch)
asassn_df= pd.read_csv(asassn_catalog)
vsx2     = pd.read_csv(VSX_CLEAN)

for df, label, req in [
    (existing, "existing", {"ra_deg","dec_deg"}),
    (vsx_cm,   "vsx",      {"ra_deg","dec_deg"}),
    (asassn_df,"asassn",   {"ra_deg","dec_deg"}),
    (vsx2,     "vsx_clean",{"ra","dec"}),
]:
    if not req.issubset(df.columns):
        raise ValueError(f"{label} CSV missing columns: {sorted(req - set(df.columns))}")

c_existing = SkyCoord(existing["ra_deg"].to_numpy()*u.deg, existing["dec_deg"].to_numpy()*u.deg, frame="icrs")
c_vsx_cm   = SkyCoord(vsx_cm["ra_deg"].to_numpy()*u.deg,   vsx_cm["dec_deg"].to_numpy()*u.deg,   frame="icrs")
c_asassn   = SkyCoord(asassn_df["ra_deg"].to_numpy()*u.deg,asassn_df["dec_deg"].to_numpy()*u.deg,frame="icrs")
c_vsx2     = SkyCoord(vsx2["ra"].to_numpy()*u.deg,         vsx2["dec"].to_numpy()*u.deg,         frame="icrs")

# 1) bj_objects.csv -> asassn_x_vsx_matches_20250920_1415.csv (VSX)
idx_vsx, sep2d_vsx, _ = c_existing.match_to_catalog_sky(c_vsx_cm)
mask_vsx = sep2d_vsx < (TOL_ARCSEC*u.arcsec)
existing["matched_vsx"] = mask_vsx
existing["match_sep_arcsec_vsx"] = sep2d_vsx.arcsec
existing["vsx_cm_index"] = idx_vsx

vsx_cols_to_keep = [
    "targ_idx","vsx_idx","sep_arcsec","asas_sn_id","ra_deg","dec_deg",
    "gaia_id","name","var_flag","class","gaia_mag","gaia_b_mag","gaia_r_mag",
    "gaia_eff_temp","gaia_g_extinc","gaia_var","pstarrs_g_mag","pstarrs_r_mag","pstarrs_i_mag"
]
vsx_cols_to_keep = [c for c in vsx_cols_to_keep if c in vsx_cm.columns]

matched_vsx = existing[existing["matched_vsx"]].copy()
matched_vsx = matched_vsx.merge(
    vsx_cm[vsx_cols_to_keep].reset_index().rename(columns={"index":"vsx_cm_index"}),
    on="vsx_cm_index", how="left", suffixes=("", "_vsx")
)
front_vsx = [c for c in ["category","name","ra_deg","dec_deg","method","matched_vsx","match_sep_arcsec_vsx"] if c in matched_vsx.columns]
matched_vsx = matched_vsx[front_vsx + [c for c in matched_vsx.columns if c not in front_vsx + ["vsx_cm_index"]]]

# 2) bj_objects.csv -> asassn_index_masked_concat_cleaned_20250920_1351.csv (ASASSN)
idx_asn, sep2d_asn, _ = c_existing.match_to_catalog_sky(c_asassn)
mask_asn = sep2d_asn < (TOL_ARCSEC*u.arcsec)
existing["matched_asassn"] = mask_asn
existing["match_sep_arcsec_asassn"] = sep2d_asn.arcsec
existing["asassn_index"] = idx_asn

asassn_cols_to_keep = [
    "targ_idx","asas_sn_id","ra_deg","dec_deg","gaia_id","name","var_flag","class",
    "gaia_mag","gaia_b_mag","gaia_r_mag","gaia_eff_temp","gaia_g_extinc","gaia_var",
    "pstarrs_g_mag","pstarrs_r_mag","pstarrs_i_mag"
]
asassn_cols_to_keep = [c for c in asassn_cols_to_keep if c in asassn_df.columns]

matched_asassn = existing[existing["matched_asassn"]].copy()
matched_asassn = matched_asassn.merge(
    asassn_df[asassn_cols_to_keep].reset_index().rename(columns={"index":"asassn_index"}),
    on="asassn_index", how="left", suffixes=("", "_asassn")
)
front_asn = [c for c in ["category","name","ra_deg","dec_deg","method","matched_asassn","match_sep_arcsec_asassn"] if c in matched_asassn.columns]
matched_asassn = matched_asassn[front_asn + [c for c in matched_asassn.columns if c not in front_asn + ["asassn_index"]]]

# 3) bj_objects.csv -> vsx_cleaned_20250920_1351.csv (VSX-cleaned)
idx_vsx2, sep2d_vsx2, _ = c_existing.match_to_catalog_sky(c_vsx2)
mask_vsx2 = sep2d_vsx2 < (TOL_ARCSEC*u.arcsec)
existing["matched_vsxclean"] = mask_vsx2
existing["match_sep_arcsec_vsxclean"] = sep2d_vsx2.arcsec
existing["vsxclean_index"] = idx_vsx2

vsx2_cols_to_keep = [
    "id_vsx","name","var_flag","ra","dec","class","l_max","mag_max","u_max","mag_band_max",
    "f_min","l_min","mag_min","u_min","mag_band_min","epoch","u_epoch","l_period","period","u_period","spectral_type"
]
vsx2_cols_to_keep = [c for c in vsx2_cols_to_keep if c in vsx2.columns]

matched_vsxclean = existing[existing["matched_vsxclean"]].copy()
matched_vsxclean = matched_vsxclean.merge(
    vsx2[vsx2_cols_to_keep].reset_index().rename(columns={"index":"vsxclean_index"}),
    on="vsxclean_index", how="left", suffixes=("", "_vsxclean")
)
front_clean = [c for c in ["category","name","ra_deg","dec_deg","method","matched_vsxclean","match_sep_arcsec_vsxclean"] if c in matched_vsxclean.columns]
matched_vsxclean = matched_vsxclean[front_clean + [c for c in matched_vsxclean.columns if c not in front_clean + ["vsxclean_index"]]]

# VSX unmatched
unmatched_vsx = existing[~existing["matched_vsx"]].copy()

matched_vsx.to_csv(OUT_MATCH_VSX, index=False)
unmatched_vsx.to_csv(OUT_UNMATCH_VSX, index=False)
matched_asassn.to_csv(OUT_MATCH_ASASSN, index=False)
matched_vsxclean.to_csv(OUT_MATCH_VSXCLEAN, index=False)

total = len(existing)
n_match_vsx     = int(mask_vsx.sum())
n_match_asassn  = int(mask_asn.sum())
n_match_vsxclean= int(mask_vsx2.sum())

print(f"VSX: matched {n_match_vsx} / {total} within {TOL_ARCSEC}\"")
if "category" in existing.columns:
    print(existing.groupby("category")["matched_vsx"].agg(["sum","count"]))

print(f"ASASSN: matched {n_match_asassn} / {total} within {TOL_ARCSEC}\"")
if "category" in existing.columns:
    print(existing.groupby("category")["matched_asassn"].agg(["sum","count"]))

print(f"VSX-cleaned: matched {n_match_vsxclean} / {total} within {TOL_ARCSEC}\"")
if "category" in existing.columns:
    print(existing.groupby("category")["matched_vsxclean"].agg(["sum","count"]))

print("Wrote:",
      OUT_MATCH_VSX, "(VSX matches),",
      OUT_UNMATCH_VSX, "(VSX non-matches),",
      OUT_MATCH_ASASSN, "(ASASSN matches),",
      OUT_MATCH_VSXCLEAN, "(VSX-cleaned matches)")

# 4) bj_objects.csv -> vsx_raw_{stamp}.csv (VSX-raw)
VSX_RAW_PATH = "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/vsx_raw_20250921_0408.csv"

vsx_raw = pd.read_csv(VSX_RAW_PATH)

if not {"ra","dec"}.issubset(vsx_raw.columns):
    raise ValueError("vsx_raw_20250921_0408.csv must contain 'ra' and 'dec' columns")

c_vsxraw = SkyCoord(vsx_raw["ra"].to_numpy()*u.deg,
                    vsx_raw["dec"].to_numpy()*u.deg,
                    frame="icrs")

idx_vsxraw, sep2d_vsxraw, _ = c_existing.match_to_catalog_sky(c_vsxraw)
mask_vsxraw = sep2d_vsxraw < (TOL_ARCSEC*u.arcsec)

existing["matched_vsxraw"] = mask_vsxraw
existing["match_sep_arcsec_vsxraw"] = sep2d_vsxraw.arcsec
existing["vsxraw_index"] = idx_vsxraw

vsxraw_cols_to_keep = [
    "id_vsx","name","var_flag","ra","dec","class","l_max","mag_max","u_max","mag_band_max",
    "f_min","l_min","mag_min","u_min","mag_band_min","epoch","u_epoch","l_period","period","u_period","spectral_type"
]
vsxraw_cols_to_keep = [c for c in vsxraw_cols_to_keep if c in vsx_raw.columns]

matched_vsxraw = existing[existing["matched_vsxraw"]].copy()
matched_vsxraw = matched_vsxraw.merge(
    vsx_raw[vsxraw_cols_to_keep].reset_index().rename(columns={"index":"vsxraw_index"}),
    on="vsxraw_index", how="left", suffixes=("", "_vsxraw")
)

front_raw = [c for c in ["category","name","ra_deg","dec_deg","method","matched_vsxraw","match_sep_arcsec_vsxraw"] if c in matched_vsxraw.columns]
matched_vsxraw = matched_vsxraw[front_raw + [c for c in matched_vsxraw.columns if c not in front_raw + ["vsxraw_index"]]]

OUT_MATCH_VSXRAW = "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/bj_objects_x_vsxraw.csv"
matched_vsxraw.to_csv(OUT_MATCH_VSXRAW, index=False)

n_match_vsxraw = int(mask_vsxraw.sum())
print(f"VSX-raw: matched {n_match_vsxraw} / {len(existing)} within {TOL_ARCSEC}\"")
if "category" in existing.columns:
    print(existing.groupby("category")["matched_vsxraw"].agg(["sum","count"]))
print("Wrote:", OUT_MATCH_VSXRAW, "(VSX-raw matches)")
