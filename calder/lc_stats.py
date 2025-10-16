"""
Outputs:
- Core timing/cadence stats (including 3-day exposure metrics and largest gaps)
- Photometric stats (weighted/unweighted/clipped/MAD/IQR/percentiles)
- Quality & error stats (SNR dist, fractions by good/saturated)
- Variability diagnostics (reduced chisq, von Neumann ratio, lag-1 autocorr, trend slope)
- Nightly/seasonal coverage & duty cycle
- Per-camera / per-field / per-band usage + offsets and scatter
"""

import sys, io, argparse, math
from collections import OrderedDict
import numpy as np
import pandas as pd
from utils import read_lc_dat

# helpers
def weighted_mean(x, w):
    w = np.asarray(w, float)
    x = np.asarray(x, float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return np.nan, np.nan
    w = w[mask]; x = x[mask]
    mu = np.sum(w * x) / np.sum(w)
    var = np.sum(w * (x - mu)**2) / np.sum(w)
    # Standard error of weighted mean (approx)
    sem = math.sqrt(1.0 / np.sum(w))
    return mu, sem

def robust_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    # 1.4826 * MAD (median absolute deviation)
    return 1.4826 * np.median(np.abs(x - np.median(x)))

def pct(x, q):
    return float(np.nanpercentile(x, q)) if len(x) else np.nan

def reduced_chisq(y, yerr, model_value):
    y  = np.asarray(y, float)
    ye = np.asarray(yerr, float)
    m  = np.asarray(model_value, float)
    mask = np.isfinite(y) & np.isfinite(ye) & (ye > 0)
    if mask.sum() < 2:
        return np.nan
    y = y[mask]; ye = ye[mask]
    chi2 = np.sum(((y - m)/ye)**2)
    dof = y.size - 1
    return chi2 / dof if dof > 0 else np.nan

def von_neumann_ratio(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3: return np.nan
    diffs = np.diff(x)
    num = np.mean(diffs**2)
    den = np.var(x, ddof=1)
    return num/den if den > 0 else np.nan

def lag1_autocorr(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3: return np.nan
    x0 = x[:-1] - x[:-1].mean()
    x1 = x[1:]  - x[1:].mean()
    den = np.sqrt(np.sum(x0**2) * np.sum(x1**2))
    return float(np.sum(x0 * x1) / den) if den > 0 else np.nan

def linear_trend(x, y):
    # returns slope, intercept, r^2; robust to NaNs
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2: return np.nan, np.nan, np.nan
    x = x[mask]; y = y[mask]
    p = np.polyfit(x, y, 1)
    yhat = np.polyval(p, x)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return float(p[0]), float(p[1]), float(r2)

def compute_stats(asassn_id, path, use_only_good=True, drop_dupes=True, use_g=True):
    
    df_g, df_v = read_lc_dat(asassn_id, path)

    if use_g:
        if df_g.empty and not df_v.empty:
            print(f"[warn] {asassn_id}: g-band empty; using V-band instead.")
            df = df_v.copy()
        else:
            df = df_g.copy()
    else:
        if df_v.empty and not df_g.empty:
            print(f"[warn] {asassn_id}: V-band empty; using g-band instead.")
            df = df_g.copy()
        else:
            df = df_v.copy()
    
    cols = ["JD",
            "mag",
            "error",
            "good_bad",
            "camera#",
            "v_g_band",
            "saturated",
            "camera_name",
            "field"]
    
    df = df.copy()
    df.columns = cols[:len(df.columns)] + [f"extra_{i}" for i in range(len(df.columns)-len(cols))]

    for c in ["JD","mag","error","good_bad","camera#","v_g_band","saturated"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["JD","mag","error"]).sort_values("JD").reset_index(drop=True)

    # drop duplicate JDs
    if drop_dupes:
        df = df[~df["JD"].duplicated(keep="first")].reset_index(drop=True)

    # filtering
    base_n = len(df)
    if use_only_good:
        df = df[(df["good_bad"] == 1) & (df["saturated"] == 0)].reset_index(drop=True)
    kept_n = len(df)

    # time axis in days since first exposure (JD is in days already)
    jd0 = df["JD"].iloc[0]
    df["t_days"] = df["JD"] - jd0

    # cadence (diffs)
    df["dt_days"] = df["t_days"].diff()
    dt = df["dt_days"].iloc[1:].values

    # 3-day exposures: binned and rolling
    df["bin3d"] = np.floor(df["t_days"]/3).astype("Int64")
    per3d_binned = df.groupby("bin3d").size().rename("count").reset_index()

    # sliding window (previous 3 days)
    t = df["t_days"].values
    counts_rolling = np.empty(len(t), dtype=int)
    j = 0
    for i in range(len(t)):
        while j < i and t[j] < t[i] - 3.0:
            j += 1
        counts_rolling[i] = i - j + 1
    df["count_in_prev_3d"] = counts_rolling

    # exposures per night & duty cycle
    # treat "night" as floor(JD)
    df["night"] = np.floor(df["JD"]).astype(int)
    per_night = df.groupby("night").size().rename("n_exp").reset_index()
    nights_observed = len(per_night)
    span_days = df["t_days"].iloc[-1] - df["t_days"].iloc[0]
    # duty cycle ~ fraction of nights with >=1 obs over total span in nights
    total_nights_in_span = int(np.floor(df["JD"].iloc[-1]) - np.floor(df["JD"].iloc[0]) + 1)
    duty_cycle = nights_observed / total_nights_in_span if total_nights_in_span > 0 else np.nan

    # "seasons": long gaps (> 30 days) split segments
    gap_threshold = 30.0
    cut_idx = np.where(dt > gap_threshold)[0]  # indices of large gaps (relative to df[1:])
    segments = []
    start = 0
    for c in cut_idx:
        end = c + 1
        segments.append((start, end))
        start = end
    segments.append((start, len(df)))
    seasons = []
    for (a,b) in segments:
        sub = df.iloc[a:b]
        seasons.append({
            "start_JD": float(sub["JD"].iloc[0]),
            "end_JD": float(sub["JD"].iloc[-1]),
            "n_obs": int(len(sub)),
            "span_days": float(sub["t_days"].iloc[-1] - sub["t_days"].iloc[0]),
        })

    # largest gaps (top 10)
    gaps = df.loc[df["dt_days"].nlargest(10).index, ["JD","t_days","dt_days"]].copy()
    gaps["JD_prev"] = df["JD"].shift(1).loc[gaps.index].values
    gaps = gaps.sort_values("dt_days", ascending=False).reset_index(drop=True)

    # photometric stats
    mag = df["mag"].values
    merr = df["error"].values
    w = 1.0 / np.where(merr>0, merr**2, np.nan)
    mean_mag = float(np.nanmean(mag))
    median_mag = float(np.nanmedian(mag))
    wmean_mag, wsem_mag = weighted_mean(mag, w)
    std_mag = float(np.nanstd(mag, ddof=1))
    rsig_mag = float(robust_sigma(mag))
    iqr_mag = float(np.nanpercentile(mag, 75) - np.nanpercentile(mag, 25))
    p05, p16, p84, p95 = [pct(mag, q) for q in [5,16,84,95]]

    # clipped stats)
    med = np.nanmedian(mag)
    rs = robust_sigma(mag)
    if np.isfinite(rs) and rs > 0:
        clip_mask = np.abs(mag - med) <= 3*rs
    else:
        clip_mask = np.isfinite(mag)
    mag_clip = mag[clip_mask]
    mean_clip = float(np.nanmean(mag_clip))
    std_clip  = float(np.nanstd(mag_clip, ddof=1))
    n_outliers = int(np.size(mag) - np.size(mag_clip))

    # error and SNR stats (SNR ~= 1.0857 / err )
    snr = 1.0857 / merr
    err_stats = {
        "error_mean": float(np.nanmean(merr)),
        "error_median": float(np.nanmedian(merr)),
        "error_p05": pct(merr,5),
        "error_p95": pct(merr,95),
        "snr_median": float(np.nanmedian(snr)),
        "snr_p05": pct(snr,5),
        "snr_p95": pct(snr,95),
    }

    # variability diagnostics vs constant model
    model = wmean_mag if np.isfinite(wmean_mag) else median_mag
    rchisq = float(reduced_chisq(mag, merr, model))
    vnr    = float(von_neumann_ratio(mag))
    ac1    = float(lag1_autocorr(mag))
    slope_d_per_day, intercept, r2 = linear_trend(df["t_days"].values, mag)
    slope_d_per_year = slope_d_per_day * 365.25 if np.isfinite(slope_d_per_day) else np.nan

    # per camera/field/band usage + offsets and scatter
    global_med = median_mag
    def per_group_stats(group, name):
        out = group.agg(
            n_obs=("mag","size"),
            med_mag=("mag","median"),
            mad_sigma=("mag", lambda x: robust_sigma(x)),
            mean_err=("error","mean"),
        ).reset_index()
        out["offset_vs_global_med"] = out["med_mag"] - global_med
        out = out.sort_values("n_obs", ascending=False).reset_index(drop=True)
        out.attrs["group_name"] = name
        return out

    by_camera  = per_group_stats(df.groupby("camera_name"), "camera")
    by_field   = per_group_stats(df.groupby("field"), "field")
    by_band    = per_group_stats(df.groupby("v_g_band"), "v_g_band")
    by_camfld  = per_group_stats(df.groupby(["camera_name","field"]), "camera_field")

    # cadence distributions per camera
    def per_cam_cadence(d):
        d = d.sort_values("t_days")
        dt = d["t_days"].diff().iloc[1:].values
        return pd.Series({
            "dt_median": np.nanmedian(dt) if dt.size else np.nan,
            "dt_mean":   np.nanmean(dt) if dt.size else np.nan,
            "dt_p05":    pct(dt,5) if dt.size else np.nan,
            "dt_p95":    pct(dt,95) if dt.size else np.nan,
        })
    cadence_by_camera = df.groupby("camera_name").apply(per_cam_cadence).reset_index()

    # nightly stats table (exposures and median mag per night)
    nightly = df.groupby("night").agg(
        n_exp=("mag","size"),
        med_mag=("mag","median"),
        med_err=("error","median"),
    ).reset_index()

    # package summary
    summary = OrderedDict([
        ("file_points_total", int(base_n)),
        ("file_points_kept_after_filter", int(kept_n)),
        ("jd_start", float(df["JD"].iloc[0])),
        ("jd_end", float(df["JD"].iloc[-1])),
        ("time_span_days", float(span_days)),
        ("n_unique_nights", int(nights_observed)),
        ("duty_cycle_fraction", float(duty_cycle)),
        ("cadence_mean_dt_days", float(np.nanmean(dt)) if dt.size else np.nan),
        ("cadence_median_dt_days", float(np.nanmedian(dt)) if dt.size else np.nan),
        ("cadence_p05_dt_days", pct(dt,5) if dt.size else np.nan),
        ("cadence_p95_dt_days", pct(dt,95) if dt.size else np.nan),
        ("largest_gaps_top10_days", gaps[["JD_prev","JD","dt_days"]].rename(columns={"dt_days":"gap_days"})),
        ("exposures_per_3d_binned", per3d_binned),
        ("rolling_count_prev_3d_at_each_obs", df[["JD","t_days","count_in_prev_3d"]]),
        ("photometry_mean_mag", mean_mag),
        ("photometry_median_mag", median_mag),
        ("photometry_weighted_mean_mag", float(wmean_mag)),
        ("photometry_weighted_mean_sem", float(wsem_mag)),
        ("photometry_std_mag", std_mag),
        ("photometry_robust_sigma_mag", rsig_mag),
        ("photometry_IQR_mag", iqr_mag),
        ("photometry_p05_mag", p05),
        ("photometry_p16_mag", p16),
        ("photometry_p84_mag", p84),
        ("photometry_p95_mag", p95),
        ("clipped_mean_mag_3sigma_about_median", mean_clip),
        ("clipped_std_mag_3sigma_about_median", std_clip),
        ("n_outliers_removed_robust_3sigma", n_outliers),
        ("error_and_snr_stats", err_stats),
        ("variability_reduced_chi2_vs_constant", rchisq),
        ("variability_von_neumann_ratio", vnr),
        ("variability_lag1_autocorr", ac1),
        ("trend_slope_mag_per_day", slope_d_per_day),
        ("trend_slope_mag_per_year", slope_d_per_year),
        ("trend_r2", r2),
        ("by_camera", by_camera),
        ("by_field", by_field),
        ("by_band", by_band),
        ("by_camera_field", by_camfld),
        ("cadence_by_camera", cadence_by_camera),
        ("nightly_table", nightly),
        ("seasons", pd.DataFrame(seasons)),
    ])

    return df, summary

def print_summary(summary, max_rows=10):
    def headframe(x, n=max_rows):
        return x.head(n).to_string(index=False)

    print("\n=== CORE TIMING ===")
    print(f"JD start/end: {summary['jd_start']:.6f} → {summary['jd_end']:.6f}  | span: {summary['time_span_days']:.2f} d")
    print(f"Unique nights: {summary['n_unique_nights']}  | Duty cycle: {summary['duty_cycle_fraction']:.3f}")

    print("\n=== CADENCE (Δt in days) ===")
    print(f"mean={summary['cadence_mean_dt_days']:.3f}  median={summary['cadence_median_dt_days']:.3f}  p05={summary['cadence_p05_dt_days']:.3f}  p95={summary['cadence_p95_dt_days']:.3f}")
    print("Top 10 gaps (days):")
    print(headframe(summary["largest_gaps_top10_days"][["JD_prev","JD","gap_days"]]))

    print("\n=== EXPOSURES PER 3 DAYS ===")
    print("Non-overlapping bins (first rows):")
    print(headframe(summary["exposures_per_3d_binned"]))
    print("Rolling count at each obs (first rows):")
    print(headframe(summary["rolling_count_prev_3d_at_each_obs"]))

    print("\n=== PHOTOMETRY ===")
    print(f"mean={summary['photometry_mean_mag']:.6f}  median={summary['photometry_median_mag']:.6f}  wmean={summary['photometry_weighted_mean_mag']:.6f}±{summary['photometry_weighted_mean_sem']:.6f}")
    print(f"std={summary['photometry_std_mag']:.6f}  robust_sigma={summary['photometry_robust_sigma_mag']:.6f}  IQR={summary['photometry_IQR_mag']:.6f}")
    print(f"p05={summary['photometry_p05_mag']:.6f}  p16={summary['photometry_p16_mag']:.6f}  p84={summary['photometry_p84_mag']:.6f}  p95={summary['photometry_p95_mag']:.6f}")
    print(f"clipped mean={summary['clipped_mean_mag_3sigma_about_median']:.6f}  clipped std={summary['clipped_std_mag_3sigma_about_median']:.6f}  outliers={summary['n_outliers_removed_robust_3sigma']}")

    es = summary["error_and_snr_stats"]
    print("\n=== ERRORS / SNR ===")
    print(f"error: mean={es['error_mean']:.6f}  median={es['error_median']:.6f}  p05={es['error_p05']:.6f}  p95={es['error_p95']:.6f}")
    print(f"SNR: median={es['snr_median']:.2f}  p05={es['snr_p05']:.2f}  p95={es['snr_p95']:.2f}")

    print("\n=== VARIABILITY / TREND ===")
    print(f"reduced χ² vs constant={summary['variability_reduced_chi2_vs_constant']:.3f}  | von Neumann={summary['variability_von_neumann_ratio']:.3f}  | lag-1 ρ={summary['variability_lag1_autocorr']:.3f}")
    print(f"trend slope={summary['trend_slope_mag_per_day']:.6e} mag/day ({summary['trend_slope_mag_per_year']:.6e} mag/yr),  R²={summary['trend_r2']:.3f}")

    print("\n=== BY CAMERA (top) ===")
    print(headframe(summary["by_camera"]))
    print("\n=== BY FIELD (top) ===")
    print(headframe(summary["by_field"]))
    print("\n=== BY BAND (all) ===")
    print(headframe(summary["by_band"]))
    print("\n=== BY CAMERA+FIELD (top) ===")
    print(headframe(summary["by_camera_field"]))

    print("\n=== CADENCE BY CAMERA ===")
    print(headframe(summary["cadence_by_camera"]))

    print("\n=== NIGHTLY TABLE (first rows) ===")
    print(headframe(summary["nightly_table"]))

    print("\n=== SEASONS (gap > 30 d defines a new season) ===")
    print(summary["seasons"].to_string(index=False))

def load_dat(path, has_header=False):
    # auto-handle comments and variable whitespace
    names = ["JD",
            "mag",
            "error",
            "good_bad",
            "camera#",
            "v_g_band",
            "saturated",
            "camera_name",
            "field"]    
    kw = dict(sep=r"\s+", comment="#", engine="python")
    if has_header:
        df = pd.read_csv(path, **kw)
        if len(df.columns) < len(names):
            # pad names if fewer columns
            df.columns = names[:len(df.columns)]
    else:
        df = pd.read_csv(path, header=None, names=names, **kw)

    return df

def main():
    ap = argparse.ArgumentParser(description="Compute rich stats for a photometry .dat file.")
    ap.add_argument("path", help="path to .dat file")
    ap.add_argument("--include-all", action="store_true", help="do NOT filter by good_bad==1 & saturated==0")
    ap.add_argument("--keep-dupes",   action="store_true", help="keep duplicate JD rows instead of dropping")
    ap.add_argument("--has-header",   action="store_true", help="file has a header row")
    args = ap.parse_args()

    df = load_dat(args.path, has_header=args.has_header)
    df2, summary = compute_stats(
        df,
        use_only_good=not args.include_all,
        drop_dupes=not args.keep_dupes
    )
    print_summary(summary)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python stats.py yourfile.dat [--include-all] [--keep-dupes] [--has-header]")
    else:
        main()
