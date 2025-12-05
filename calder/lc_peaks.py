from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
from astropy.stats import biweight_location, biweight_scale
from scipy.signal import find_peaks

from lc_utils import read_lc_dat, read_lc_raw, match_index_to_lc
from lc_baseline import per_camera_trend_baseline
from lc_dips import clean_lc, _biweight_gaussian_metrics

MAG_BINS = ["12_12.5", "12.5_13", "13_13.5", "13.5_14", "14_14.5", "14.5_15"]


def _compute_biweight_delta_peaks(df, *, mag_col="mag", t_col="JD", err_col="error", biweight_c=6.0, eps=1e-6):
    """
    Peak-oriented version of biweight delta where brightenings are positive.

    Uses (R - mag) so that microlensing-like peaks (smaller magnitudes) produce
    positive delta values suitable for peak finding.
    """
    mag = np.asarray(df[mag_col], float) if mag_col in df.columns else np.array([], float)
    jd = np.asarray(df[t_col], float) if t_col in df.columns else np.array([], float)
    if err_col in df.columns:
        err = np.asarray(df[err_col], float)
    else:
        err = np.full_like(mag, np.nan, dtype=float)

    finite_m = np.isfinite(mag)
    R = float(biweight_location(mag[finite_m], c=biweight_c)) if finite_m.any() else np.nan
    S = float(biweight_scale(mag[finite_m], c=biweight_c)) if finite_m.any() else 0.0
    if not np.isfinite(S) or S < 0:
        S = 0.0

    err2 = np.where(np.isfinite(err), err**2, 0.0)
    denom = np.sqrt(err2 + S**2)
    denom = np.where(denom > 0, denom, eps)
    delta = (R - mag) / denom
    return delta, jd, R


def process_record_microlensing(
    record: dict,
    *,
    peak_kwargs: dict | None = None,
) -> dict:
    """
    Process a single light curve record looking for microlensing-like peaks
    using the biweight-delta (brightenings positive) series.
    """
    peak_kwargs = dict(peak_kwargs or {})
    sigma_threshold = float(peak_kwargs.get("sigma_threshold", 3.0))

    asn = record["asas_sn_id"]
    dfg, dfv = read_lc_dat(asn, record["lc_dir"])
    raw_df = read_lc_raw(asn, record["lc_dir"])

    jd_first = np.nan
    jd_last = np.nan

    def _analyze_band(df_band):
        if df_band.empty or len(df_band) < 30:
            return {
                "n": 0,
                "R": np.nan,
                "peaks_idx": [],
                "peaks_jd": [],
                "fit": {
                    "total_score": 0.0,
                    "best_score": 0.0,
                    "scores": [],
                    "fwhm": [],
                    "delta_peak": [],
                    "n_det": [],
                    "chi2_red": [],
                },
            }

        df_band = clean_lc(df_band).sort_values("JD").reset_index(drop=True)
        delta, jd, R = _compute_biweight_delta_peaks(df_band, **peak_kwargs)
        peaks, _ = find_peaks(
            np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0),
            height=sigma_threshold,
            distance=int(peak_kwargs.get("distance", 50)),
        )

        fit = _biweight_gaussian_metrics(
            delta,
            jd,
            df_band["error"].to_numpy(float) if "error" in df_band.columns else None,
            peaks,
            sigma_threshold,
        )

        return {
            "n": len(peaks),
            "R": R,
            "peaks_idx": np.asarray(peaks, dtype=int).tolist(),
            "peaks_jd": df_band["JD"].values[np.asarray(peaks, dtype=int)].tolist() if len(peaks) else [],
            "fit": fit,
        }

    g_res = _analyze_band(dfg)
    v_res = _analyze_band(dfv)

    if not dfg.empty:
        jd_first = float(dfg["JD"].iloc[0])
        jd_last = float(dfg["JD"].iloc[-1])
    if not dfv.empty:
        if np.isnan(jd_first):
            jd_first = float(dfv["JD"].iloc[0])
        if np.isnan(jd_last):
            jd_last = float(dfv["JD"].iloc[-1])

    row = {
        "mag_bin": record.get("mag_bin"),
        "asas_sn_id": asn,
        "index_num": record.get("index_num"),
        "index_csv": record.get("index_csv"),
        "lc_dir": record.get("lc_dir"),
        "dat_path": record.get("dat_path"),
        "raw_path": os.path.join(record.get("lc_dir"), f"{asn}.raw"),
        "g_n_peaks": g_res["n"],
        "g_biweight_R": g_res["R"],
        "g_peaks_idx": g_res["peaks_idx"],
        "g_peaks_jd": g_res["peaks_jd"],
        "g_total_score": g_res["fit"]["total_score"],
        "g_best_score": g_res["fit"]["best_score"],
        "g_scores": g_res["fit"]["scores"],
        "g_fwhm": g_res["fit"]["fwhm"],
        "g_delta_peak": g_res["fit"]["delta_peak"],
        "g_n_det": g_res["fit"]["n_det"],
        "g_chi2_red": g_res["fit"]["chi2_red"],
        "v_n_peaks": v_res["n"],
        "v_biweight_R": v_res["R"],
        "v_peaks_idx": v_res["peaks_idx"],
        "v_peaks_jd": v_res["peaks_jd"],
        "v_total_score": v_res["fit"]["total_score"],
        "v_best_score": v_res["fit"]["best_score"],
        "v_scores": v_res["fit"]["scores"],
        "v_fwhm": v_res["fit"]["fwhm"],
        "v_delta_peak": v_res["fit"]["delta_peak"],
        "v_n_det": v_res["fit"]["n_det"],
        "v_chi2_red": v_res["fit"]["chi2_red"],
        "jd_first": jd_first,
        "jd_last": jd_last,
        "n_rows_g": int(len(dfg)) if not dfg.empty else 0,
        "n_rows_v": int(len(dfv)) if not dfv.empty else 0,
    }

    if not raw_df.empty:
        scatter_vals = (raw_df["sig1_high"] - raw_df["sig1_low"]).to_numpy(dtype=float)
        finite_scatter = scatter_vals[np.isfinite(scatter_vals)]
        row["raw_robust_scatter"] = float(np.nanmedian(finite_scatter)) if finite_scatter.size > 0 else np.nan
    else:
        row["raw_robust_scatter"] = np.nan

    return row


def microlensing_peak_finder(
    *,
    index_path="/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked/",
    lc_path="/data/poohbah/1/assassin/rowan.90/lcsv2",
    mag_bins=MAG_BINS,
    id_column="asas_sn_id",
    out_dir="./results_microlensing",
    out_format="csv",
    n_workers=8,
    chunk_size=250000,
    max_inflight=None,
    peak_kwargs: dict | None = None,
    collected_rows: list | None = None,
) -> pd.DataFrame | None:
    """
    Run microlensing peak search across magnitude bins, writing CSV/Parquet.
    """
    peak_kwargs = dict(peak_kwargs or {})
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"microlensing_peaks.{out_format}")
    if max_inflight is None:
        max_inflight = max(4, n_workers)

    rows_buffer: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for b in mag_bins:
            pbar = tqdm(total=1, desc=f"bin {b}", leave=False)

            def flush_if_needed(force: bool = False):
                if not rows_buffer:
                    return
                if len(rows_buffer) >= chunk_size or force:
                    if out_format == "parquet":
                        pd.DataFrame(rows_buffer).to_parquet(out_path, index=False)
                    else:
                        mode = "a" if os.path.exists(out_path) else "w"
                        header = not os.path.exists(out_path)
                        pd.DataFrame(rows_buffer).to_csv(out_path, index=False, mode=mode, header=header)
                    rows_buffer.clear()

            pending = set()
            scheduled = 0

            def drain_some(all_pending):
                done_now = 0
                done = []
                for fut in as_completed(all_pending, timeout=0.1):
                    done.append(fut)
                for fut in done:
                    row = fut.result()
                    rows_buffer.append(row)
                    if collected_rows is not None:
                        collected_rows.append(row)
                    all_pending.remove(fut)
                    done_now += 1
                    flush_if_needed()
                return done_now

            record_iter = match_index_to_lc(
                index_path=index_path,
                lc_path=lc_path,
                mag_bins=[b],
                id_column=id_column,
            )

            for rec in record_iter:
                if not rec.get("found", False):
                    continue
                pending.add(
                    ex.submit(
                        process_record_microlensing,
                        rec,
                        peak_kwargs=peak_kwargs,
                    )
                )
                scheduled += 1
                pbar.total = scheduled
                pbar.refresh()
                if len(pending) >= max_inflight:
                    done = drain_some(pending)
                    pbar.update(done)

            while pending:
                done = drain_some(pending)
                pbar.update(done)

            pbar.close()
            flush_if_needed(force=True)

    if collected_rows is not None:
        return pd.DataFrame(collected_rows)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run microlensing peak finder across bins.")
    parser.add_argument("--mag-bin", dest="mag_bins", action="append", choices=MAG_BINS, help="Specify bins to run; omit to process all.")
    parser.add_argument("--out-dir", default="./results_microlensing")
    parser.add_argument("--format", choices=("parquet", "csv"), default="csv")
    parser.add_argument("--n-workers", type=int, default=8, help="Parallel processes")
    parser.add_argument("--chunk-size", type=int, default=250000, help="Rows per CSV flush")
    args = parser.parse_args()
    bins = args.mag_bins or MAG_BINS
    microlensing_peak_finder(
        mag_bins=bins,
        out_dir=args.out_dir,
        out_format=args.format,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
    )

