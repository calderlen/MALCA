from __future__ import annotations

import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from malca.stv.events import score_lightcurve
from malca.core.baseline import per_camera_gp_baseline
from malca.core.utils import read_lc_dat2
from malca.config import (
    LOGBF_THRESHOLD_DIP,
    LOGBF_THRESHOLD_JUMP,
    SIGNIFICANCE_THRESHOLD,
    P_POINTS,
    MAG_POINTS,
    RUN_MIN_POINTS,
    RUN_MAX_GAP_POINTS,
    FP_TRIALS_PER_FAMILY,
    INJECTION_SEED,
    DEFAULT_OUTPUT_DIR,
)
from malca.io.table_io import read_parquet_table, write_parquet_table


def _build_detection_kwargs(trigger_mode: str) -> dict:
    return dict(
        trigger_mode=trigger_mode,
        logbf_threshold_dip=LOGBF_THRESHOLD_DIP,
        logbf_threshold_jump=LOGBF_THRESHOLD_JUMP,
        significance_threshold=SIGNIFICANCE_THRESHOLD,
        p_points=P_POINTS,
        mag_points=MAG_POINTS,
        run_min_points=RUN_MIN_POINTS,
        max_gap_points=RUN_MAX_GAP_POINTS,
        run_max_gap_days=None,
        run_min_duration_days=None,
        compute_event_prob=True,
        baseline_func=per_camera_gp_baseline,
        baseline_kwargs=dict(sigma_floor=0.03),
    )


def _default_detection_func(df: pd.DataFrame, detection_kwargs: dict) -> dict:
    res = score_lightcurve(df, **detection_kwargs)
    dip = res["dip"]
    jump = res["jump"]
    return {
        "detected": bool(dip.get("significant", False)),
        "dip_significant": bool(dip.get("significant", False)),
        "jump_significant": bool(jump.get("significant", False)),
        "dip_trigger_max": float(dip.get("trigger_max", np.nan)),
        "jump_trigger_max": float(jump.get("trigger_max", np.nan)),
    }


def _inject_camera_offset(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    cams = out["camera#"].dropna().unique().tolist() if "camera#" in out.columns else []
    if not cams:
        return out
    bad_cam = rng.choice(cams)
    out.loc[out["camera#"] == bad_cam, "mag"] += rng.uniform(0.05, 0.35)
    return out


def _inject_semiregular(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    t = out["JD"].to_numpy(dtype=float)
    period = rng.uniform(20.0, 180.0)
    amp = rng.uniform(0.05, 0.4)
    out["mag"] = out["mag"].to_numpy(dtype=float) + amp * np.sin(2.0 * np.pi * (t - t.min()) / period)
    return out


def _inject_rcb_like(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    t = out["JD"].to_numpy(dtype=float)
    if len(t) == 0:
        return out
    center = rng.uniform(t.min(), t.max())
    width = rng.uniform(30.0, 250.0)
    depth = rng.uniform(0.4, 2.0)
    profile = np.exp(-0.5 * ((t - center) / max(width, 1e-3)) ** 2)
    out["mag"] = out["mag"].to_numpy(dtype=float) + depth * profile
    return out


def _inject_camera_cluster(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    if n < 10:
        return out
    idx = rng.choice(np.arange(n), size=max(3, int(0.1 * n)), replace=False)
    out.loc[out.index[idx], "mag"] += rng.normal(0.0, 0.6, size=len(idx))
    return out


def _inject_cv_outburst(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    t = out["JD"].to_numpy(dtype=float)
    if len(t) == 0:
        return out
    t_0 = rng.uniform(t.min(), t.max())
    tau_rise = rng.uniform(0.5, 5.0)
    tau_decay = rng.uniform(5.0, 40.0)
    amp = rng.uniform(-6.0, -2.0)  # negative because brightening

    dt = t - t_0
    profile = np.where(
        dt >= 0,
        np.exp(-dt / tau_decay),
        np.exp(dt / tau_rise)
    )
    out["mag"] = out["mag"].to_numpy(dtype=float) + amp * profile
    return out


def _inject_drw_agn(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    t = out["JD"].to_numpy(dtype=float)
    if len(t) == 0:
        return out

    # Ensure time is sorted for autoregressive generation
    sort_idx = np.argsort(t)
    t_sorted = t[sort_idx]

    tau = rng.uniform(50.0, 600.0)
    sf_inf = rng.uniform(0.1, 1.0)

    s = np.zeros(len(t_sorted))
    s[0] = rng.normal(0, sf_inf / np.sqrt(2.0))

    for i in range(1, len(t_sorted)):
        dt = t_sorted[i] - t_sorted[i-1]
        decay = np.exp(-dt / tau)
        sigma_drive = (sf_inf / np.sqrt(2.0)) * np.sqrt(max(0.0, 1.0 - decay**2))
        s[i] = decay * s[i-1] + rng.normal(0, sigma_drive)

    # Unsort to match original dataframe index order
    unsort_idx = np.argsort(sort_idx)
    s_original_order = s[unsort_idx]

    out["mag"] = out["mag"].to_numpy(dtype=float) + s_original_order
    return out


def _inject_eclipsing_binary(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    t = out["JD"].to_numpy(dtype=float)
    if len(t) == 0:
        return out

    period = rng.uniform(0.5, 30.0)
    t0 = rng.uniform(0.0, period)

    depth_pri = rng.uniform(0.3, 1.5)
    width_pri = rng.uniform(0.02, 0.1)  # phase width

    depth_sec = rng.uniform(0.05, 0.5)
    width_sec = width_pri * rng.uniform(0.8, 1.2)

    phase = ((t - t0) % period) / period

    # Distance to primary eclipse at phase 0 or 1
    dist_pri = np.minimum(phase, 1.0 - phase)
    # Distance to secondary eclipse at phase 0.5
    dist_sec = np.abs(phase - 0.5)

    profile = depth_pri * np.exp(-0.5 * (dist_pri / width_pri)**2) + \
              depth_sec * np.exp(-0.5 * (dist_sec / width_sec)**2)

    out["mag"] = out["mag"].to_numpy(dtype=float) + profile
    return out


def _inject_contact_binary(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Simulates a contact (W UMa) or semi-detached binary using ellipsoidal variations.
    Modeled as continuous Fourier series rather than discrete Gaussians.
    """
    out = df.copy()
    t = out["JD"].to_numpy(dtype=float)
    if len(t) == 0:
        return out

    # Contact binaries have short periods
    period = rng.uniform(0.2, 2.0)
    t0 = rng.uniform(0.0, period)

    phase = ((t - t0) % period) / period

    # Ellipsoidal variation (causes the two minima per orbit)
    amp_ellips = rng.uniform(0.05, 0.4)
    # Asymmetry (difference in depth between primary and secondary minima)
    amp_asym = rng.uniform(0.0, 0.2)

    # Positive mag = fainter.
    # cos(4*pi*phase) is +1 at phase 0 and 0.5 (the two eclipses/minima) and -1 at 0.25, 0.75 (maxima)
    # cos(2*pi*phase) is +1 at phase 0 (making primary deeper) and -1 at 0.5 (making secondary shallower)
    profile = amp_ellips * np.cos(4.0 * np.pi * phase) + amp_asym * np.cos(2.0 * np.pi * phase)

    out["mag"] = out["mag"].to_numpy(dtype=float) + profile
    return out


CONTAMINANT_FUNCS = {
    "camera_offset": _inject_camera_offset,
    "camera_cluster": _inject_camera_cluster,
    "semiregular": _inject_semiregular,
    "rcb_like": _inject_rcb_like,
    "cv_outburst": _inject_cv_outburst,
    "drw_agn": _inject_drw_agn,
    "eclipsing_binary": _inject_eclipsing_binary,
    "contact_binary": _inject_contact_binary,
}


def _run_single_injection(family: str, asas_sn_id: str, lc_dir: str, file_ext: str, trial: int, detection_kwargs: dict, seed: int) -> dict:
    fn = CONTAMINANT_FUNCS.get(family)
    rng = np.random.default_rng(seed)
    try:
        df_g, df_v = read_lc_dat2(asas_sn_id, lc_dir, file_ext=file_ext)
        df_lc = pd.concat([df_g, df_v], ignore_index=True)
        if df_lc.empty:
            return {"family": family, "trial": trial, "asas_sn_id": asas_sn_id, "detected": False, "error": "empty"}
        df_bad = fn(df_lc, rng)
        det = _default_detection_func(df_bad, detection_kwargs=detection_kwargs)
        return {
            "family": family,
            "trial": trial,
            "asas_sn_id": asas_sn_id,
            "detected": bool(det.get("detected", False)),
            "dip_significant": bool(det.get("dip_significant", False)),
            "jump_significant": bool(det.get("jump_significant", False)),
            "dip_trigger_max": float(det.get("dip_trigger_max", np.nan)),
            "jump_trigger_max": float(det.get("jump_trigger_max", np.nan)),
        }
    except Exception as e:
        return {"family": family, "trial": trial, "asas_sn_id": asas_sn_id, "detected": False, "error": str(e)}


def run_false_positive_benchmark(
    manifest_df: pd.DataFrame,
    *,
    out_dir: Path,
    families: list[str],
    n_trials_per_family: int,
    detection_kwargs: dict,
    seed: int,
    workers: int = 1,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    required = ["asas_sn_id", "path"]
    if not all(c in manifest_df.columns for c in required):
        raise ValueError(f"manifest_df must contain asas_sn_id and path columns, got {manifest_df.columns}")

    tasks = []
    for family in families:
        if family not in CONTAMINANT_FUNCS:
            continue
        for trial in range(int(n_trials_per_family)):
            pick = manifest_df.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
            trial_seed = int(rng.integers(0, 2**31 - 1))
            
            p = Path(str(pick["path"]))
            lc_dir = str(p.parent) if p.suffix else str(p)
            file_ext = p.suffix[1:] if p.suffix else "dat3"
            tasks.append((family, str(pick["asas_sn_id"]), lc_dir, file_ext, trial, detection_kwargs, trial_seed))

    rows = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_run_single_injection, *t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="False Positive Trials"):
                rows.append(fut.result())
    else:
        for t in tqdm(tasks, desc="False Positive Trials"):
            rows.append(_run_single_injection(*t))

    df = pd.DataFrame(rows)
    if df.empty:
        write_parquet_table(df, out_dir / "false_positive_trials.parquet")
        write_parquet_table(pd.DataFrame(), out_dir / "false_positive_summary.parquet")
        return df

    summary = (
        df.groupby("family", dropna=False)["detected"]
        .agg(["count", "sum"])
        .rename(columns={"sum": "n_false_positive"})
        .reset_index()
    )
    summary["false_positive_rate"] = summary["n_false_positive"] / summary["count"].replace(0, np.nan)

    write_parquet_table(df, out_dir / "false_positive_trials.parquet")
    write_parquet_table(summary, out_dir / "false_positive_summary.parquet")
    return df


def plot_false_alarm_survival(df: pd.DataFrame, out_dir: Path, trigger_mode: str = "posterior_prob"):
    """Plot False Alarm Rate vs Trigger Threshold."""
    plt.figure(figsize=(8, 6))
    
    col = "dip_trigger_max" if "dip_trigger_max" in df.columns else None
    if col is None:
        return
        
    # If using logbf, the range is 0 to ~20. If posterior_prob, 0 to 1.
    is_prob = (trigger_mode == "posterior_prob")
    t_min = 0.0
    t_max = 1.0 if is_prob else min(20.0, df[col].max() if pd.notnull(df[col].max()) else 10.0)
    thresholds = np.linspace(t_min, t_max, 100)
    
    for family, group in df.groupby("family"):
        rates = []
        valid_scores = group[col].dropna().values
        n_total = len(group)
        if n_total == 0:
            continue
            
        for t in thresholds:
            n_trigger = np.sum(valid_scores >= t)
            rates.append(n_trigger / n_total)
            
        plt.plot(thresholds, rates, label=family, lw=2)
        
    plt.xlabel(f"Trigger Threshold ({trigger_mode})", fontsize=12)
    plt.ylabel("False Alarm Rate (FAR)", fontsize=12)
    plt.title("False Alarm Survival Curve", fontsize=14)
    plt.yscale("log")
    plt.ylim(bottom=1e-4, top=1.0)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / "false_alarm_survival.pdf")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="False-positive contaminant benchmark for MALCA")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "lc_manifest_all.parquet", help="Manifest Parquet with asas_sn_id and path")
    parser.add_argument("--output-dir", dest="out_dir", type=Path, default=DEFAULT_OUTPUT_DIR / "false_positive")
    parser.add_argument("--families", type=str, default="camera_offset,camera_cluster,semiregular,rcb_like,cv_outburst,drw_agn,eclipsing_binary,contact_binary")
    parser.add_argument("--n-trials-per-family", type=int, default=FP_TRIALS_PER_FAMILY)
    parser.add_argument("--seed", type=int, default=INJECTION_SEED)
    parser.add_argument("--trigger-mode", type=str, default="posterior_prob", choices=["logbf", "posterior_prob"])
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()

    df_manifest = read_parquet_table(args.manifest)
    
    # Handle schema from manifest builder
    if "source_id" in df_manifest.columns and "asas_sn_id" not in df_manifest.columns:
        df_manifest = df_manifest.rename(columns={"source_id": "asas_sn_id"})
    if "dat_path" in df_manifest.columns and "path" not in df_manifest.columns:
        df_manifest = df_manifest.rename(columns={"dat_path": "path"})
    elif "lc_dir" in df_manifest.columns and "path" not in df_manifest.columns:
        df_manifest = df_manifest.rename(columns={"lc_dir": "path"})

    detection_kwargs = _build_detection_kwargs(args.trigger_mode)

    df_results = run_false_positive_benchmark(
        df_manifest,
        out_dir=args.out_dir,
        families=[f.strip() for f in args.families.split(",") if f.strip()],
        n_trials_per_family=args.n_trials_per_family,
        detection_kwargs=detection_kwargs,
        seed=args.seed,
        workers=args.workers,
    )
    
    plot_false_alarm_survival(df_results, args.out_dir, args.trigger_mode)
    print(f"Wrote false-positive benchmark outputs and plot to {args.out_dir}")


if __name__ == "__main__":
    main()
