from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from malca.events import score_lightcurve
from malca.baseline import per_camera_gp_baseline
from malca.utils import read_lc_dat2
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
)


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


CONTAMINANT_FUNCS = {
    "camera_offset": _inject_camera_offset,
    "camera_cluster": _inject_camera_cluster,
    "semiregular": _inject_semiregular,
    "rcb_like": _inject_rcb_like,
}


def run_false_positive_benchmark(
    manifest_df: pd.DataFrame,
    *,
    out_dir: Path,
    families: list[str],
    n_trials_per_family: int,
    detection_kwargs: dict,
    seed: int,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    required = ["asas_sn_id", "path"]
    if not all(c in manifest_df.columns for c in required):
        raise ValueError("manifest_df must contain asas_sn_id and path columns")

    rows: list[dict] = []
    for family in families:
        fn = CONTAMINANT_FUNCS.get(family)
        if fn is None:
            continue
        for trial in range(int(n_trials_per_family)):
            pick = manifest_df.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
            asas_sn_id = str(pick["asas_sn_id"])
            lc_dir = str(pick["path"])
            try:
                df_g, df_v = read_lc_dat2(asas_sn_id, lc_dir)
                df_lc = pd.concat([df_g, df_v], ignore_index=True)
                if df_lc.empty:
                    continue
                df_bad = fn(df_lc, rng)
                det = _default_detection_func(df_bad, detection_kwargs=detection_kwargs)
                rows.append(
                    {
                        "family": family,
                        "trial": trial,
                        "asas_sn_id": asas_sn_id,
                        "detected": bool(det.get("detected", False)),
                        "dip_significant": bool(det.get("dip_significant", False)),
                        "jump_significant": bool(det.get("jump_significant", False)),
                    }
                )
            except Exception as e:
                rows.append({"family": family, "trial": trial, "asas_sn_id": asas_sn_id, "detected": False, "error": str(e)})

    df = pd.DataFrame(rows)
    if df.empty:
        df.to_csv(out_dir / "false_positive_trials.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "false_positive_summary.csv", index=False)
        return df

    summary = (
        df.groupby("family", dropna=False)["detected"]
        .agg(["count", "sum"])
        .rename(columns={"sum": "n_false_positive"})
        .reset_index()
    )
    summary["false_positive_rate"] = summary["n_false_positive"] / summary["count"].replace(0, np.nan)

    df.to_csv(out_dir / "false_positive_trials.csv", index=False)
    summary.to_csv(out_dir / "false_positive_summary.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="False-positive contaminant benchmark for MALCA")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest CSV/Parquet with asas_sn_id and path")
    parser.add_argument("--output-dir", dest="out_dir", type=Path, default=Path("output/false_positive"))
    parser.add_argument("--families", type=str, default="camera_offset,camera_cluster,semiregular,rcb_like")
    parser.add_argument("--n-trials-per-family", type=int, default=FP_TRIALS_PER_FAMILY)
    parser.add_argument("--seed", type=int, default=INJECTION_SEED)
    parser.add_argument("--trigger-mode", type=str, default="posterior_prob", choices=["logbf", "posterior_prob"])
    args = parser.parse_args()

    if args.manifest.suffix.lower() in {".parquet", ".pq"}:
        df_manifest = pd.read_parquet(args.manifest)
    else:
        df_manifest = pd.read_csv(args.manifest)

    detection_kwargs = _build_detection_kwargs(args.trigger_mode)

    run_false_positive_benchmark(
        df_manifest,
        out_dir=args.out_dir,
        families=[f.strip() for f in args.families.split(",") if f.strip()],
        n_trials_per_family=args.n_trials_per_family,
        detection_kwargs=detection_kwargs,
        seed=args.seed,
    )
    print(f"Wrote false-positive benchmark outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
