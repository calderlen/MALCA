from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as pl

from lc_excursions_naive import dip_finder_naive
from lc_excursions import excursion_finder
from lc_excursions_bayes import run_bayesian_significance
from lc_utils import read_lc_dat2
from lc_baseline import per_camera_gp_baseline


brayden_candidates: list[dict[str, object]] = [
    {"source": "J042214+152530", "source_id": "377957522430", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J202402+383938", "source_id": "42950993887", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J174328+343315", "source_id": "223339338105", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J080327-261620", "source_id": "601296043597", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": False},
    {"source": "J184916-473251", "source_id": "472447294641", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Known", "expected_detected": True},
    {"source": "J183153-284827", "source_id": "455267102087", "category": "Dippers", "mag_bin": "13.5_14", "search_method": "Known", "expected_detected": False},
    {"source": "J070519+061219", "source_id": "266288137752", "category": "Dippers", "mag_bin": "13.5_14", "search_method": "Known", "expected_detected": False},
    {"source": "J081523-385923", "source_id": "532576686103", "category": "Dippers", "mag_bin": "13.5_14", "search_method": "Known", "expected_detected": False},
    {"source": "J085816-430955", "source_id": "352187470767", "category": "Dippers", "mag_bin": "12_12.5", "search_method": "Known", "expected_detected": False},
    {"source": "J114712-621037", "source_id": "609886184506", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Known", "expected_detected": False},
    {"source": "J005437+644347", "source_id": "68720274411", "category": "Multiple Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Known", "expected_detected": True},
    {"source": "J062510-075341", "source_id": "377958261591", "category": "Multiple Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J124745-622756", "source_id": "515397118400", "category": "Multiple Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J175912-120956", "source_id": "326417831663", "category": "Multiple Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J181752-580749", "source_id": "644245387906", "category": "Multiple Eclipse Binaries", "mag_bin": "12_12.5", "search_method": "Known", "expected_detected": True},
    {"source": "J160757-574540", "source_id": "661425129485", "category": "Multiple Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": False},
    {"source": "J073924-272916", "source_id": "438086977939", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J074007-161608", "source_id": "360777377116", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J094848-545959", "source_id": "635655234580", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J162209-444247", "source_id": "412317159120", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J183606-314826", "source_id": "438086901547", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J205245-713514", "source_id": "463856535113", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J212132+480140", "source_id": "120259184943", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J225702+562312", "source_id": "25770019815", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J190316-195739", "source_id": "515396514761", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J175602+013135", "source_id": "231929175915", "category": "Single Eclipse Binaries", "mag_bin": "14_14.5", "search_method": "Known", "expected_detected": True},
    {"source": "J073234-200049", "source_id": "335007754417", "category": "Single Eclipse Binaries", "mag_bin": "14.5_15", "search_method": "Known", "expected_detected": True},
    {"source": "J223332+565552", "source_id": "60130040391", "category": "Single Eclipse Binaries", "mag_bin": "12.5_13", "search_method": "Known", "expected_detected": True},
    {"source": "J183210-173432", "source_id": "317827964025", "category": "Single Eclipse Binaries", "mag_bin": "12.5_13", "search_method": "Pipeline", "expected_detected": False},
]


def _load_manifest_df(manifest_path: Path | str) -> pd.DataFrame:
    path = Path(manifest_path).expanduser()
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "source_id" not in df.columns and "asas_sn_id" in df.columns:
        df = df.rename(columns={"asas_sn_id": "source_id"})
    df["source_id"] = df["source_id"].astype(str)
    return df


def _dataframe_from_candidates(data: Sequence[Mapping[str, object]] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(data or brayden_candidates).copy()
    df.rename(columns={"Source": "source", "Source ID": "source_id"}, inplace=True)
    df["source_id"] = df["source_id"].astype(str)
    return df


def _target_map(df: pd.DataFrame) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for mag_bin, chunk in df.groupby("mag_bin"):
        grouped[str(mag_bin)] = set(chunk["source_id"].astype(str))
    return grouped


def _records_from_manifest(df: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    records: dict[str, list[dict[str, object]]] = {}
    for rec in df.to_dict("records"):
        source_id = str(rec.get("source_id"))
        mag_bin = str(rec.get("mag_bin"))
        lc_dir = str(rec.get("lc_dir"))
        dat_path = rec.get("dat_path") or str(Path(lc_dir) / f"{source_id}.dat")
        record = {
            "mag_bin": mag_bin,
            "index_num": rec.get("index_num"),
            "index_csv": rec.get("index_csv"),
            "lc_dir": lc_dir,
            "asas_sn_id": source_id,
            "dat_path": dat_path,
            "found": bool(rec.get("dat_exists", True)),
        }
        records.setdefault(mag_bin, []).append(record)
    return records


def _records_from_skypatrol_dir(df_targets: pd.DataFrame, skypatrol_dir: Path) -> dict[str, list[dict[str, object]]]:
    base = Path(skypatrol_dir)
    if not base.exists():
        return {}

    records: dict[str, list[dict[str, object]]] = {}
    for _, row in df_targets.iterrows():
        source_id = str(row.get("source_id"))
        mag_bin = str(row.get("mag_bin"))
        csv_path = base / f"{source_id}-light-curves.csv"
        if not csv_path.exists():
            continue
        rec = {
            "mag_bin": mag_bin,
            "index_num": None,
            "index_csv": None,
            "lc_dir": str(base),
            "asas_sn_id": source_id,
            "dat_path": str(csv_path),
            "found": True,
        }
        records.setdefault(mag_bin, []).append(rec)
    return records


def _coerce_candidate_records(data) -> list[dict[str, object]]:
    if data is None:
        return list(brayden_candidates)

    if isinstance(data, pd.DataFrame):
        records = data.to_dict("records")
    else:
        records = list(data)

    if not records:
        return []

    first = records[0]
    if isinstance(first, Mapping):
        coerced: list[dict[str, object]] = []
        for rec in records:
            if not isinstance(rec, Mapping):
                continue
            new = dict(rec)
            source_id = str(new.get("source_id", new.get("Source_ID", ""))).strip()
            if not source_id:
                continue
            new["source_id"] = source_id
            new.setdefault("source", new.get("source", source_id))
            coerced.append(new)
        return coerced

    ids = []
    for entry in records:
        entry_str = str(entry)
        entry_path = Path(entry_str)
        stem = entry_path.stem
        source_id = stem.split("-")[0] if stem else entry_str
        ids.append(source_id)

    lookup = {c["source_id"]: c for c in brayden_candidates}
    coerced = []
    seen = set()
    for source_id in ids:
        if source_id in seen:
            continue
        seen.add(source_id)
        if source_id in lookup:
            coerced.append(lookup[source_id])
        else:
            coerced.append({"source": source_id, "source_id": source_id, "mag_bin": None})
    return coerced


def _resolve_candidates(spec: str | None):
    if spec is None:
        return list(brayden_candidates)

    env = globals()
    if spec in env:
        return _coerce_candidate_records(env[spec])

    cand_path = Path(spec)
    if cand_path.exists():
        if cand_path.suffix.lower() in {".csv", ".tsv"}:
            df = pd.read_csv(cand_path)
        elif cand_path.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(cand_path)
        else:
            raise SystemExit(f"Unsupported candidates file format: {cand_path}")
        return _coerce_candidate_records(df)

    raise SystemExit(f"Unknown candidates spec '{spec}'. Provide a built-in name or a valid file path.")


def _clean_for_bayes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep this consistent with the Bayesian module's cleaning so event_indices align with plotting."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    mask = np.ones(len(out), dtype=bool)

    if "saturated" in out.columns:
        mask &= (out["saturated"] == 0)

    mask &= out["JD"].notna() & out["mag"].notna()

    if "error" in out.columns:
        mask &= out["error"].notna() & (out["error"] > 0.0) & (out["error"] < 1.0)

    out = out.loc[mask].sort_values("JD").reset_index(drop=True)
    return out


def _plot_light_curve_with_dips(
    dfg: pd.DataFrame,
    dfv: pd.DataFrame,
    res_g: dict,
    res_v: dict,
    source_id: str,
    plot_path: Path,
):
    fig, axes = pl.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Baseline parameters (match the SHO-ish defaults; note baseline function takes q not Q)
    baseline_kwargs = {
        "S0": 0.0005,
        "w0": 0.0031415926535897933,
        "q": 0.7,
        "jitter": 0.006,
        "sigma_floor": None,
        "add_sigma_eff_col": True,
    }

    camera_colors = pl.cm.tab10(np.linspace(0, 1, 10))

    def _plot_band(ax, df_band: pd.DataFrame, res: dict, band_label: str):
        if df_band is None or df_band.empty or "JD" not in df_band.columns or "mag" not in df_band.columns:
            ax.text(0.5, 0.5, f"No {band_label}-band data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{source_id} - {band_label}-band (no data)", fontsize=12)
            return

        df_plot = df_band.copy()
        if "error" in df_plot.columns:
            df_plot = df_plot[df_plot["error"] <= 1.0]

        df_baseline = None
        try:
            df_baseline = per_camera_gp_baseline(df_plot, **baseline_kwargs)
        except Exception:
            df_baseline = None

        cam_col = "camera#" if "camera#" in df_plot.columns else None
        if cam_col and cam_col in df_plot.columns:
            cameras = sorted(df_plot[cam_col].dropna().unique())
            for i, cam in enumerate(cameras):
                cam_mask = df_plot[cam_col] == cam
                cam_data = df_plot[cam_mask]
                if cam_data.empty:
                    continue

                color = camera_colors[i % len(camera_colors)]
                label = f"Cam {cam}"

                if "error" in cam_data.columns:
                    ax.errorbar(
                        cam_data["JD"], cam_data["mag"], yerr=cam_data["error"],
                        fmt="o", markersize=3, alpha=0.6, color=color,
                        label=label, elinewidth=1.0, capsize=1.5
                    )
                else:
                    ax.scatter(cam_data["JD"], cam_data["mag"], s=10, alpha=0.6, color=color, label=label)

                if df_baseline is not None and "baseline" in df_baseline.columns:
                    cam_base = df_baseline[df_baseline[cam_col] == cam].sort_values("JD")
                    if not cam_base.empty:
                        ax.plot(
                            cam_base["JD"], cam_base["baseline"],
                            color=color, linewidth=2.0, alpha=0.9, linestyle="--",
                            label=f"Baseline {label}"
                        )
        else:
            if "error" in df_plot.columns:
                ax.errorbar(
                    df_plot["JD"], df_plot["mag"], yerr=df_plot["error"],
                    fmt="o", markersize=3, alpha=0.6,
                    label=f"{band_label}-band", elinewidth=1.0, capsize=1.5
                )
            else:
                ax.scatter(df_plot["JD"], df_plot["mag"], s=10, alpha=0.6, label=f"{band_label}-band")

            if df_baseline is not None and "baseline" in df_baseline.columns:
                df_sorted = df_baseline.sort_values("JD")
                ax.plot(
                    df_sorted["JD"], df_sorted["baseline"],
                    linewidth=2.0, alpha=0.9, linestyle="--", label="baseline"
                )

        # -------------------------------------------------------------
        # MODIFIED LOGIC: Plot lines ONLY for BIC-confirmed events
        # -------------------------------------------------------------
        
        # 1. Attempt to plot based on run_summaries (BIC aware)
        run_summaries = res.get("dip", {}).get("run_summaries", [])
        confirmed_count = 0
        
        if run_summaries:
            for summary in run_summaries:
                # Filter: skip if morphology is 'noise' or 'none'
                morph = summary.get("morphology", "none")
                if morph not in ("noise", "none"):
                    confirmed_count += 1
                    
                    # Try to get t0 from params, otherwise use range center
                    t0 = summary.get("params", {}).get("t0")
                    if t0 is None:
                        start_jd = summary.get("start_jd")
                        end_jd = summary.get("end_jd")
                        if start_jd is not None and end_jd is not None:
                            t0 = (start_jd + end_jd) / 2.0
                    
                    if t0 is not None and np.isfinite(t0):
                        # Plot distinct solid red line for confirmed event
                        ax.axvline(t0, color='red', alpha=0.6, linestyle="-", linewidth=1.5)

        else:
            # 2. Fallback for legacy results (no summaries): Plot all triggered indices
            dip_indices = res.get("dip", {}).get("event_indices", np.array([], dtype=int))
            if isinstance(dip_indices, (list, tuple)):
                dip_indices = np.asarray(dip_indices, dtype=int)
            
            if isinstance(dip_indices, np.ndarray) and dip_indices.size > 0:
                valid = dip_indices[(dip_indices >= 0) & (dip_indices < len(df_plot))]
                if valid.size:
                    dip_jds = df_plot.iloc[valid]["JD"].to_numpy()
                    for jd in dip_jds:
                        ax.axvline(jd, alpha=0.3, linestyle="--", linewidth=1)
                    confirmed_count = len(dip_jds) # approximations

        n_triggers = int(res.get("dip", {}).get("n_dips", 0))
        
        ax.set_ylabel(f"Magnitude ({band_label})", fontsize=12)
        ax.set_title(f"{source_id} - {band_label}-band (triggers={n_triggers}, confirmed={confirmed_count})", fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", ncol=2)

    _plot_band(axes[0], dfg, res_g, "g")
    _plot_band(axes[1], dfv, res_v, "V")
    axes[1].set_xlabel("JD", fontsize=12)

    pl.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    pl.savefig(plot_path, dpi=150, bbox_inches="tight")
    pl.close()


def build_reproduction_report(
    candidates: Sequence[Mapping[str, object]] | None = None,
    *,
    out_dir: Path | str = "./peak_results_repro",
    out_format: str = "csv",
    n_workers: int | None = None,
    chunk_size: int = 250000,
    metrics_baseline_func=None,
    metrics_dip_threshold: float = 0.3,
    # posterior-prob thresholding (legacy)
    bayes_significance_threshold: float | None = 99.99997,
    bayes_p_points: int = 80,
    # NEW: log BF triggering
    bayes_trigger_mode: str = "logbf",          # "logbf" or "posterior_prob"
    bayes_logbf_threshold_dip: float = 5.0,     # trigger if max log BF >= this
    bayes_logbf_threshold_jump: float = 5.0,
    skypatrol_dir: Path | str | None = None,
    extra_columns: Iterable[str] | None = None,
    manifest_path: Path | str | None = None,
    method: str = "naive",
    verbose: bool = False,
    **baseline_kwargs,
) -> pd.DataFrame:
    manifest_df = _load_manifest_df(manifest_path) if manifest_path is not None else None

    baseline_candidates = candidates if candidates is not None else brayden_candidates
    df_targets = _dataframe_from_candidates(baseline_candidates)

    manifest_subset = None
    if manifest_df is not None:
        manifest_subset = manifest_df[manifest_df["source_id"].isin(df_targets["source_id"])].copy()
        if not manifest_subset.empty:
            cols = [
                col
                for col in ["source_id", "mag_bin", "lc_dir", "index_num", "index_csv", "dat_path", "dat_exists"]
                if col in manifest_subset.columns
            ]
            df_targets = df_targets.merge(manifest_subset[cols], on="source_id", how="left")
            if "mag_bin_x" in df_targets.columns and "mag_bin_y" in df_targets.columns:
                df_targets["mag_bin"] = df_targets["mag_bin_y"].fillna(df_targets["mag_bin_x"])
                df_targets = df_targets.drop(columns=["mag_bin_x", "mag_bin_y"])

    target_map = _target_map(df_targets)
    records_map = _records_from_manifest(manifest_subset) if manifest_subset is not None else None

    if (records_map is None or not records_map) and skypatrol_dir is not None:
        records_map = _records_from_skypatrol_dir(df_targets, Path(skypatrol_dir))
        if verbose:
            n_found = sum(len(v) for v in records_map.values())
            print(f"[DEBUG] Built records_map from skypatrol_dir: {n_found} light curves found")

    if method == "naive":
        rows = dip_finder_naive(
            mag_bins=sorted(target_map),
            out_dir=out_dir,
            out_format=out_format,
            n_workers=n_workers,
            chunk_size=chunk_size,
            metrics_baseline_func=metrics_baseline_func,
            metrics_dip_threshold=metrics_dip_threshold,
            target_ids_by_bin=target_map,
            records_by_bin=records_map,
            return_rows=True,
            **baseline_kwargs,
        )

    elif method == "biweight":
        rows = excursion_finder(
            mode="dips",
            mag_bins=sorted(target_map),
            out_dir=out_dir,
            out_format=out_format,
            n_workers=n_workers,
            chunk_size=chunk_size,
            target_ids_by_bin=target_map,
            records_by_bin=records_map,
            return_rows=True,
            peak_kwargs={"sigma_threshold": metrics_dip_threshold},
        )

    elif method == "bayes":
        if records_map is None or not records_map:
            raise SystemExit("Bayesian method requires light-curve paths. Provide --manifest or --skypatrol-dir.")

        rows = []

        for mag_bin in sorted(records_map):
            for rec in records_map[mag_bin]:
                asn = rec.get("asas_sn_id")
                lc_dir = rec.get("lc_dir")
                dat_path = rec.get("dat_path")
                has_path = bool(dat_path) and Path(str(dat_path)).exists()

                try:
                    dfg, dfv = (
                        read_lc_dat2(asn, lc_dir)
                        if asn and lc_dir and has_path
                        else (pd.DataFrame(), pd.DataFrame())
                    )
                except Exception as e:
                    if verbose:
                        print(f"[DEBUG] {asn}: read_lc_dat2 failed: {e}")
                    dfg, dfv = pd.DataFrame(), pd.DataFrame()

                if verbose:
                    print(f"[DEBUG] {asn}: loaded g={len(dfg)} rows, v={len(dfv)} rows from {dat_path}")

                def _apply_triggering(result: dict, band_name: str) -> dict:
                    """
                    Force triggering to be based on either:
                      - log BF local (preferred)
                      - posterior probability threshold (legacy)
                    """
                    for kind in ("dip", "jump"):
                        block = result.get(kind, {})
                        if not isinstance(block, dict):
                            result[kind] = {"significant": False}
                            continue

                        if bayes_trigger_mode == "logbf":
                            thr = bayes_logbf_threshold_dip if kind == "dip" else bayes_logbf_threshold_jump
                            log_bf_local = block.get("log_bf_local", None)
                            if log_bf_local is None:
                                # if the bayes module wasn't updated, this stays off
                                block["event_indices"] = np.array([], dtype=int)
                                block["significant"] = False
                                block["max_log_bf_local"] = np.nan
                            else:
                                lb = np.asarray(log_bf_local, float)
                                finite = np.isfinite(lb)
                                max_lb = float(np.nanmax(lb)) if finite.any() else np.nan
                                idx = np.nonzero(finite & (lb >= float(thr)))[0].astype(int)
                                block["event_indices"] = idx
                                block["significant"] = bool(np.isfinite(max_lb) and (max_lb >= float(thr)))
                                block["max_log_bf_local"] = max_lb

                            # counts
                            block["max_event_prob"] = np.nan
                            block["n_dips"] = int(len(block.get("event_indices", []))) if kind == "dip" else block.get("n_dips", 0)
                            block["n_jumps"] = int(len(block.get("event_indices", []))) if kind == "jump" else block.get("n_jumps", 0)

                            if verbose:
                                mx = block.get("max_log_bf_local", np.nan)
                                bf = block.get("bayes_factor", np.nan)
                                ct = len(block.get("event_indices", []))
                                print(f"[DEBUG] {asn} {band_name}: {kind} max_logBF={mx:.2f} thr={thr:.2f} "
                                      f"count={ct} globalBF={bf:.2f}")

                        else:
                            # posterior_prob mode
                            event_prob = np.asarray(block.get("event_probability", np.array([])), float)
                            max_prob = float(np.nanmax(event_prob)) if event_prob.size else np.nan
                            block["max_event_prob"] = max_prob

                            event_indices = block.get("event_indices", np.array([], dtype=int))
                            if isinstance(event_indices, (list, tuple)):
                                event_indices = np.asarray(event_indices, dtype=int)
                            if not isinstance(event_indices, np.ndarray):
                                event_indices = np.array([], dtype=int)

                            count_key = "n_dips" if kind == "dip" else "n_jumps"
                            block[count_key] = int(event_indices.size)

                            if verbose:
                                dip_bf = block.get("bayes_factor", np.nan)
                                print(f"[DEBUG] {asn} {band_name}: {kind} max_prob={max_prob:.6f} "
                                      f"count={int(event_indices.size)} globalBF={dip_bf:.2f}")

                        result[kind] = block

                    return result

                def _bayes(df: pd.DataFrame, band_name: str):
                    dfc = _clean_for_bayes(df)
                    if dfc is None or dfc.empty:
                        return {
                            "dip": {"significant": False, "bayes_factor": np.nan, "max_event_prob": np.nan, "n_dips": 0, "max_log_bf_local": np.nan},
                            "jump": {"significant": False, "bayes_factor": np.nan, "max_event_prob": np.nan, "n_jumps": 0, "max_log_bf_local": np.nan},
                        }

                    try:
                        # physics convention note: if converting sigma->prob, we use one-tailed Gaussian CDF (norm.cdf)
                        result = run_bayesian_significance(
                            dfc,
                            significance_threshold=float(bayes_significance_threshold) if bayes_significance_threshold is not None else 99.99997,
                            p_points=int(bayes_p_points),
                        )
                        result = _apply_triggering(result, band_name)
                        return result
                    except Exception as e:
                        if verbose:
                            print(f"[DEBUG] {asn} {band_name}: run_bayesian_significance failed: {e}")
                        return {
                            "dip": {"significant": False, "bayes_factor": np.nan, "max_event_prob": np.nan, "n_dips": 0, "max_log_bf_local": np.nan},
                            "jump": {"significant": False, "bayes_factor": np.nan, "max_event_prob": np.nan, "n_jumps": 0, "max_log_bf_local": np.nan},
                        }

                res_g = _bayes(dfg, "g")
                res_v = _bayes(dfv, "V")

                if out_dir:
                    plot_path = Path(out_dir) / f"{asn}_dips.png"
                    _plot_light_curve_with_dips(
                        _clean_for_bayes(dfg),
                        _clean_for_bayes(dfv),
                        res_g,
                        res_v,
                        str(asn),
                        plot_path,
                    )

                rows.append(
                    {
                        "mag_bin": str(rec.get("mag_bin")),
                        "asas_sn_id": asn,
                        "index_num": rec.get("index_num"),
                        "index_csv": rec.get("index_csv"),
                        "lc_dir": lc_dir,
                        "dat_path": dat_path,

                        "g_bayes_dip_significant": bool(res_g["dip"].get("significant", False)),
                        "v_bayes_dip_significant": bool(res_v["dip"].get("significant", False)),
                        "g_bayes_jump_significant": bool(res_g["jump"].get("significant", False)),
                        "v_bayes_jump_significant": bool(res_v["jump"].get("significant", False)),

                        "g_bayes_dip_max_prob": float(res_g["dip"].get("max_event_prob", np.nan)),
                        "v_bayes_dip_max_prob": float(res_v["dip"].get("max_event_prob", np.nan)),

                        "g_bayes_dip_bayes_factor": float(res_g["dip"].get("bayes_factor", np.nan)),
                        "v_bayes_dip_bayes_factor": float(res_v["dip"].get("bayes_factor", np.nan)),
                        "g_bayes_jump_bayes_factor": float(res_g["jump"].get("bayes_factor", np.nan)),
                        "v_bayes_jump_bayes_factor": float(res_v["jump"].get("bayes_factor", np.nan)),

                        "g_bayes_dip_max_logbf": float(res_g["dip"].get("max_log_bf_local", np.nan)),
                        "v_bayes_dip_max_logbf": float(res_v["dip"].get("max_log_bf_local", np.nan)),
                        "g_bayes_jump_max_logbf": float(res_g["jump"].get("max_log_bf_local", np.nan)),
                        "v_bayes_jump_max_logbf": float(res_v["jump"].get("max_log_bf_local", np.nan)),

                        "g_bayes_n_dips": int(res_g["dip"].get("n_dips", 0)),
                        "v_bayes_n_dips": int(res_v["dip"].get("n_dips", 0)),
                        "g_bayes_n_jumps": int(res_g["jump"].get("n_jumps", 0)),
                        "v_bayes_n_jumps": int(res_v["jump"].get("n_jumps", 0)),
                    }
                )
    else:
        rows = None

    if rows is None:
        rows_df = pd.DataFrame(columns=["source_id", "mag_bin"])
    elif isinstance(rows, list):
        rows_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["source_id", "mag_bin"])
    elif isinstance(rows, pd.DataFrame):
        rows_df = rows.copy() if not rows.empty else pd.DataFrame(columns=["source_id", "mag_bin"])
    else:
        rows_df = pd.DataFrame(columns=["source_id", "mag_bin"])

    if "source_id" not in rows_df.columns:
        if "asas_sn_id" in rows_df.columns:
            rows_df["source_id"] = rows_df["asas_sn_id"].astype(str)
        else:
            rows_df["source_id"] = ""
    rows_df = rows_df.drop(columns=["asas_sn_id"], errors="ignore")
    if "mag_bin" not in rows_df.columns:
        rows_df["mag_bin"] = ""

    merged = df_targets.merge(rows_df, on=["source_id", "mag_bin"], how="left", suffixes=("", "_det"))

    g_peaks = merged["g_n_peaks"] if "g_n_peaks" in merged.columns else pd.Series(np.nan, index=merged.index)
    v_peaks = merged["v_n_peaks"] if "v_n_peaks" in merged.columns else pd.Series(np.nan, index=merged.index)
    g_bayes = (
        merged["g_bayes_dip_significant"].fillna(False).astype(bool)
        if "g_bayes_dip_significant" in merged.columns
        else pd.Series(False, index=merged.index)
    )
    v_bayes = (
        merged["v_bayes_dip_significant"].fillna(False).astype(bool)
        if "v_bayes_dip_significant" in merged.columns
        else pd.Series(False, index=merged.index)
    )

    merged["detected"] = (
        (g_peaks.fillna(0).astype(float) > 0)
        | (v_peaks.fillna(0).astype(float) > 0)
        | g_bayes
        | v_bayes
    )

    def _format_detection(row: pd.Series) -> str:
        if not row.get("detected", False):
            return "—"
        parts = [f"mag_bin={row.get('mag_bin', '')}"]

        if "g_n_peaks" in row and pd.notna(row["g_n_peaks"]):
            parts.append(f"g_peaks={int(row['g_n_peaks'])}")
        if "v_n_peaks" in row and pd.notna(row["v_n_peaks"]):
            parts.append(f"v_peaks={int(row['v_n_peaks'])}")

        def _bayes_part(prefix: str) -> str | None:
            sig = bool(row.get(f"{prefix}_bayes_dip_significant", False))
            if not sig:
                return None
            bf = row.get(f"{prefix}_bayes_dip_bayes_factor")
            bf_str = f"{float(bf):.3f}" if pd.notna(bf) else "nan"
            n_dips = int(row.get(f"{prefix}_bayes_n_dips", 0))
            mxlog = row.get(f"{prefix}_bayes_dip_max_logbf")
            if pd.notna(mxlog):
                return f"{prefix}_bayes_dip (maxlogBF={float(mxlog):.2f}, BF={bf_str}, n={n_dips})"
            return f"{prefix}_bayes_dip (BF={bf_str}, n={n_dips})"

        for pref in ("g", "v"):
            part = _bayes_part(pref)
            if part:
                parts.append(part)

        return "; ".join(parts)

    merged["detection_details"] = merged.apply(_format_detection, axis=1)
    if "expected_detected" in merged.columns:
        merged["matches_expected"] = merged["detected"] == merged["expected_detected"].astype(bool)

    if extra_columns:
        cols = [c for c in extra_columns if c in merged.columns]
    else:
        cols = []

    ordered_cols = [
        "source",
        "source_id",
        "category",
        "mag_bin",
        "detected",
        "detection_details",
    ]
    for col in [
        "expected_detected",
        "matches_expected",
        "g_n_peaks",
        "v_n_peaks",
        "g_bayes_dip_significant",
        "v_bayes_dip_significant",
        "g_bayes_dip_max_prob",
        "v_bayes_dip_max_prob",
        "g_bayes_dip_max_logbf",
        "v_bayes_dip_max_logbf",
        "g_bayes_jump_significant",
        "v_bayes_jump_significant",
        "g_bayes_dip_bayes_factor",
        "v_bayes_dip_bayes_factor",
        "g_bayes_jump_bayes_factor",
        "v_bayes_jump_bayes_factor",
        "g_bayes_n_dips",
        "v_bayes_n_dips",
        "g_bayes_n_jumps",
        "v_bayes_n_jumps",
        "g_max_depth",
        "v_max_depth",
        "jd_first",
        "jd_last",
    ]:
        if col in merged.columns:
            ordered_cols.append(col)
    ordered_cols.extend([c for c in cols if c not in ordered_cols])

    remaining = [c for c in merged.columns if c not in ordered_cols]
    ordered_cols.extend(remaining)

    return merged[ordered_cols]


__all__ = [
    "brayden_candidates",
    "build_reproduction_report",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the targeted reproduction search and summarize results.")
    parser.add_argument("--out-dir", default="./results_test", help="Directory for peak_results output")
    parser.add_argument("--out-format", choices=("csv", "parquet"), default="csv")
    parser.add_argument("--n-workers", type=int, default=10, help="ProcessPool worker count for dip_finder_naive")
    parser.add_argument("--chunk-size", type=int, default=250000, help="Rows per chunk flush for CSV output")
    parser.add_argument("--metrics-dip-threshold", type=float, default=0.3, help="Dip threshold for run_metrics")

    # posterior-prob thresholding (legacy)
    parser.add_argument(
        "--bayes-significance-threshold",
        type=float,
        default=None,
        help="Per-point posterior threshold (percentage, e.g. 99.99997). Only used if --bayes-trigger-mode=posterior_prob.",
    )
    parser.add_argument(
        "--bayes-sigma",
        type=float,
        default=None,
        help="Sigma threshold converted to a one-tailed Gaussian CDF (%). Only used if --bayes-trigger-mode=posterior_prob.",
    )
    parser.add_argument("--bayes-p-points", type=int, default=80, help="Number of logit-spaced probability grid points")

    # NEW: log BF triggering
    parser.add_argument(
        "--bayes-trigger-mode",
        choices=("logbf", "posterior_prob"),
        default="logbf",
        help="Trigger Bayesian detections using log BF (recommended) or posterior probability (legacy).",
    )
    parser.add_argument(
        "--bayes-logbf-threshold-dip",
        type=float,
        default=5.0,
        help="Dip triggers when max per-point log BF >= this (only if --bayes-trigger-mode=logbf).",
    )
    parser.add_argument(
        "--bayes-logbf-threshold-jump",
        type=float,
        default=5.0,
        help="Jump triggers when max per-point log BF >= this (only if --bayes-trigger-mode=logbf).",
    )

    parser.add_argument("--skypatrol-dir", default=None, help="Directory with SkyPatrol CSV files (<source_id>-light-curves.csv)")
    parser.add_argument("--manifest", default=None, help="Path to lc_manifest CSV/Parquet for targeted reproduction")
    parser.add_argument(
        "--baseline-func",
        default=None,
        help="Naive-method baseline function import path (e.g. module:func). This does not affect Bayesian baseline.",
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Candidate spec (built-in list name or path to CSV/Parquet file).",
    )
    parser.add_argument(
        "--method",
        choices=("naive", "biweight", "bayes"),
        default="naive",
        help="Dip finder to use: baseline-residual (naive), biweight/fit-based (biweight), or Bayesian (bayes).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print debug info")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    kwargs = {}
    if args.baseline_func:
        if ":" in args.baseline_func:
            mod_name, func_name = args.baseline_func.split(":", 1)
        else:
            mod_name, func_name = "lc_baseline", args.baseline_func
        mod = __import__(mod_name, fromlist=[func_name])
        kwargs["metrics_baseline_func"] = getattr(mod, func_name)

    candidate_data = _resolve_candidates(args.candidates)

    # posterior-prob threshold (only if requested)
    bayes_significance_threshold = args.bayes_significance_threshold
    if args.bayes_trigger_mode == "posterior_prob":
        # physics convention note: one-tailed Gaussian threshold uses CDF(sigma)
        if args.bayes_sigma is not None:
            prob = stats.norm.cdf(args.bayes_sigma)
            bayes_significance_threshold = prob * 100.0
            if args.verbose:
                print(f"[DEBUG] Converting {args.bayes_sigma}-sigma to significance threshold: {bayes_significance_threshold:.8f}%")
        elif bayes_significance_threshold is None:
            prob = stats.norm.cdf(5.0)
            bayes_significance_threshold = prob * 100.0
            if args.verbose:
                print(f"[DEBUG] Using default 5-sigma significance threshold: {bayes_significance_threshold:.8f}%")
    else:
        bayes_significance_threshold = None
        if args.verbose:
            print(f"[DEBUG] Using logBF triggering: dip_thr={args.bayes_logbf_threshold_dip}, jump_thr={args.bayes_logbf_threshold_jump}")

    report = build_reproduction_report(
        candidates=candidate_data,
        out_dir=args.out_dir,
        out_format=args.out_format,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
        metrics_dip_threshold=args.metrics_dip_threshold,
        bayes_significance_threshold=bayes_significance_threshold,
        bayes_p_points=args.bayes_p_points,
        bayes_trigger_mode=args.bayes_trigger_mode,
        bayes_logbf_threshold_dip=args.bayes_logbf_threshold_dip,
        bayes_logbf_threshold_jump=args.bayes_logbf_threshold_jump,
        skypatrol_dir=args.skypatrol_dir,
        manifest_path=args.manifest,
        method=args.method,
        verbose=args.verbose,
        **kwargs,
    )

    columns = [
        "source",
        "source_id",
        "mag_bin",
        "detected",
        "detection_details",
        "g_n_peaks",
        "v_n_peaks",
        "g_bayes_dip_significant",
        "v_bayes_dip_significant",
        "g_bayes_n_dips",
        "v_bayes_n_dips",
        "g_bayes_dip_max_prob",
        "v_bayes_dip_max_prob",
        "g_bayes_dip_max_logbf",
        "v_bayes_dip_max_logbf",
        "g_bayes_dip_bayes_factor",
        "v_bayes_dip_bayes_factor",
        "matches_expected",
    ]
    existing = [c for c in columns if c in report.columns]
    print(report[existing].to_string(index=False))


if __name__ == "__main__":
    main()