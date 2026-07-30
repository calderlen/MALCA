"""Long-period dipper injection benchmark.

Injects synthetic Gaussian dips at configurable recurrence periods into real
control light curves, then scores period recovery under:

* short-period PDM/CE only (legacy 1–100 d window)
* long-period LS discovery
* event-period + consensus

Designed to complement ``periodicity_gate_injection_benchmark.py`` for
baseline/P ∈ {1.5, 2, 3, 5} regimes where short-period methods alias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from malca.config import (
    ADAPTIVE_BOUNDS_ENABLED,
    LONG_PERIOD_ENABLED,
    POST_FILTER_LEGACY_MAX_PERIOD,
    POST_FILTER_LEGACY_MIN_PERIOD,
)
from malca.core.period_pipeline import compute_period_consensus_for_lc
from malca.core.stats import compute_ce_stats, compute_pdm_stats, long_period_ls_search
from malca.evaluation.periodicity_gate_injection_benchmark import (
    _gaussian_dip_profile,
    _periodic_centers,
    period_match_quality,
)
from malca.stv.event_period import event_based_period


CYCLE_BINS: tuple[float, ...] = (1.5, 2.0, 3.0, 5.0)
EVENT_COUNTS: tuple[int, ...] = (1, 2, 3, 5)


@dataclass
class LongPeriodDipperBenchmarkConfig:
    output_base_dir: str | Path = "output/evaluation/long_period_dipper"
    run_tag: str | None = None
    n_trials_per_setting: int = 20
    seed: int = 0
    period_rel_tol: float = 0.08
    dip_amplitude_range: tuple[float, float] = (0.15, 0.8)
    dip_duration_range: tuple[float, float] = (20.0, 120.0)
    baseline_cycles: tuple[float, ...] = CYCLE_BINS
    event_counts: tuple[int, ...] = EVENT_COUNTS
    n_bootstrap: int = 0
    significance_level: float = 0.01


@dataclass
class LongPeriodDipperBenchmarkRun:
    config: LongPeriodDipperBenchmarkConfig
    run_dir: Path
    trial_design: pd.DataFrame
    results: pd.DataFrame
    summary: pd.DataFrame


def make_run_dir(config: LongPeriodDipperBenchmarkConfig) -> Path:
    tag = config.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_base_dir).expanduser() / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_trial_design(
    controls: pd.DataFrame,
    config: LongPeriodDipperBenchmarkConfig,
) -> pd.DataFrame:
    """Build a Cartesian grid of (control, baseline_cycles, event_count) trials."""
    rng = np.random.default_rng(int(config.seed))
    rows: list[dict[str, Any]] = []
    trial_id = 0
    for _, control in controls.iterrows():
        span = float(control.get("jd_span", control.get("control_jd_span", np.nan)))
        if not np.isfinite(span) or span <= 0:
            continue
        for cycles in config.baseline_cycles:
            true_period = span / float(cycles)
            if true_period <= POST_FILTER_LEGACY_MAX_PERIOD:
                continue
            for event_count in config.event_counts:
                for rep in range(int(config.n_trials_per_setting)):
                    trial_id += 1
                    rows.append(
                        {
                            "trial_id": int(trial_id),
                            "source_id": str(control.get("source_id", "")),
                            "source_path": str(control.get("source_path", "")),
                            "trial_seed": int(rng.integers(0, 2**31 - 1)),
                            "baseline_days": float(span),
                            "baseline_cycles": float(cycles),
                            "true_period_days": float(true_period),
                            "requested_event_count": int(event_count),
                            "dip_amplitude": float(rng.uniform(*config.dip_amplitude_range)),
                            "dip_duration": float(rng.uniform(*config.dip_duration_range)),
                            "phase0": float(rng.uniform(0.0, 1.0)),
                        }
                    )
    return pd.DataFrame(rows)


def inject_long_period_dips(
    df_lc: pd.DataFrame,
    *,
    true_period_days: float,
    event_count: int,
    amplitude: float,
    duration: float,
    phase0: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[float]]:
    """Return a copy of ``df_lc`` with injected dips and the dip-center JDs."""
    out = df_lc.copy()
    jd = out["JD"].to_numpy(dtype=float)
    mag = out["mag"].to_numpy(dtype=float)
    if event_count <= 0:
        return out, []

    if event_count == 1:
        finite = jd[np.isfinite(jd)]
        center = float(np.median(finite)) if finite.size else float(jd[0])
        centers = [center]
    else:
        centers = _periodic_centers(
            jd,
            period_days=float(true_period_days),
            phase0=float(phase0),
            duration=float(duration),
            jitter_frac=0.02,
            rng=rng,
        )
        if len(centers) > int(event_count):
            stride = max(1, len(centers) // int(event_count))
            centers = centers[::stride][: int(event_count)]
        elif len(centers) < int(event_count) and centers:
            # Pad with additional cycles when the LC span is tight.
            while len(centers) < int(event_count):
                centers.append(float(centers[-1] + true_period_days))

    for center in centers:
        profile = _gaussian_dip_profile(jd, float(center), float(duration), float(amplitude))
        mag = mag + profile
    out["mag"] = mag
    return out, [float(c) for c in centers]


def evaluate_period_recovery(
    jd: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    *,
    dip_epochs: list[float],
    baseline_days: float,
    true_period_days: float,
    rel_tol: float,
    n_bootstrap: int,
    significance_level: float,
) -> dict[str, Any]:
    """Score short PDM/CE, long LS, event-period, and consensus on one trial."""
    if ADAPTIVE_BOUNDS_ENABLED:
        from malca.core.period_bounds import STAGE_POSTFILTER, bounds_from_jd

        min_p, max_p = bounds_from_jd(jd, stage=STAGE_POSTFILTER).as_tuple()
    else:
        min_p = float(POST_FILTER_LEGACY_MIN_PERIOD)
        max_p = float(POST_FILTER_LEGACY_MAX_PERIOD)

    pdm = compute_pdm_stats(
        jd, mag, err,
        min_period=float(min_p),
        max_period=float(max_p),
        n_bootstrap=int(n_bootstrap),
        significance_level=float(significance_level),
    )
    ce = compute_ce_stats(
        jd, mag, err,
        min_period=float(min_p),
        max_period=float(max_p),
        n_bootstrap=int(n_bootstrap),
        significance_level=float(significance_level),
    )
    long_ls = long_period_ls_search(jd, mag, err, n_bootstrap=min(int(n_bootstrap), 50))
    event = event_based_period(dip_epochs, baseline_days=baseline_days)

    consensus: dict[str, Any] = {}
    if LONG_PERIOD_ENABLED:
        consensus = compute_period_consensus_for_lc(
            jd,
            mag,
            err,
            pdm_result=pdm,
            ce_result=ce,
            dip_epochs_override=dip_epochs,
            detect_dip_epochs_fallback=False,
            event_period_result=event,
            long_ls_kwargs={"n_bootstrap": min(int(n_bootstrap), 50)},
        )

    def _match(period: float) -> tuple[bool, float, float]:
        return period_match_quality(period, true_period_days, rel_tol=rel_tol)

    pdm_ok, pdm_err, pdm_factor = _match(float(pdm.get("pdm_period", np.nan)))
    ce_ok, ce_err, ce_factor = _match(float(ce.get("ce_period", np.nan)))
    long_ok, long_err, long_factor = _match(float(long_ls.get("long_ls_period_days", np.nan)))
    event_ok, event_err, event_factor = _match(float(event.get("event_period_days", np.nan)))
    cons_ok, cons_err, cons_factor = _match(float(consensus.get("period_consensus_days", np.nan)))

    return {
        "pdm_period_days": float(pdm.get("pdm_period", np.nan)),
        "ce_period_days": float(ce.get("ce_period", np.nan)),
        "long_ls_period_days": float(long_ls.get("long_ls_period_days", np.nan)),
        "event_period_days": float(event.get("event_period_days", np.nan)),
        "consensus_period_days": float(consensus.get("period_consensus_days", np.nan)),
        "consensus_method": str(consensus.get("period_method", "none")),
        "consensus_confidence": str(consensus.get("period_confidence", "none")),
        "pdm_match": bool(pdm_ok),
        "ce_match": bool(ce_ok),
        "long_ls_match": bool(long_ok),
        "event_period_match": bool(event_ok),
        "consensus_match": bool(cons_ok),
        "pdm_rel_error": float(pdm_err),
        "long_ls_rel_error": float(long_err),
        "consensus_rel_error": float(cons_err),
        "pdm_harmonic_factor": float(pdm_factor),
        "long_ls_harmonic_factor": float(long_factor),
        "consensus_harmonic_factor": float(cons_factor),
        "dip_epochs_n": int(len(dip_epochs)),
        "event_period_method": str(event.get("event_period_method", "none")),
    }


def run_benchmark(
    controls: pd.DataFrame,
    config: LongPeriodDipperBenchmarkConfig,
    *,
    load_lightcurve: Any,
) -> LongPeriodDipperBenchmarkRun:
    """Run the full benchmark.

    Parameters
    ----------
    controls:
        DataFrame with at least ``source_path``, ``source_id``, ``jd_span``.
    load_lightcurve:
        Callable ``(path) -> pd.DataFrame`` returning a cleaned LC frame.
    """
    run_dir = make_run_dir(config)
    design = build_trial_design(controls, config)
    rows: list[dict[str, Any]] = []

    for record in design.to_dict(orient="records"):
        path = str(record["source_path"])
        try:
            df_lc = load_lightcurve(path)
        except Exception as exc:
            rows.append({**record, "status": "load_error", "error": str(exc)})
            continue
        if df_lc is None or df_lc.empty:
            rows.append({**record, "status": "empty_lc", "error": "empty"})
            continue

        rng = np.random.default_rng(int(record["trial_seed"]))
        df_inj, dip_epochs = inject_long_period_dips(
            df_lc,
            true_period_days=float(record["true_period_days"]),
            event_count=int(record["requested_event_count"]),
            amplitude=float(record["dip_amplitude"]),
            duration=float(record["dip_duration"]),
            phase0=float(record["phase0"]),
            rng=rng,
        )
        jd = df_inj["JD"].to_numpy(dtype=float)
        mag = df_inj["mag"].to_numpy(dtype=float)
        err = (
            df_inj["error"].to_numpy(dtype=float)
            if "error" in df_inj.columns
            else np.full_like(mag, 0.02)
        )
        baseline = float(record["baseline_days"])
        metrics = evaluate_period_recovery(
            jd,
            mag,
            err,
            dip_epochs=dip_epochs,
            baseline_days=baseline,
            true_period_days=float(record["true_period_days"]),
            rel_tol=float(config.period_rel_tol),
            n_bootstrap=int(config.n_bootstrap),
            significance_level=float(config.significance_level),
        )
        rows.append({**record, **metrics, "status": "ok", "error": None})

    results = pd.DataFrame(rows)
    summary = summarize_results(results)
    results.to_parquet(run_dir / "trial_results.parquet", index=False)
    design.to_parquet(run_dir / "trial_design.parquet", index=False)
    summary.to_parquet(run_dir / "summary.parquet", index=False)
    return LongPeriodDipperBenchmarkRun(
        config=config,
        run_dir=run_dir,
        trial_design=design,
        results=results,
        summary=summary,
    )


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Recall by baseline_cycles × event_count for each method."""
    ok = results[results.get("status", "ok") == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    group_cols = ["baseline_cycles", "requested_event_count"]
    methods = [
        ("pdm_match", "pdm"),
        ("ce_match", "ce"),
        ("long_ls_match", "long_ls"),
        ("event_period_match", "event_period"),
        ("consensus_match", "consensus"),
    ]
    parts: list[pd.DataFrame] = []
    for col, label in methods:
        if col not in ok.columns:
            continue
        part = (
            ok.groupby(group_cols, dropna=False)[col]
            .agg(trials="count", recall="mean")
            .reset_index()
        )
        part["method"] = label
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


__all__ = [
    "LongPeriodDipperBenchmarkConfig",
    "LongPeriodDipperBenchmarkRun",
    "build_trial_design",
    "evaluate_period_recovery",
    "inject_long_period_dips",
    "make_run_dir",
    "run_benchmark",
    "summarize_results",
]
