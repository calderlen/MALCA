"""CLI for the standalone ``malca ltv-new`` evidence pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from malca.io.table_io import write_parquet_table
from malca.ltv_new.api import fit_ltv_evidence
from malca.ltv_new.io import iter_light_curve_jobs, load_light_curve
from malca.ltv_new.likelihood import LightCurveData
from malca.ltv_new.models import DEFAULT_MODEL_NAMES
from malca.ltv_new.results import MODEL_EVIDENCE_FILE, SUMMARY_FILE, write_result_tables
from malca.ltv_new.samplers import SamplerConfig


def _parse_model_list(value: str | None) -> tuple[str, ...] | None:
    if value is None or not str(value).strip():
        return None
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _sampler_config_from_args(args: argparse.Namespace) -> SamplerConfig:
    return SamplerConfig(
        backend=str(args.backend),
        nlive=int(args.nlive),
        dlogz=float(args.dlogz),
        maxcall=args.maxcall,
        seed=int(args.seed),
        mc_samples=int(args.mc_samples),
    )


def _fit_jobs(args: argparse.Namespace) -> tuple[Path, Path]:
    jobs = iter_light_curve_jobs(args.input)
    if args.max_targets is not None:
        jobs = jobs[: int(args.max_targets)]
    if not jobs:
        raise ValueError(f"No light-curve jobs found in {args.input}")

    model_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    models = _parse_model_list(args.models)
    sampler_config = _sampler_config_from_args(args)

    for job in jobs:
        try:
            data = load_light_curve(job.path, target_id=job.target_id)
            if not args.include_v:
                data = data.g_only()
            result = fit_ltv_evidence(
                data,
                model_names=models,
                include_band_offset=bool(args.include_v),
                sampler_config=sampler_config,
            )
            model_rows.extend(result.model_rows)
            summary = dict(result.summary)
            summary["path"] = str(job.path)
            summary_rows.append(summary)
        except Exception as exc:
            target_id = str(job.target_id)
            summary_rows.append(
                {
                    "target_id": target_id,
                    "best_model": None,
                    "best_logz": np.nan,
                    "overall_best_model": None,
                    "overall_best_logz": np.nan,
                    "flat_logz": np.nan,
                    "stochastic_logz": np.nan,
                    "logbf_best_vs_flat": np.nan,
                    "logbf_best_vs_stochastic": np.nan,
                    "n_models_ok": 0,
                    "n_models_failed": 0,
                    "total_runtime_sec": 0.0,
                    "path": str(job.path),
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )

    return write_result_tables(model_rows, summary_rows, args.output)


def _inject_signal(data: LightCurveData, *, amplitude: float, timescale_days: float, seed: int) -> LightCurveData:
    rng = np.random.default_rng(int(seed))
    t0 = float(rng.uniform(data.jd_min, data.jd_max))
    tau = max(float(timescale_days), 1.0)
    trend = float(amplitude) * (1.0 / (1.0 + np.exp(-np.clip((data.jd - t0) / tau, -80.0, 80.0))))
    return LightCurveData(
        jd=data.jd,
        mag=data.mag + trend,
        err=data.err,
        band=data.band,
        target_id=data.target_id,
    )


def _run_injection(args: argparse.Namespace) -> tuple[Path, Path]:
    jobs = iter_light_curve_jobs(args.manifest)
    if not jobs:
        raise ValueError(f"No injection control jobs found in {args.manifest}")
    rng = np.random.default_rng(int(args.seed))
    amplitudes = np.linspace(float(args.amp_min), float(args.amp_max), int(args.amp_steps))
    timescales = np.logspace(np.log10(float(args.timescale_min)), np.log10(float(args.timescale_max)), int(args.timescale_steps))
    sampler_config = _sampler_config_from_args(args)
    models = _parse_model_list(args.models)

    model_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []
    trial_index = 0
    for amp in amplitudes:
        for timescale in timescales:
            for _ in range(int(args.repeats)):
                job = jobs[int(rng.integers(len(jobs)))]
                try:
                    base = load_light_curve(job.path, target_id=job.target_id)
                    if not args.include_v:
                        base = base.g_only()
                    injected = _inject_signal(base, amplitude=float(amp), timescale_days=float(timescale), seed=int(args.seed) + trial_index)
                    result = fit_ltv_evidence(
                        injected,
                        model_names=models,
                        include_band_offset=bool(args.include_v),
                        sampler_config=sampler_config,
                    )
                    rows = [dict(row, trial_index=trial_index) for row in result.model_rows]
                    model_rows.extend(rows)
                    summary = dict(result.summary)
                    summary.update(
                        {
                            "trial_index": trial_index,
                            "path": str(job.path),
                            "amplitude_mag": float(amp),
                            "timescale_days": float(timescale),
                        }
                    )
                    summary_rows.append(summary)
                    trial_rows.append(summary)
                except Exception as exc:
                    trial_rows.append(
                        {
                            "trial_index": trial_index,
                            "target_id": str(job.target_id),
                            "path": str(job.path),
                            "amplitude_mag": float(amp),
                            "timescale_days": float(timescale),
                            "status": "error",
                            "message": f"{type(exc).__name__}: {exc}",
                        }
                    )
                trial_index += 1

    output_dir = Path(args.output).expanduser()
    model_path, summary_path = write_result_tables(model_rows, summary_rows, output_dir)
    write_parquet_table(pd.DataFrame(trial_rows), output_dir / "ltv_new_injection_trials.parquet")
    return model_path, summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malca ltv-new",
        description="Standalone evidence-based long-term variability modeling.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_sampling(p: argparse.ArgumentParser) -> None:
        p.add_argument("--models", default=",".join(DEFAULT_MODEL_NAMES), help="Comma-separated model names to fit.")
        p.add_argument("--backend", default="auto", choices=["auto", "dynesty", "monte-carlo", "mc"])
        p.add_argument("--nlive", type=int, default=200)
        p.add_argument("--dlogz", type=float, default=0.1)
        p.add_argument("--maxcall", type=int, default=None)
        p.add_argument("--mc-samples", type=int, default=2048)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--include-v", action=argparse.BooleanOptionalAction, default=True)

    fit = sub.add_parser("fit", help="Fit LTV evidence models to one light curve, a directory, or a manifest.")
    fit.add_argument("--input", required=True, help="Light curve file, directory, or manifest table.")
    fit.add_argument("--output", required=True, help="Output directory for evidence tables.")
    fit.add_argument("--max-targets", type=int, default=None)
    add_common_sampling(fit)
    fit.set_defaults(func=_fit_jobs)

    inj = sub.add_parser("injection", help="Run a simple Bayes-factor injection recovery benchmark.")
    inj.add_argument("--manifest", required=True, help="Manifest table or directory of control light curves.")
    inj.add_argument("--output", required=True, help="Output directory for injection evidence tables.")
    inj.add_argument("--amp-min", type=float, default=0.05)
    inj.add_argument("--amp-max", type=float, default=1.0)
    inj.add_argument("--amp-steps", type=int, default=4)
    inj.add_argument("--timescale-min", type=float, default=30.0)
    inj.add_argument("--timescale-max", type=float, default=3000.0)
    inj.add_argument("--timescale-steps", type=int, default=4)
    inj.add_argument("--repeats", type=int, default=1)
    add_common_sampling(inj)
    inj.set_defaults(func=_run_injection)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    model_path, summary_path = args.func(args)
    print(f"Wrote model evidence: {model_path}")
    print(f"Wrote summaries: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
