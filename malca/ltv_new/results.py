"""Result tables for the standalone LTV evidence pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from malca.io.table_io import write_parquet_table
from malca.ltv_new.models import is_astrophysical_model
from malca.ltv_new.samplers import SamplerResult


MODEL_EVIDENCE_FILE = "ltv_new_model_evidence.parquet"
SUMMARY_FILE = "ltv_new_summary.parquet"


def model_result_rows(target_id: str, results: list[SamplerResult]) -> list[dict[str, object]]:
    return [
        {
            "target_id": str(target_id),
            "model_name": result.model_name,
            "logz": float(result.logz),
            "logzerr": float(result.logzerr),
            "status": result.status,
            "backend": result.backend,
            "ncall": int(result.ncall),
            "runtime_sec": float(result.runtime_sec),
            "message": result.message,
        }
        for result in results
    ]


def summarize_target(target_id: str, results: list[SamplerResult]) -> dict[str, object]:
    finite = [r for r in results if r.status == "ok" and np.isfinite(r.logz)]
    by_name = {r.model_name: r for r in finite}
    astro = [r for r in finite if is_astrophysical_model(r.model_name)]
    best_astro = max(astro, key=lambda r: r.logz) if astro else None
    best_overall = max(finite, key=lambda r: r.logz) if finite else None
    flat = by_name.get("flat")
    stochastic = by_name.get("stochastic_drw")

    best_logz = float(best_astro.logz) if best_astro is not None else np.nan
    flat_logz = float(flat.logz) if flat is not None else np.nan
    stochastic_logz = float(stochastic.logz) if stochastic is not None else np.nan
    return {
        "target_id": str(target_id),
        "best_model": best_astro.model_name if best_astro is not None else None,
        "best_logz": best_logz,
        "overall_best_model": best_overall.model_name if best_overall is not None else None,
        "overall_best_logz": float(best_overall.logz) if best_overall is not None else np.nan,
        "flat_logz": flat_logz,
        "stochastic_logz": stochastic_logz,
        "logbf_best_vs_flat": best_logz - flat_logz if np.isfinite(best_logz) and np.isfinite(flat_logz) else np.nan,
        "logbf_best_vs_stochastic": (
            best_logz - stochastic_logz if np.isfinite(best_logz) and np.isfinite(stochastic_logz) else np.nan
        ),
        "n_models_ok": int(len(finite)),
        "n_models_failed": int(len(results) - len(finite)),
        "total_runtime_sec": float(sum(r.runtime_sec for r in results)),
    }


def write_result_tables(
    model_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / MODEL_EVIDENCE_FILE
    summary_path = out_dir / SUMMARY_FILE
    write_parquet_table(pd.DataFrame(model_rows), model_path)
    write_parquet_table(pd.DataFrame(summary_rows), summary_path)
    return model_path, summary_path
