"""Public API for standalone LTV evidence fitting."""

from __future__ import annotations

from dataclasses import dataclass

from malca.ltv_new.likelihood import LightCurveData
from malca.ltv_new.models import validate_model_names
from malca.ltv_new.results import model_result_rows, summarize_target
from malca.ltv_new.samplers import SamplerConfig, SamplerResult, run_model_evidence


@dataclass(frozen=True)
class EvidenceResult:
    target_id: str
    model_results: list[SamplerResult]
    summary: dict[str, object]
    model_rows: list[dict[str, object]]


def fit_ltv_evidence(
    data: LightCurveData,
    *,
    model_names: list[str] | tuple[str, ...] | None = None,
    include_band_offset: bool = True,
    sampler_config: SamplerConfig | None = None,
) -> EvidenceResult:
    models = validate_model_names(model_names)
    results = [
        run_model_evidence(
            model,
            data,
            include_band_offset=include_band_offset,
            sampler_config=sampler_config,
        )
        for model in models
    ]
    target_id = data.target_id or ""
    return EvidenceResult(
        target_id=target_id,
        model_results=results,
        summary=summarize_target(target_id, results),
        model_rows=model_result_rows(target_id, results),
    )
