"""Evidence sampler adapters for the standalone LTV pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from malca.ltv_new.likelihood import LightCurveData, gaussian_log_likelihood
from malca.ltv_new.priors import PriorTransform, build_prior_transform


@dataclass(frozen=True)
class SamplerConfig:
    backend: str = "auto"
    nlive: int = 200
    dlogz: float = 0.1
    maxcall: int | None = None
    seed: int = 0
    mc_samples: int = 2048


@dataclass(frozen=True)
class SamplerResult:
    model_name: str
    logz: float
    logzerr: float
    ncall: int
    runtime_sec: float
    backend: str
    status: str
    message: str = ""


def _logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -np.inf
    max_value = float(np.max(finite))
    return float(max_value + np.log(np.sum(np.exp(finite - max_value))))


def _resolved_backend(backend: str) -> str:
    backend = str(backend or "auto").lower().replace("_", "-")
    if backend == "auto":
        try:
            import dynesty  # noqa: F401

            return "dynesty"
        except Exception:
            return "monte-carlo"
    if backend in {"montecarlo", "mc"}:
        return "monte-carlo"
    return backend


def run_model_evidence(
    model_name: str,
    data: LightCurveData,
    *,
    include_band_offset: bool = True,
    sampler_config: SamplerConfig | None = None,
) -> SamplerResult:
    cfg = sampler_config or SamplerConfig()
    prior = build_prior_transform(model_name, data, include_band_offset=include_band_offset)
    backend = _resolved_backend(cfg.backend)
    if backend == "dynesty":
        return _run_dynesty(model_name, data, prior, include_band_offset=include_band_offset, config=cfg)
    if backend == "monte-carlo":
        return _run_monte_carlo(model_name, data, prior, include_band_offset=include_band_offset, config=cfg)
    return SamplerResult(
        model_name=model_name,
        logz=np.nan,
        logzerr=np.nan,
        ncall=0,
        runtime_sec=0.0,
        backend=backend,
        status="error",
        message=f"Unknown evidence backend: {cfg.backend}",
    )


def _run_dynesty(
    model_name: str,
    data: LightCurveData,
    prior: PriorTransform,
    *,
    include_band_offset: bool,
    config: SamplerConfig,
) -> SamplerResult:
    start = time.monotonic()
    try:
        from dynesty import NestedSampler
    except Exception as exc:
        return SamplerResult(
            model_name=model_name,
            logz=np.nan,
            logzerr=np.nan,
            ncall=0,
            runtime_sec=time.monotonic() - start,
            backend="dynesty",
            status="error",
            message=f"dynesty is not installed: {exc}",
        )

    def prior_transform(unit_cube: np.ndarray) -> np.ndarray:
        return prior.transform_array(unit_cube)

    def loglike(theta: np.ndarray) -> float:
        params = {name: float(theta[i]) for i, name in enumerate(prior.names)}
        return gaussian_log_likelihood(data, model_name, params, include_band_offset=include_band_offset)

    try:
        sampler = NestedSampler(
            loglike,
            prior_transform,
            prior.ndim,
            nlive=int(config.nlive),
            rstate=np.random.default_rng(int(config.seed)),
        )
        run_kwargs: dict[str, object] = {"dlogz": float(config.dlogz), "print_progress": False}
        if config.maxcall is not None:
            run_kwargs["maxcall"] = int(config.maxcall)
        sampler.run_nested(**run_kwargs)
        results = sampler.results
        logz = float(results.logz[-1])
        logzerr = float(results.logzerr[-1]) if len(results.logzerr) else np.nan
        ncall = int(np.sum(results.ncall)) if hasattr(results, "ncall") else 0
        return SamplerResult(
            model_name=model_name,
            logz=logz,
            logzerr=logzerr,
            ncall=ncall,
            runtime_sec=time.monotonic() - start,
            backend="dynesty",
            status="ok",
        )
    except Exception as exc:
        return SamplerResult(
            model_name=model_name,
            logz=np.nan,
            logzerr=np.nan,
            ncall=0,
            runtime_sec=time.monotonic() - start,
            backend="dynesty",
            status="error",
            message=f"{type(exc).__name__}: {exc}",
        )


def _run_monte_carlo(
    model_name: str,
    data: LightCurveData,
    prior: PriorTransform,
    *,
    include_band_offset: bool,
    config: SamplerConfig,
) -> SamplerResult:
    start = time.monotonic()
    rng = np.random.default_rng(int(config.seed))
    n = max(1, int(config.mc_samples))
    loglikes = np.empty(n, dtype=float)
    for i in range(n):
        params = prior.transform(rng.random(prior.ndim))
        loglikes[i] = gaussian_log_likelihood(data, model_name, params, include_band_offset=include_band_offset)

    finite = np.isfinite(loglikes)
    if not finite.any():
        return SamplerResult(
            model_name=model_name,
            logz=-np.inf,
            logzerr=np.inf,
            ncall=n,
            runtime_sec=time.monotonic() - start,
            backend="monte-carlo",
            status="error",
            message="all likelihood evaluations were non-finite",
        )
    finite_ll = loglikes[finite]
    norm = _logsumexp(finite_ll)
    logz = float(norm - np.log(float(n)))
    weights = np.exp(finite_ll - norm)
    # A simple importance-weight concentration diagnostic, not a formal nested
    # sampling uncertainty. It keeps smoke tests and fallback runs interpretable.
    ess = 1.0 / float(np.sum(weights * weights))
    logzerr = float(1.0 / np.sqrt(max(ess, 1.0)))
    return SamplerResult(
        model_name=model_name,
        logz=logz,
        logzerr=logzerr,
        ncall=n,
        runtime_sec=time.monotonic() - start,
        backend="monte-carlo",
        status="ok",
    )
