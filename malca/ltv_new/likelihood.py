"""Likelihoods for the standalone LTV evidence pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from malca.ltv_new.models import evaluate_mean, get_model_spec


@dataclass(frozen=True)
class LightCurveData:
    jd: np.ndarray
    mag: np.ndarray
    err: np.ndarray
    band: np.ndarray
    target_id: str = ""

    def __post_init__(self) -> None:
        jd = np.asarray(self.jd, dtype=float)
        mag = np.asarray(self.mag, dtype=float)
        err = np.asarray(self.err, dtype=float)
        band = np.asarray(self.band, dtype=int)
        if not (jd.shape == mag.shape == err.shape == band.shape):
            raise ValueError("jd, mag, err, and band arrays must have the same shape")
        finite = np.isfinite(jd) & np.isfinite(mag) & np.isfinite(err) & (err > 0.0)
        if int(finite.sum()) < 2:
            raise ValueError("Light curve needs at least two finite points with positive errors")
        order = np.argsort(jd[finite])
        object.__setattr__(self, "jd", jd[finite][order])
        object.__setattr__(self, "mag", mag[finite][order])
        object.__setattr__(self, "err", np.maximum(err[finite][order], 1e-6))
        object.__setattr__(self, "band", band[finite][order])

    @property
    def n_points(self) -> int:
        return int(self.jd.size)

    @property
    def jd_min(self) -> float:
        return float(np.min(self.jd))

    @property
    def jd_max(self) -> float:
        return float(np.max(self.jd))

    @property
    def t_ref(self) -> float:
        return float(np.median(self.jd))

    @property
    def has_v_band(self) -> bool:
        return bool(np.any(self.band == 1))

    def g_only(self) -> "LightCurveData":
        mask = self.band == 0
        if int(mask.sum()) < 2:
            raise ValueError("Cannot make g-only light curve with fewer than two g-band points")
        return LightCurveData(
            jd=self.jd[mask],
            mag=self.mag[mask],
            err=self.err[mask],
            band=np.zeros(int(mask.sum()), dtype=int),
            target_id=self.target_id,
        )


def _model_observed_mean(
    data: LightCurveData,
    model_name: str,
    params: dict[str, float],
    *,
    include_band_offset: bool,
) -> np.ndarray:
    mean = evaluate_mean(model_name, data.jd, params, t_ref=data.t_ref)
    if include_band_offset and data.has_v_band:
        mean = mean + float(params.get("delta_vg", 0.0)) * (data.band == 1)
    return mean


def gaussian_log_likelihood(
    data: LightCurveData,
    model_name: str,
    params: dict[str, float],
    *,
    include_band_offset: bool = True,
) -> float:
    spec = get_model_spec(model_name)
    if spec.is_stochastic:
        return drw_log_likelihood(data, params, include_band_offset=include_band_offset)
    mean = _model_observed_mean(data, model_name, params, include_band_offset=include_band_offset)
    resid = data.mag - mean
    var = data.err * data.err
    loglike = -0.5 * np.sum(resid * resid / var + np.log(2.0 * np.pi * var))
    return float(loglike) if np.isfinite(loglike) else -np.inf


def drw_log_likelihood(
    data: LightCurveData,
    params: dict[str, float],
    *,
    include_band_offset: bool = True,
) -> float:
    mean = _model_observed_mean(data, "stochastic_drw", params, include_band_offset=include_band_offset)
    resid = data.mag - mean
    sigma = max(float(params.get("drw_sigma", 0.01)), 1e-8)
    tau = max(float(params.get("drw_tau", 100.0)), 1e-8)
    dt = np.abs(data.jd[:, None] - data.jd[None, :])
    cov = sigma * sigma * np.exp(-dt / tau)
    cov[np.diag_indices_from(cov)] += data.err * data.err
    try:
        chol = np.linalg.cholesky(cov)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, resid))
        logdet = 2.0 * np.sum(np.log(np.diag(chol)))
        loglike = -0.5 * (float(resid @ alpha) + logdet + data.n_points * np.log(2.0 * np.pi))
    except np.linalg.LinAlgError:
        return -np.inf
    return float(loglike) if np.isfinite(loglike) else -np.inf
