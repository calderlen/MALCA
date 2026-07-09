"""Prior transforms for standalone LTV evidence models."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from malca.ltv_new.likelihood import LightCurveData
from malca.ltv_new.models import get_model_spec


@dataclass(frozen=True)
class PriorSpec:
    name: str
    low: float
    high: float
    scale: str = "linear"

    def transform(self, value: float) -> float:
        u = float(np.clip(value, 0.0, 1.0))
        if self.scale == "log":
            low = max(float(self.low), 1e-12)
            high = max(float(self.high), low * (1.0 + 1e-9))
            return float(math.exp(math.log(low) + u * (math.log(high) - math.log(low))))
        return float(self.low + u * (self.high - self.low))


@dataclass(frozen=True)
class PriorTransform:
    model_name: str
    specs: tuple[PriorSpec, ...]

    @property
    def ndim(self) -> int:
        return len(self.specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def transform(self, unit_cube: np.ndarray) -> dict[str, float]:
        cube = np.asarray(unit_cube, dtype=float)
        if cube.shape[0] != self.ndim:
            raise ValueError(f"{self.model_name} expects {self.ndim} prior dimensions, got {cube.shape[0]}")
        return {spec.name: spec.transform(float(cube[i])) for i, spec in enumerate(self.specs)}

    def transform_array(self, unit_cube: np.ndarray) -> np.ndarray:
        params = self.transform(unit_cube)
        return np.asarray([params[name] for name in self.names], dtype=float)


def _data_scales(data: LightCurveData) -> dict[str, float]:
    mag = data.mag[np.isfinite(data.mag)]
    median = float(np.median(mag)) if mag.size else 0.0
    spread = float(np.percentile(mag, 95) - np.percentile(mag, 5)) if mag.size >= 2 else 0.2
    spread = max(spread, float(np.nanmedian(data.err)) * 5.0 if data.err.size else 0.1, 0.2)
    span = max(float(data.jd_max - data.jd_min), 1.0)
    span_years = max(span / 365.25, 0.05)
    amp_limit = max(2.0 * spread, 0.5)
    slope_limit = max(amp_limit / span_years, 0.1)
    quad_limit = max(amp_limit / (span_years * span_years), 0.05)
    min_scale = max(1.0, span / 500.0)
    max_scale = max(2.0, span * 2.0)
    min_period = max(2.0, span / 1000.0)
    max_period = max(min_period * 1.01, span * 2.0)
    return {
        "median": median,
        "amp_limit": amp_limit,
        "slope_limit": slope_limit,
        "quad_limit": quad_limit,
        "t_min": float(data.jd_min),
        "t_max": float(data.jd_max),
        "min_scale": min_scale,
        "max_scale": max_scale,
        "min_period": min_period,
        "max_period": max_period,
    }


def build_prior_transform(
    model_name: str,
    data: LightCurveData,
    *,
    include_band_offset: bool = True,
    band_offset_limit: float = 1.0,
) -> PriorTransform:
    spec = get_model_spec(model_name)
    scales = _data_scales(data)
    prior_specs: list[PriorSpec] = []
    for name in spec.parameter_names:
        if name == "mu":
            prior_specs.append(
                PriorSpec(name, scales["median"] - scales["amp_limit"], scales["median"] + scales["amp_limit"])
            )
        elif name == "amp":
            prior_specs.append(PriorSpec(name, -scales["amp_limit"], scales["amp_limit"]))
        elif name in {"slope", "slope1", "slope2"}:
            prior_specs.append(PriorSpec(name, -scales["slope_limit"], scales["slope_limit"]))
        elif name == "quad":
            prior_specs.append(PriorSpec(name, -scales["quad_limit"], scales["quad_limit"]))
        elif name == "t0":
            prior_specs.append(PriorSpec(name, scales["t_min"], scales["t_max"]))
        elif name in {"tau", "sigma_rise", "sigma_decay", "tau_rise", "tau_decay", "drw_tau"}:
            prior_specs.append(PriorSpec(name, scales["min_scale"], scales["max_scale"], scale="log"))
        elif name == "period":
            prior_specs.append(PriorSpec(name, scales["min_period"], scales["max_period"], scale="log"))
        elif name == "phase":
            prior_specs.append(PriorSpec(name, -math.pi, math.pi))
        elif name == "drw_sigma":
            prior_specs.append(PriorSpec(name, 0.001, max(scales["amp_limit"], 0.01), scale="log"))
        else:
            raise ValueError(f"No prior configured for parameter '{name}' in model '{model_name}'")

    if include_band_offset and data.has_v_band:
        prior_specs.append(PriorSpec("delta_vg", -float(band_offset_limit), float(band_offset_limit)))

    return PriorTransform(model_name=spec.name, specs=tuple(prior_specs))
