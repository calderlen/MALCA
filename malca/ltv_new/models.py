"""Deterministic mean models for the standalone LTV evidence pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


ASTROPHYSICAL_MODELS = (
    "linear",
    "quadratic",
    "step",
    "soft_step",
    "piecewise_linear",
    "piecewise_gaussian",
    "sinusoid",
    "trend_plus_periodic",
    "exponential_relaxation",
    "damped_oscillation",
    "fred",
)
NULL_MODELS = ("flat", "stochastic_drw")
DEFAULT_MODEL_NAMES = (*NULL_MODELS, *ASTROPHYSICAL_MODELS)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    parameter_names: tuple[str, ...]
    is_stochastic: bool = False
    is_astrophysical: bool = True


MODEL_SPECS: dict[str, ModelSpec] = {
    "flat": ModelSpec("flat", ("mu",), is_astrophysical=False),
    "linear": ModelSpec("linear", ("mu", "slope")),
    "quadratic": ModelSpec("quadratic", ("mu", "slope", "quad")),
    "step": ModelSpec("step", ("mu", "amp", "t0")),
    "soft_step": ModelSpec("soft_step", ("mu", "amp", "t0", "tau")),
    "piecewise_linear": ModelSpec("piecewise_linear", ("mu", "slope1", "slope2", "t0")),
    "piecewise_gaussian": ModelSpec("piecewise_gaussian", ("mu", "amp", "t0", "sigma_rise", "sigma_decay")),
    "sinusoid": ModelSpec("sinusoid", ("mu", "amp", "period", "phase")),
    "trend_plus_periodic": ModelSpec("trend_plus_periodic", ("mu", "slope", "amp", "period", "phase")),
    "exponential_relaxation": ModelSpec("exponential_relaxation", ("mu", "amp", "t0", "tau")),
    "damped_oscillation": ModelSpec("damped_oscillation", ("mu", "amp", "t0", "tau", "period", "phase")),
    "fred": ModelSpec("fred", ("mu", "amp", "t0", "tau_rise", "tau_decay")),
    "stochastic_drw": ModelSpec(
        "stochastic_drw",
        ("mu", "drw_sigma", "drw_tau"),
        is_stochastic=True,
        is_astrophysical=False,
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[str(name)]
    except KeyError as exc:
        raise ValueError(f"Unknown LTV model: {name}") from exc


def is_astrophysical_model(name: str) -> bool:
    return get_model_spec(name).is_astrophysical


def _centered_years(t: np.ndarray, t_ref: float) -> np.ndarray:
    return (np.asarray(t, dtype=float) - float(t_ref)) / 365.25


def _safe_scale(value: float | np.ndarray, floor: float = 1e-6) -> float | np.ndarray:
    return np.maximum(np.abs(value), floor)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def _normalized_positive_shape(shape: np.ndarray) -> np.ndarray:
    shape = np.asarray(shape, dtype=float)
    finite = np.isfinite(shape)
    if not finite.any():
        return np.zeros_like(shape, dtype=float)
    peak = float(np.nanmax(shape[finite]))
    if peak <= 0.0 or not np.isfinite(peak):
        return np.zeros_like(shape, dtype=float)
    return shape / peak


def evaluate_mean(
    model_name: str,
    t: np.ndarray,
    params: dict[str, float],
    *,
    t_ref: float | None = None,
) -> np.ndarray:
    """Evaluate a mean magnitude model at observation times.

    Parameters are physical quantities in days and magnitudes. ``t_ref`` is the
    reference time for polynomial terms; if omitted it is the median observation
    time.
    """
    name = get_model_spec(model_name).name
    t = np.asarray(t, dtype=float)
    if t_ref is None:
        t_ref = float(np.nanmedian(t)) if t.size else 0.0
    x = _centered_years(t, t_ref)
    mu = float(params.get("mu", 0.0))

    if name in {"flat", "stochastic_drw"}:
        return np.full_like(t, mu, dtype=float)
    if name == "linear":
        return mu + float(params["slope"]) * x
    if name == "quadratic":
        return mu + float(params["slope"]) * x + float(params["quad"]) * x * x
    if name == "step":
        return mu + float(params["amp"]) * (t >= float(params["t0"]))
    if name == "soft_step":
        tau = _safe_scale(float(params["tau"]))
        return mu + float(params["amp"]) * _sigmoid((t - float(params["t0"])) / tau)
    if name == "piecewise_linear":
        t0 = float(params["t0"])
        x0 = (t0 - float(t_ref)) / 365.25
        slope1 = float(params["slope1"])
        slope2 = float(params["slope2"])
        before = mu + slope1 * x
        at_break = mu + slope1 * x0
        after = at_break + slope2 * (x - x0)
        return np.where(t < t0, before, after)
    if name == "piecewise_gaussian":
        t0 = float(params["t0"])
        rise = _safe_scale(float(params["sigma_rise"]))
        decay = _safe_scale(float(params["sigma_decay"]))
        sigma = np.where(t < t0, rise, decay)
        return mu + float(params["amp"]) * np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    if name == "sinusoid":
        period = _safe_scale(float(params["period"]))
        phase = float(params["phase"])
        return mu + float(params["amp"]) * np.sin(2.0 * math.pi * (t - float(t_ref)) / period + phase)
    if name == "trend_plus_periodic":
        period = _safe_scale(float(params["period"]))
        phase = float(params["phase"])
        return (
            mu
            + float(params["slope"]) * x
            + float(params["amp"]) * np.sin(2.0 * math.pi * (t - float(t_ref)) / period + phase)
        )
    if name == "exponential_relaxation":
        dt = t - float(params["t0"])
        tau = _safe_scale(float(params["tau"]))
        return mu + float(params["amp"]) * np.where(dt >= 0.0, np.exp(-dt / tau), 0.0)
    if name == "damped_oscillation":
        dt = t - float(params["t0"])
        tau = _safe_scale(float(params["tau"]))
        period = _safe_scale(float(params["period"]))
        phase = float(params["phase"])
        shape = np.where(
            dt >= 0.0,
            np.exp(-dt / tau) * np.sin(2.0 * math.pi * dt / period + phase),
            0.0,
        )
        return mu + float(params["amp"]) * shape
    if name == "fred":
        dt = t - float(params["t0"])
        rise = _safe_scale(float(params["tau_rise"]))
        decay = _safe_scale(float(params["tau_decay"]))
        positive_dt = np.maximum(dt, 1e-6)
        shape = np.where(dt > 0.0, np.exp(-rise / positive_dt - positive_dt / decay), 0.0)
        return mu + float(params["amp"]) * _normalized_positive_shape(shape)

    raise ValueError(f"Unhandled LTV model: {model_name}")


def validate_model_names(model_names: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if model_names is None:
        return DEFAULT_MODEL_NAMES
    out: list[str] = []
    for name in model_names:
        spec = get_model_spec(str(name))
        if spec.name not in out:
            out.append(spec.name)
    return tuple(out)
