from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import plotly.graph_objects as go


DEFAULT_CONTINUUM_DEGREE = 9
DEFAULT_CONTINUUM_MODE = "pseudo"
DEFAULT_CONTINUUM_GAP_FACTOR = 5.0
DEFAULT_CONTINUUM_MIN_GAP_ANGSTROM = 2.0
DEFAULT_LINE_NOISE_FACTOR = 10.0
DEFAULT_LINE_WINDOW_AA = 2.5
DEFAULT_MAX_ABSORPTION_LINE_LABELS = 24

PSEUDO_CONTINUUM_MODES = {"pseudo", "pseudocontinuum", "pseudo-continuum", "chipwise"}
PSEUDO_CONTINUUM_EDGE_AA = 15.0
PSEUDO_CONTINUUM_OUTPUT_EDGE_AA = 35.0
PSEUDO_CONTINUUM_ENVELOPE_WINDOW_AA = 60.0
PSEUDO_CONTINUUM_ENVELOPE_PERCENTILE = 90.0
PSEUDO_CONTINUUM_KNOT_SPACING_AA = 60.0
PSEUDO_CONTINUUM_LOW_SIGMA = 2.0
PSEUDO_CONTINUUM_HIGH_SIGMA = 5.0
PSEUDO_CONTINUUM_MAX_ITER = 7
APOGEE_DETECTOR_RANGES_AA = (
    (15145.0, 15810.0),
    (15860.0, 16435.0),
    (16470.0, 16970.0),
)
APOGEE_DETECTOR_BREAKS_AA = (15825.0, 16445.0)
APOGEE_STRONG_STELLAR_LINE_WINDOWS_AA = (
    (15260.0, 15269.0),  # H I Brackett 19
    (15342.0, 15351.0),  # H I Brackett 18
    (15439.0, 15448.0),  # H I Brackett 17
    (15555.0, 15565.0),  # H I Brackett 16
    (15698.0, 15712.0),  # H I Brackett 15
    (15878.0, 15892.0),  # H I Brackett 14
    (16105.0, 16121.0),  # H I Brackett 13
    (16403.0, 16419.0),  # H I Brackett 12
    (16803.0, 16820.0),  # H I Brackett 11
)


@dataclass(frozen=True)
class SpectrumContinuumModel:
    mode: str
    degree: int
    models: tuple[Any, ...]
    segments: tuple[tuple[int, int, float, float], ...]
    valid_pixel_mask: np.ndarray | None = None
    continuum_fit_mask: np.ndarray | None = None
    continuum_output_mask: np.ndarray | None = None


@dataclass(frozen=True)
class SpectrumLineFit:
    line_type: str
    center: float
    center_index: int
    significance: float
    amplitude: float
    mean: float
    stddev: float
    model: Any


@dataclass(frozen=True)
class SpectrumAnalysis:
    spectrum: Any
    continuum_model: Any
    normalized_residual_spectrum: Any
    wavelength: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray | None
    finite: np.ndarray
    continuum: np.ndarray
    valid_pixel_mask: np.ndarray | None
    continuum_fit_mask: np.ndarray | None
    continuum_output_mask: np.ndarray | None
    normalized_flux: np.ndarray
    normalized_residual_flux: np.ndarray
    normalized_flux_err: np.ndarray | None
    relative_flux_err: np.ndarray | None
    detected_lines: Any
    line_fits: list[SpectrumLineFit]

    @property
    def continuum_subtracted_spectrum(self) -> Any:
        return self.normalized_residual_spectrum

    @property
    def continuum_subtracted_flux(self) -> np.ndarray:
        return self.normalized_residual_flux


def build_spectrum_figure(
    wavelength: np.ndarray,
    flux: np.ndarray,
    *,
    flux_err: np.ndarray | None = None,
    survey: str = "",
    candidate_id: str = "",
    redshift: float | None = None,
    theme: str = "dark",
    height: int = 400,
) -> go.Figure:
    """Build a λ vs flux plot for a single spectrum."""
    tokens = _theme_tokens(theme)
    fig = go.Figure()

    bounds = _uncertainty_bounds(wavelength, flux, flux_err)
    spectrum_name = survey or "spectrum"
    if bounds is not None:
        lower, upper = bounds
        fig.add_trace(go.Scatter(
            x=np.concatenate([wavelength, wavelength[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=tokens["error_fill"],
            line=dict(color="rgba(0, 0, 0, 0)", width=0),
            legendgroup="spectrum",
            name=spectrum_name,
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=wavelength,
        y=flux,
        mode="lines",
        line=dict(color=tokens["line_color"], width=1.2),
        name=spectrum_name,
        legendgroup="spectrum",
    ))

    title_parts = []
    if survey:
        title_parts.append(survey)
    if candidate_id:
        title_parts.append(candidate_id)
    if redshift is not None and np.isfinite(redshift):
        title_parts.append(f"z={redshift:.4f}")
    title = " | ".join(title_parts) if title_parts else "Spectrum"

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=tokens["font"])),
        xaxis=dict(
            title="Wavelength (Å)",
            color=tokens["font"],
            gridcolor=tokens["grid"],
            zeroline=False,
        ),
        yaxis=dict(
            title="Flux",
            color=tokens["font"],
            gridcolor=tokens["grid"],
            zeroline=False,
        ),
        paper_bgcolor=tokens["paper_bg"],
        plot_bgcolor=tokens["plot_bg"],
        font=dict(color=tokens["font"]),
        margin=dict(l=60, r=20, t=40, b=50),
        height=height,
        legend=dict(
            bgcolor=tokens["legend_bg"],
            bordercolor=tokens["legend_border"],
        ),
    )
    return fig


def render_spectrum_pdf(
    spectrum_data: Any,
    *,
    survey: str = "",
    candidate_id: str = "",
    redshift: float | None = None,
    title: str = "Spectrum",
    continuum_degree: int = DEFAULT_CONTINUUM_DEGREE,
    continuum_mode: str = DEFAULT_CONTINUUM_MODE,
    continuum_gap_factor: float = DEFAULT_CONTINUUM_GAP_FACTOR,
    continuum_min_gap_angstrom: float = DEFAULT_CONTINUUM_MIN_GAP_ANGSTROM,
    line_noise_factor: float = DEFAULT_LINE_NOISE_FACTOR,
    line_window_AA: float = DEFAULT_LINE_WINDOW_AA,
    max_line_fits: int | None = None,
    absorption_line_labels: Mapping[float, str] | None = None,
    max_absorption_line_labels: int = DEFAULT_MAX_ABSORPTION_LINE_LABELS,
) -> bytes:
    """Render a spectrum as a Matplotlib-backed publication PDF."""
    return _render_spectrum_matplotlib(
        spectrum_data,
        survey=survey,
        candidate_id=candidate_id,
        redshift=redshift,
        title=title,
        output_format="pdf",
        dpi=300,
        continuum_degree=continuum_degree,
        continuum_mode=continuum_mode,
        continuum_gap_factor=continuum_gap_factor,
        continuum_min_gap_angstrom=continuum_min_gap_angstrom,
        line_noise_factor=line_noise_factor,
        line_window_AA=line_window_AA,
        max_line_fits=max_line_fits,
        absorption_line_labels=absorption_line_labels,
        max_absorption_line_labels=max_absorption_line_labels,
    )


def render_spectrum_png(
    spectrum_data: Any,
    *,
    survey: str = "",
    candidate_id: str = "",
    redshift: float | None = None,
    title: str = "Spectrum",
    continuum_degree: int = DEFAULT_CONTINUUM_DEGREE,
    continuum_mode: str = DEFAULT_CONTINUUM_MODE,
    continuum_gap_factor: float = DEFAULT_CONTINUUM_GAP_FACTOR,
    continuum_min_gap_angstrom: float = DEFAULT_CONTINUUM_MIN_GAP_ANGSTROM,
    line_noise_factor: float = DEFAULT_LINE_NOISE_FACTOR,
    line_window_AA: float = DEFAULT_LINE_WINDOW_AA,
    max_line_fits: int | None = None,
    absorption_line_labels: Mapping[float, str] | None = None,
    max_absorption_line_labels: int = DEFAULT_MAX_ABSORPTION_LINE_LABELS,
) -> bytes:
    """Render a spectrum as a Matplotlib PNG for static review previews."""
    return _render_spectrum_matplotlib(
        spectrum_data,
        survey=survey,
        candidate_id=candidate_id,
        redshift=redshift,
        title=title,
        output_format="png",
        dpi=180,
        continuum_degree=continuum_degree,
        continuum_mode=continuum_mode,
        continuum_gap_factor=continuum_gap_factor,
        continuum_min_gap_angstrom=continuum_min_gap_angstrom,
        line_noise_factor=line_noise_factor,
        line_window_AA=line_window_AA,
        max_line_fits=max_line_fits,
        absorption_line_labels=absorption_line_labels,
        max_absorption_line_labels=max_absorption_line_labels,
    )


def _render_spectrum_matplotlib(
    spectrum_data: Any,
    *,
    survey: str,
    candidate_id: str,
    redshift: float | None,
    title: str,
    output_format: str,
    dpi: int,
    continuum_degree: int,
    continuum_mode: str,
    continuum_gap_factor: float,
    continuum_min_gap_angstrom: float,
    line_noise_factor: float,
    line_window_AA: float,
    max_line_fits: int | None,
    absorption_line_labels: Mapping[float, str] | None,
    max_absorption_line_labels: int,
) -> bytes:
    from malca.plotting.lightcurve_publication import PUBLICATION_STYLE
    from malca.review.publication import _matplotlib_imports

    plt, _to_rgba, _Circle, _Rectangle = _matplotlib_imports()
    analysis = analyze_spectrum(
        spectrum_data,
        continuum_degree=continuum_degree,
        continuum_mode=continuum_mode,
        continuum_gap_factor=continuum_gap_factor,
        continuum_min_gap_angstrom=continuum_min_gap_angstrom,
        line_noise_factor=line_noise_factor,
        line_window_AA=line_window_AA,
        max_line_fits=max_line_fits,
    )
    wavelength = analysis.wavelength
    flux = analysis.flux
    flux_err = analysis.flux_err
    finite = analysis.finite
    continuum = analysis.continuum
    normalized_residual = analysis.normalized_residual_flux
    line_fits_to_draw = _strongest_line_fits(analysis.line_fits, max_draw=50)
    max_absorption_line_markers = min(90, max(60, max_absorption_line_labels * 3))
    absorption_line_fits_to_draw = _strongest_line_fits(
        [line_fit for line_fit in analysis.line_fits if line_fit.line_type == "absorption"],
        max_draw=max_absorption_line_markers,
    )
    absorption_labels_to_draw = _labeled_line_fits(
        absorption_line_fits_to_draw,
        absorption_line_labels,
        max_labels=max_absorption_line_labels,
        min_spacing_angstrom=42.0,
    )
    plot_gap_threshold = _plot_gap_threshold(
        wavelength,
        finite,
        gap_factor=continuum_gap_factor,
        min_gap_angstrom=continuum_min_gap_angstrom,
    )

    n_panels = 2
    height_ratios = [2.2, 1.7]
    figsize = (7.25, 5.15)

    with plt.rc_context(PUBLICATION_STYLE | {
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    }):
        fig, axes = plt.subplots(
            n_panels,
            1,
            figsize=figsize,
            sharex=True,
            constrained_layout=False,
            gridspec_kw={"height_ratios": height_ratios},
        )
        fig.subplots_adjust(left=0.105, right=0.985, top=0.90, bottom=0.13, hspace=0.38)
        axes = np.atleast_1d(axes)

        title_parts = [part for part in (survey, str(candidate_id or "")) if part]
        if redshift is not None and np.isfinite(redshift):
            title_parts.append(f"z={redshift:.4f}")
        if analysis.line_fits:
            title_parts.append(f"{len(analysis.line_fits)} lines")
        fig.suptitle(" | ".join(title_parts) or title, fontsize=11.5, color="black")

        bounds = _uncertainty_bounds(wavelength, flux, flux_err, finite=finite)
        if bounds is not None:
            lower, upper = bounds
            _fill_between_segments(
                axes[0],
                wavelength,
                lower,
                upper,
                finite=finite,
                max_gap_angstrom=plot_gap_threshold,
                color="black",
                alpha=0.08,
                linewidth=0,
            )
        _plot_segments(axes[0], wavelength, flux, finite=finite, max_gap_angstrom=plot_gap_threshold, color="black", lw=0.65)
        _plot_segments(
            axes[0],
            wavelength,
            continuum,
            finite=finite,
            max_gap_angstrom=plot_gap_threshold,
            color="black",
            lw=0.85,
            ls="--",
            alpha=0.75,
        )
        raw_limits = np.concatenate([flux[finite], continuum[finite & np.isfinite(continuum)]])
        axes[0].set_ylim(*_robust_limits(raw_limits, 0.5, 99.5, min_pad=1.0))
        axes[0].set_ylabel(r"$F_\lambda$")

        _plot_segments(
            axes[1],
            wavelength,
            normalized_residual,
            finite=finite,
            max_gap_angstrom=plot_gap_threshold,
            color="black",
            lw=0.65,
        )
        axes[1].axhline(0.0, color="black", lw=0.55, alpha=0.45)
        residual_pad = _scale_pad(normalized_residual, fraction=0.03, minimum=1e-3)
        residual_limits = _robust_limits(normalized_residual, 0.5, 99.5, min_pad=residual_pad)
        axes[1].set_ylim(*residual_limits)
        for line_fit in absorption_line_fits_to_draw:
            axes[1].axvline(line_fit.center, color="#cc0000", lw=0.24, alpha=0.10, zorder=0.5)
        for line_fit, _label in absorption_labels_to_draw:
            axes[1].axvline(line_fit.center, color="#cc0000", lw=0.46, alpha=0.30, zorder=1.2)
        _draw_interpanel_line_labels(
            fig,
            axes[0],
            axes[1],
            absorption_labels_to_draw,
            color="#cc0000",
            fontsize=7.4,
        )
        for line_fit in line_fits_to_draw:
            local = np.abs(wavelength - line_fit.mean) <= max(4.0 * line_fit.stddev, 0.4)
            if np.count_nonzero(local) < 2:
                continue
            try:
                from astropy import units as u

                model_quantity = line_fit.model(wavelength[local] * u.AA)
                if hasattr(model_quantity, "unit"):
                    model_flux = np.asarray(model_quantity.to_value(analysis.normalized_residual_spectrum.flux.unit), dtype=np.float64)
                else:
                    model_flux = np.asarray(model_quantity, dtype=np.float64)
            except Exception:
                continue
            axes[1].plot(wavelength[local], model_flux, color="black", lw=0.45, alpha=0.35)
        axes[1].set_ylabel(r"$F_\lambda/C_\lambda - 1$")

        axes[-1].set_xlabel(r"$\lambda$ [Å]")
        for ax in axes:
            ax.grid(True, color="black", alpha=0.12, linewidth=0.45)
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(0.75)
            ax.tick_params(color="black", labelcolor="black", width=0.75)

        buf = BytesIO()
        try:
            metadata = {"Creator": "MALCA"} if output_format == "pdf" else None
            fig.savefig(buf, format=output_format, metadata=metadata, dpi=dpi, bbox_inches=None)
            return buf.getvalue()
        finally:
            plt.close(fig)


def analyze_spectrum(
    spectrum_data: Any,
    *,
    continuum_degree: int = DEFAULT_CONTINUUM_DEGREE,
    continuum_mode: str = DEFAULT_CONTINUUM_MODE,
    continuum_gap_factor: float = DEFAULT_CONTINUUM_GAP_FACTOR,
    continuum_min_gap_angstrom: float = DEFAULT_CONTINUUM_MIN_GAP_ANGSTROM,
    line_noise_factor: float = DEFAULT_LINE_NOISE_FACTOR,
    line_window_AA: float = DEFAULT_LINE_WINDOW_AA,
    max_line_fits: int | None = None,
) -> SpectrumAnalysis:
    """Normalize and summarize a spectrum through a specutils Spectrum object."""
    spectrum = _as_specutils_spectrum(spectrum_data)
    continuum_model, continuum, residual_spectrum = _fit_normalized_residual_spectrum(
        spectrum,
        degree=continuum_degree,
        mode=continuum_mode,
        gap_factor=continuum_gap_factor,
        min_gap_angstrom=continuum_min_gap_angstrom,
    )
    wavelength, flux, flux_err = _arrays_from_specutils(spectrum)
    continuum = np.asarray(continuum, dtype=np.float64)
    finite = np.isfinite(wavelength) & np.isfinite(flux) & np.isfinite(continuum)
    normalized_residual = np.asarray(residual_spectrum.flux.value, dtype=np.float64)
    normalized_flux = normalized_residual + 1.0
    rel_err = None
    normalized_flux_err = _uncertainty_values(residual_spectrum.uncertainty, residual_spectrum.flux.unit)
    if len(normalized_flux_err) != len(wavelength):
        normalized_flux_err = None
    if flux_err is not None:
        err_finite = finite & np.isfinite(flux_err) & (flux != 0)
        rel_err = np.where(err_finite, np.abs(flux_err / flux), np.nan)
    detected_lines = _find_significant_lines(residual_spectrum, noise_factor=line_noise_factor)
    line_fits = _fit_significant_lines(
        residual_spectrum,
        detected_lines,
        line_window_AA=line_window_AA,
        max_line_fits=max_line_fits,
    )
    return SpectrumAnalysis(
        spectrum=spectrum,
        continuum_model=continuum_model,
        normalized_residual_spectrum=residual_spectrum,
        wavelength=wavelength,
        flux=flux,
        flux_err=flux_err,
        finite=finite,
        continuum=continuum,
        valid_pixel_mask=getattr(continuum_model, "valid_pixel_mask", None),
        continuum_fit_mask=getattr(continuum_model, "continuum_fit_mask", None),
        continuum_output_mask=getattr(continuum_model, "continuum_output_mask", None),
        normalized_flux=normalized_flux,
        normalized_residual_flux=normalized_residual,
        normalized_flux_err=normalized_flux_err,
        relative_flux_err=rel_err,
        detected_lines=detected_lines,
        line_fits=line_fits,
    )


def line_fits_to_dataframe(analysis: SpectrumAnalysis) -> Any:
    import pandas as pd

    rows = [
        {
            "line_center": line_fit.center,
            "line_type": line_fit.line_type,
            "line_center_index": line_fit.center_index,
            "significance": line_fit.significance,
            "fit_amplitude": line_fit.amplitude,
            "fit_mean": line_fit.mean,
            "fit_stddev": line_fit.stddev,
            "fit_fwhm": 2.354820045 * line_fit.stddev,
            "signed_equivalent_width": -line_fit.amplitude * np.sqrt(2.0 * np.pi) * line_fit.stddev,
        }
        for line_fit in analysis.line_fits
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "line_center",
            "line_type",
            "line_center_index",
            "significance",
            "fit_amplitude",
            "fit_mean",
            "fit_stddev",
            "fit_fwhm",
            "signed_equivalent_width",
        ],
    )


def _as_specutils_spectrum(spectrum_data: Any) -> Any:
    if hasattr(spectrum_data, "to_specutils"):
        return spectrum_data.to_specutils()
    if hasattr(spectrum_data, "spectral_axis") and hasattr(spectrum_data, "flux"):
        return spectrum_data

    from malca.enrich.spectrum_fetch import SpectrumData

    return SpectrumData(
        wavelength=np.asarray(spectrum_data.wavelength, dtype=np.float64),
        flux=np.asarray(spectrum_data.flux, dtype=np.float64),
        flux_err=np.asarray(spectrum_data.flux_err, dtype=np.float64) if spectrum_data.flux_err is not None else None,
    ).to_specutils()


def _arrays_from_specutils(spectrum: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    try:
        from astropy import units as u
    except ImportError as exc:
        raise ImportError("astropy is required for specutils spectrum plotting") from exc

    wavelength = np.asarray(spectrum.spectral_axis.to_value(u.AA), dtype=np.float64)
    flux = np.asarray(spectrum.flux.value, dtype=np.float64)
    flux_err = None
    uncertainty = getattr(spectrum, "uncertainty", None)
    if uncertainty is not None:
        quantity = getattr(uncertainty, "quantity", None)
        if quantity is None and getattr(uncertainty, "array", None) is not None:
            quantity = uncertainty.array * spectrum.flux.unit
        if quantity is not None:
            flux_err = np.asarray(quantity.to_value(spectrum.flux.unit), dtype=np.float64)
    return wavelength, flux, flux_err


def _fit_normalized_residual_spectrum(
    spectrum: Any,
    *,
    degree: int,
    mode: str,
    gap_factor: float,
    min_gap_angstrom: float,
) -> tuple[SpectrumContinuumModel, np.ndarray, Any]:
    import warnings

    from astropy import units as u
    from astropy.modeling import fitting, models
    from specutils import Spectrum
    from specutils.fitting import fit_continuum

    wavelength = np.asarray(spectrum.spectral_axis.to_value(u.AA), dtype=np.float64)
    flux_values = np.asarray(spectrum.flux.to_value(spectrum.flux.unit), dtype=np.float64)
    finite = np.isfinite(wavelength) & np.isfinite(flux_values)
    if not np.any(finite):
        raise ValueError("Cannot fit continuum: spectrum has no finite flux samples")

    mode = str(mode or "global").strip().lower()
    if mode not in {"global", "segmented"} and mode not in PSEUDO_CONTINUUM_MODES:
        raise ValueError("continuum_mode must be 'pseudo', 'global', or 'segmented'")

    if mode in PSEUDO_CONTINUUM_MODES:
        continuum_model, continuum = _fit_pseudo_continuum(
            wavelength,
            flux_values,
            flux_err=_spectrum_flux_err_values(spectrum, len(wavelength)),
            degree=max(1, int(degree)),
            gap_factor=gap_factor,
            min_gap_angstrom=min_gap_angstrom,
        )
        normalized_residual_values = np.full_like(flux_values, np.nan, dtype=np.float64)
        valid_continuum = np.isfinite(flux_values) & np.isfinite(continuum) & (continuum != 0)
        normalized_residual_values[valid_continuum] = flux_values[valid_continuum] / continuum[valid_continuum] - 1.0
        residual_flux = normalized_residual_values * u.dimensionless_unscaled
        residual_uncertainty = _normalized_residual_uncertainty(
            spectrum,
            continuum,
            normalized_residual_values,
        )
        residual_spectrum = Spectrum(
            spectral_axis=spectrum.spectral_axis,
            flux=residual_flux,
            uncertainty=residual_uncertainty,
        )
        return continuum_model, continuum, residual_spectrum

    finite_indices = np.flatnonzero(finite)
    if mode == "segmented":
        segments = _continuum_segments(
            wavelength,
            finite_indices,
            gap_factor=gap_factor,
            min_gap_angstrom=min_gap_angstrom,
        )
    else:
        segments = [finite_indices]

    continuum = np.full_like(wavelength, np.nan, dtype=np.float64)
    fitted_models: list[Any] = []
    segment_meta: list[tuple[int, int, float, float]] = []
    requested_degree = max(1, int(degree))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Model is linear in parameters.*")
        for segment in segments:
            if len(segment) < 4:
                continue
            segment_degree = min(requested_degree, max(1, len(segment) - 2))
            domain = [float(wavelength[segment[0]]), float(wavelength[segment[-1]])]
            continuum_seed = models.Chebyshev1D(segment_degree, domain=domain)
            segment_spectrum = Spectrum(
                spectral_axis=spectrum.spectral_axis[segment],
                flux=spectrum.flux[segment],
            )
            continuum_model = fit_continuum(
                segment_spectrum,
                model=continuum_seed,
                fitter=fitting.LinearLSQFitter(),
            )
            continuum_quantity = continuum_model(segment_spectrum.spectral_axis)
            if hasattr(continuum_quantity, "unit"):
                continuum_values = np.asarray(continuum_quantity.to_value(spectrum.flux.unit), dtype=np.float64)
            else:
                continuum_values = np.asarray(continuum_quantity, dtype=np.float64)
            continuum[segment] = continuum_values
            fitted_models.append(continuum_model)
            segment_meta.append((int(segment[0]), int(segment[-1]), domain[0], domain[1]))

    if not fitted_models:
        raise ValueError("Cannot fit continuum: no valid continuum segments")

    normalized_residual_values = np.full_like(flux_values, np.nan, dtype=np.float64)
    valid_continuum = np.isfinite(flux_values) & np.isfinite(continuum) & (continuum != 0)
    normalized_residual_values[valid_continuum] = flux_values[valid_continuum] / continuum[valid_continuum] - 1.0
    residual_flux = normalized_residual_values * u.dimensionless_unscaled
    residual_uncertainty = _normalized_residual_uncertainty(
        spectrum,
        continuum,
        normalized_residual_values,
    )
    residual_spectrum = Spectrum(
        spectral_axis=spectrum.spectral_axis,
        flux=residual_flux,
        uncertainty=residual_uncertainty,
    )
    continuum_model = SpectrumContinuumModel(
        mode=mode,
        degree=requested_degree,
        models=tuple(fitted_models),
        segments=tuple(segment_meta),
    )
    return continuum_model, continuum, residual_spectrum


def _fit_pseudo_continuum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    *,
    flux_err: np.ndarray | None,
    degree: int,
    gap_factor: float,
    min_gap_angstrom: float,
) -> tuple[SpectrumContinuumModel, np.ndarray]:
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    valid = np.isfinite(wavelength) & np.isfinite(flux) & (flux > 0)
    fit_exclusion_mask = np.zeros_like(valid, dtype=bool)
    if _looks_like_apogee_h_band(wavelength, np.flatnonzero(valid)):
        valid &= _apogee_detector_range_mask(wavelength)
        fit_exclusion_mask |= _apogee_strong_stellar_line_mask(wavelength)
    valid_pixel_mask = valid.copy()
    if not np.any(valid):
        raise ValueError("Cannot fit continuum: spectrum has no positive finite flux samples")

    segments = _chipwise_continuum_segments(
        wavelength,
        np.flatnonzero(valid),
        gap_factor=gap_factor,
        min_gap_angstrom=min_gap_angstrom,
    )
    continuum = np.full_like(flux, np.nan, dtype=np.float64)
    continuum_fit_mask = np.zeros_like(valid, dtype=bool)
    continuum_output_mask = np.zeros_like(valid, dtype=bool)
    fitted_models: list[Any] = []
    segment_meta: list[tuple[int, int, float, float]] = []

    for segment in segments:
        if len(segment) < 4:
            continue
        segment_continuum, segment_fit_mask, segment_model = _fit_pseudo_continuum_segment(
            wavelength,
            flux,
            flux_err,
            segment,
            fit_exclusion_mask=fit_exclusion_mask,
        )
        if segment_continuum is None or not np.any(np.isfinite(segment_continuum)):
            continue
        segment_output_mask = _chip_edge_fit_mask(
            wavelength[segment],
            edge_angstrom=PSEUDO_CONTINUUM_OUTPUT_EDGE_AA,
        )
        continuum[segment] = segment_continuum
        continuum_fit_mask[segment] = segment_fit_mask
        continuum_output_mask[segment] = segment_output_mask
        fitted_models.append(segment_model)
        segment_meta.append((
            int(segment[0]),
            int(segment[-1]),
            float(wavelength[segment[0]]),
            float(wavelength[segment[-1]]),
        ))

    if not fitted_models:
        raise ValueError("Cannot fit continuum: no valid pseudo-continuum segments")

    continuum = _repair_continuum_values(wavelength, flux, continuum, valid & continuum_output_mask)
    continuum[~continuum_output_mask] = np.nan
    return (
        SpectrumContinuumModel(
            mode="pseudo",
            degree=degree,
            models=tuple(fitted_models),
            segments=tuple(segment_meta),
            valid_pixel_mask=valid_pixel_mask,
            continuum_fit_mask=continuum_fit_mask,
            continuum_output_mask=continuum_output_mask,
        ),
        continuum,
    )


def _fit_pseudo_continuum_segment(
    wavelength: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None,
    segment: np.ndarray,
    *,
    fit_exclusion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray, Any]:
    x = wavelength[segment]
    y = flux[segment]
    finite = np.isfinite(x) & np.isfinite(y) & (y > 0)
    excluded = (
        np.asarray(fit_exclusion_mask[segment], dtype=bool)
        if fit_exclusion_mask is not None and len(fit_exclusion_mask) >= int(segment[-1]) + 1
        else np.zeros_like(finite, dtype=bool)
    )
    fit_base = finite & ~excluded & _chip_edge_fit_mask(x, edge_angstrom=PSEUDO_CONTINUUM_EDGE_AA)
    if np.count_nonzero(fit_base) < 8:
        fit_base = finite & ~excluded
    if np.count_nonzero(fit_base) < 4:
        fit_base = finite.copy()
    if np.count_nonzero(fit_base) < 4:
        return None, np.zeros_like(finite, dtype=bool), None

    envelope = _rolling_upper_envelope(
        x,
        y,
        fit_base,
        window_angstrom=PSEUDO_CONTINUUM_ENVELOPE_WINDOW_AA,
        percentile=PSEUDO_CONTINUUM_ENVELOPE_PERCENTILE,
    )
    envelope_residual = np.full_like(y, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        envelope_residual[fit_base] = y[fit_base] / envelope[fit_base] - 1.0
    envelope_sigma = _robust_sigma(envelope_residual[fit_base])
    if not np.isfinite(envelope_sigma) or envelope_sigma <= 0:
        envelope_sigma = 0.02

    good = (
        fit_base
        & np.isfinite(envelope_residual)
        & (envelope_residual > -PSEUDO_CONTINUUM_LOW_SIGMA * envelope_sigma)
        & (envelope_residual < PSEUDO_CONTINUUM_HIGH_SIGMA * envelope_sigma)
    )
    if np.count_nonzero(good) < max(8, min(25, np.count_nonzero(fit_base) // 3)):
        good = _upper_fraction_mask(y, fit_base, fraction=0.55)

    last_model: Any = None
    continuum = envelope.copy()
    for _iteration in range(PSEUDO_CONTINUUM_MAX_ITER):
        weights = _fit_weights(flux_err, segment, good)
        candidate_continuum, candidate_model = _fit_smooth_continuum(
            x[good],
            y[good],
            x,
            weights=weights,
            knot_spacing_angstrom=PSEUDO_CONTINUUM_KNOT_SPACING_AA,
        )
        candidate_continuum = _repair_segment_continuum(x, y, candidate_continuum, envelope)
        with np.errstate(divide="ignore", invalid="ignore"):
            residual = y / candidate_continuum - 1.0
        sigma = _robust_sigma(residual[fit_base])
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = envelope_sigma
        new_good = (
            fit_base
            & np.isfinite(residual)
            & (residual > -PSEUDO_CONTINUUM_LOW_SIGMA * sigma)
            & (residual < PSEUDO_CONTINUUM_HIGH_SIGMA * sigma)
        )
        if np.count_nonzero(new_good) < max(8, min(25, np.count_nonzero(fit_base) // 3)):
            new_good = _upper_fraction_mask(residual, fit_base & np.isfinite(residual), fraction=0.55)

        changed = np.count_nonzero(new_good != good) / max(1, len(good))
        continuum = candidate_continuum
        last_model = candidate_model
        good = new_good
        if changed < 0.005:
            break

    if np.count_nonzero(good) >= 4:
        weights = _fit_weights(flux_err, segment, good)
        continuum, last_model = _fit_smooth_continuum(
            x[good],
            y[good],
            x,
            weights=weights,
            knot_spacing_angstrom=PSEUDO_CONTINUUM_KNOT_SPACING_AA,
        )
        continuum = _repair_segment_continuum(x, y, continuum, envelope)

    return continuum, good, last_model


def _chipwise_continuum_segments(
    wavelength: np.ndarray,
    finite_indices: np.ndarray,
    *,
    gap_factor: float,
    min_gap_angstrom: float,
) -> list[np.ndarray]:
    segments = _continuum_segments(
        wavelength,
        finite_indices,
        gap_factor=gap_factor,
        min_gap_angstrom=min_gap_angstrom,
    )
    if not _looks_like_apogee_h_band(wavelength, finite_indices):
        return segments

    split_segments: list[np.ndarray] = []
    for segment in segments:
        split_segments.extend(_split_segment_at_wavelengths(segment, wavelength, APOGEE_DETECTOR_BREAKS_AA))
    return [segment for segment in split_segments if len(segment)]


def _looks_like_apogee_h_band(wavelength: np.ndarray, finite_indices: np.ndarray) -> bool:
    if len(finite_indices) < 50:
        return False
    values = wavelength[finite_indices]
    min_wave = float(np.nanmin(values))
    max_wave = float(np.nanmax(values))
    return min_wave < 15350.0 and max_wave > 16800.0


def _apogee_detector_range_mask(wavelength: np.ndarray) -> np.ndarray:
    values = np.asarray(wavelength, dtype=np.float64)
    mask = np.zeros_like(values, dtype=bool)
    for lo, hi in APOGEE_DETECTOR_RANGES_AA:
        mask |= (values >= lo) & (values <= hi)
    return mask


def _apogee_strong_stellar_line_mask(wavelength: np.ndarray) -> np.ndarray:
    values = np.asarray(wavelength, dtype=np.float64)
    mask = np.zeros_like(values, dtype=bool)
    for lo, hi in APOGEE_STRONG_STELLAR_LINE_WINDOWS_AA:
        mask |= (values >= lo) & (values <= hi)
    return mask


def _split_segment_at_wavelengths(
    segment: np.ndarray,
    wavelength: np.ndarray,
    breakpoints: tuple[float, ...],
) -> list[np.ndarray]:
    if len(segment) < 2:
        return [segment]
    values = wavelength[segment]
    split_positions = []
    for breakpoint in breakpoints:
        pos = int(np.searchsorted(values, breakpoint, side="left"))
        if 0 < pos < len(segment):
            split_positions.append(pos)
    if not split_positions:
        return [segment]
    return [part for part in np.split(segment, sorted(set(split_positions))) if len(part)]


def _chip_edge_fit_mask(wavelength: np.ndarray, *, edge_angstrom: float) -> np.ndarray:
    values = np.asarray(wavelength, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.zeros_like(values, dtype=bool)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if hi - lo <= 3.0 * float(edge_angstrom):
        return np.isfinite(values)
    return np.isfinite(values) & (values >= lo + edge_angstrom) & (values <= hi - edge_angstrom)


def _rolling_upper_envelope(
    wavelength: np.ndarray,
    flux: np.ndarray,
    valid: np.ndarray,
    *,
    window_angstrom: float,
    percentile: float,
) -> np.ndarray:
    import pandas as pd

    values = np.where(valid, np.asarray(flux, dtype=np.float64), np.nan)
    median_step = _median_wavelength_step(wavelength[valid])
    if not np.isfinite(median_step) or median_step <= 0:
        median_step = 1.0
    window = int(round(float(window_angstrom) / median_step))
    window = max(15, min(len(values), window))
    if window % 2 == 0:
        window += 1
    min_periods = max(5, min(window, window // 5))

    rolled = pd.Series(values).rolling(
        window=window,
        center=True,
        min_periods=min_periods,
    ).quantile(float(percentile) / 100.0)
    envelope = rolled.interpolate(limit_direction="both").to_numpy(dtype=np.float64)

    finite_envelope = np.isfinite(envelope)
    if np.count_nonzero(finite_envelope) == 0:
        fallback = np.nanpercentile(values[valid], percentile) if np.any(valid) else np.nan
        envelope = np.full_like(values, fallback, dtype=np.float64)
    else:
        envelope = _interpolate_finite(wavelength, envelope)
    return envelope


def _fit_smooth_continuum(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray,
    *,
    weights: np.ndarray | None,
    knot_spacing_angstrom: float,
) -> tuple[np.ndarray, Any]:
    from scipy.interpolate import LSQUnivariateSpline

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if weights is not None and len(weights) == len(good):
        weights = np.asarray(weights, dtype=np.float64)
        good &= np.isfinite(weights) & (weights > 0)
    x = x[good]
    y = y[good]
    w = np.asarray(weights, dtype=np.float64)[good] if weights is not None and len(weights) == len(good) else None
    if len(x) < 4:
        fallback = np.full_like(x_eval, np.nanmedian(y) if len(y) else np.nan, dtype=np.float64)
        return _drop_fit_extrapolation(x, x_eval, fallback), None

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    if w is not None:
        w = w[order]
    x, unique_idx = np.unique(x, return_index=True)
    y = y[unique_idx]
    if w is not None:
        w = w[unique_idx]
        median_weight = np.nanmedian(w[np.isfinite(w) & (w > 0)])
        if np.isfinite(median_weight) and median_weight > 0:
            w = w / median_weight

    if len(x) < 4 or float(x[-1] - x[0]) <= 0:
        fallback = np.full_like(x_eval, np.nanmedian(y), dtype=np.float64)
        return fallback, None

    knot_spacing = max(float(knot_spacing_angstrom), 5.0)
    knots = np.arange(x[0] + knot_spacing, x[-1] - knot_spacing * 0.5, knot_spacing)
    knots = knots[(knots > x[1]) & (knots < x[-2])]
    try:
        if len(knots) > 0 and len(x) > len(knots) + 4:
            model = LSQUnivariateSpline(x, y, t=knots, w=w, k=3)
            values = np.asarray(model(x_eval), dtype=np.float64)
            return _drop_fit_extrapolation(x, x_eval, values), model
    except Exception:
        pass

    degree = min(3, len(x) - 1)
    coeff = np.polyfit(x - np.nanmedian(x), y, deg=degree, w=w)
    values = np.polyval(coeff, x_eval - np.nanmedian(x))
    return _drop_fit_extrapolation(x, x_eval, values), coeff


def _drop_fit_extrapolation(x_fit: np.ndarray, x_eval: np.ndarray, values: np.ndarray) -> np.ndarray:
    x_fit = np.asarray(x_fit, dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    finite_fit = x_fit[np.isfinite(x_fit)]
    if len(finite_fit) < 2 or len(values) != len(x_eval):
        return values
    lo = float(np.nanmin(finite_fit))
    hi = float(np.nanmax(finite_fit))
    clipped = values.copy()
    clipped[(x_eval < lo) | (x_eval > hi)] = np.nan
    return clipped


def _fit_weights(flux_err: np.ndarray | None, segment: np.ndarray, good: np.ndarray) -> np.ndarray | None:
    if flux_err is None or len(flux_err) <= int(np.nanmax(segment)):
        return None
    err = np.asarray(flux_err[segment], dtype=np.float64)
    if len(err) != len(good):
        return None
    valid = good & np.isfinite(err) & (err > 0)
    if not np.any(valid):
        return None
    weights = np.full_like(err, np.nan, dtype=np.float64)
    weights[valid] = 1.0 / err[valid]
    return weights[good]


def _upper_fraction_mask(values: np.ndarray, valid: np.ndarray, *, fraction: float) -> np.ndarray:
    mask = np.zeros_like(valid, dtype=bool)
    usable = valid & np.isfinite(values)
    if not np.any(usable):
        return mask
    threshold = np.nanpercentile(values[usable], 100.0 * (1.0 - float(fraction)))
    mask[usable] = values[usable] >= threshold
    return mask


def _repair_segment_continuum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    continuum: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    continuum = np.asarray(continuum, dtype=np.float64)
    bad = ~np.isfinite(continuum) | (continuum <= 0)
    if np.any(bad):
        continuum = continuum.copy()
        continuum[bad] = fallback[bad]
    continuum = _interpolate_finite(wavelength, continuum)
    flux_scale = np.nanmedian(flux[np.isfinite(flux) & (flux > 0)])
    if not np.isfinite(flux_scale) or flux_scale <= 0:
        return continuum
    continuum = np.where(np.isfinite(continuum) & (continuum > 0), continuum, flux_scale)
    return continuum


def _repair_continuum_values(
    wavelength: np.ndarray,
    flux: np.ndarray,
    continuum: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    continuum = np.asarray(continuum, dtype=np.float64).copy()
    for segment in _continuum_segments(
        wavelength,
        np.flatnonzero(valid),
        gap_factor=DEFAULT_CONTINUUM_GAP_FACTOR,
        min_gap_angstrom=DEFAULT_CONTINUUM_MIN_GAP_ANGSTROM,
    ):
        values = continuum[segment]
        if np.all(np.isfinite(values) & (values > 0)):
            continue
        fallback = _rolling_upper_envelope(
            wavelength[segment],
            flux[segment],
            valid[segment],
            window_angstrom=PSEUDO_CONTINUUM_ENVELOPE_WINDOW_AA,
            percentile=PSEUDO_CONTINUUM_ENVELOPE_PERCENTILE,
        )
        continuum[segment] = _repair_segment_continuum(wavelength[segment], flux[segment], values, fallback)
    return continuum


def _interpolate_finite(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) == 0:
        return y
    if np.count_nonzero(good) == 1:
        return np.full_like(y, y[good][0], dtype=np.float64)
    return np.interp(x, x[good], y[good], left=y[good][0], right=y[good][-1])


def _median_wavelength_step(wavelength: np.ndarray) -> float:
    values = np.asarray(wavelength, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    steps = np.diff(np.sort(values))
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if len(steps) == 0:
        return np.nan
    return float(np.nanmedian(steps))


def _spectrum_flux_err_values(spectrum: Any, n_values: int) -> np.ndarray | None:
    uncertainty = getattr(spectrum, "uncertainty", None)
    if uncertainty is None:
        return None
    values = _uncertainty_values(uncertainty, spectrum.flux.unit)
    values = np.asarray(values, dtype=np.float64)
    return values if len(values) == n_values else None


def _normalized_residual_uncertainty(
    spectrum: Any,
    continuum: np.ndarray,
    normalized_residual: np.ndarray,
) -> Any:
    from astropy import units as u
    from astropy.nddata import StdDevUncertainty

    fallback_sigma = _robust_sigma(normalized_residual)
    if not np.isfinite(fallback_sigma) or fallback_sigma <= 0:
        fallback_sigma = 1.0

    uncertainty = getattr(spectrum, "uncertainty", None)
    if uncertainty is None:
        return StdDevUncertainty(
            np.full(normalized_residual.shape, fallback_sigma, dtype=np.float64) * u.dimensionless_unscaled
        )

    values = _uncertainty_values(uncertainty, spectrum.flux.unit)
    values = np.asarray(values, dtype=np.float64)
    if len(values) != len(normalized_residual):
        return StdDevUncertainty(
            np.full(normalized_residual.shape, fallback_sigma, dtype=np.float64) * u.dimensionless_unscaled
        )
    valid = np.isfinite(values) & (values > 0) & np.isfinite(continuum) & (continuum != 0)
    normalized_sigma = np.full_like(normalized_residual, np.nan, dtype=np.float64)
    normalized_sigma[valid] = values[valid] / np.abs(continuum[valid])
    valid_sigma = np.isfinite(normalized_sigma) & (normalized_sigma > 0)
    if np.any(valid_sigma):
        fallback_sigma = float(np.nanmedian(normalized_sigma[valid_sigma]))
    fixed = np.where(valid_sigma, normalized_sigma, fallback_sigma)
    return StdDevUncertainty(fixed * u.dimensionless_unscaled)


def _continuum_segments(
    wavelength: np.ndarray,
    finite_indices: np.ndarray,
    *,
    gap_factor: float,
    min_gap_angstrom: float,
) -> list[np.ndarray]:
    if len(finite_indices) == 0:
        return []
    if len(finite_indices) == 1:
        return [finite_indices]

    dw = np.diff(wavelength[finite_indices])
    median_step = np.nanmedian(dw[np.isfinite(dw) & (dw > 0)])
    if not np.isfinite(median_step) or median_step <= 0:
        median_step = 0.0
    gap_threshold = max(float(gap_factor) * median_step, float(min_gap_angstrom))
    breaks = np.flatnonzero(dw > gap_threshold) + 1
    return [segment for segment in np.split(finite_indices, breaks) if len(segment)]


def _residual_uncertainty(spectrum: Any, residual_flux: Any) -> Any:
    from astropy.nddata import StdDevUncertainty

    fallback_sigma = _robust_sigma(np.asarray(residual_flux.value, dtype=np.float64))
    if not np.isfinite(fallback_sigma) or fallback_sigma <= 0:
        fallback_sigma = 1.0

    uncertainty = getattr(spectrum, "uncertainty", None)
    if uncertainty is None:
        return StdDevUncertainty(np.full(residual_flux.shape, fallback_sigma, dtype=np.float64) * spectrum.flux.unit)

    values = _uncertainty_values(uncertainty, spectrum.flux.unit)
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values) & (values > 0)
    if np.any(valid):
        fallback_sigma = float(np.nanmedian(values[valid]))
    fixed = np.where(valid, values, fallback_sigma)
    return StdDevUncertainty(fixed * spectrum.flux.unit)


def _find_significant_lines(spectrum: Any, *, noise_factor: float) -> Any:
    import warnings

    from astropy.table import QTable
    from specutils.fitting import find_lines_threshold

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Spectrum is not below the threshold.*")
            return find_lines_threshold(spectrum, noise_factor=float(noise_factor))
    except Exception:
        return QTable(names=("line_center", "line_type", "line_center_index"))


def _fit_significant_lines(
    spectrum: Any,
    lines: Any,
    *,
    line_window_AA: float,
    max_line_fits: int | None,
) -> list[SpectrumLineFit]:
    import warnings

    from astropy import units as u
    from astropy.modeling import models
    from astropy.utils.exceptions import AstropyWarning
    from specutils.fitting import fit_lines
    from specutils.spectra import SpectralRegion

    if lines is None or len(lines) == 0:
        return []

    flux = np.asarray(spectrum.flux.value, dtype=np.float64)
    uncertainty = _uncertainty_values(spectrum.uncertainty, spectrum.flux.unit)
    order = range(len(lines))
    if max_line_fits is not None and len(lines) > max_line_fits:
        strengths = []
        for idx, row in enumerate(lines):
            center_index = _line_center_index(row)
            strengths.append((idx, _line_significance(flux, uncertainty, center_index)))
        order = [idx for idx, _sig in sorted(strengths, key=lambda item: item[1], reverse=True)[:max_line_fits]]

    fitted_lines: list[SpectrumLineFit] = []
    half_width = max(float(line_window_AA), 0.1)
    mean_tolerance = min(1.0, half_width / 2.0)
    initial_stddev = max(min(half_width / 4.0, 0.8), 0.08)

    for row_index in order:
        row = lines[row_index]
        center = _line_center_quantity(row)
        center_index = _line_center_index(row)
        if center_index < 0 or center_index >= len(flux):
            continue
        amplitude = spectrum.flux[center_index]
        if not np.isfinite(amplitude.value) or amplitude.value == 0:
            continue

        gaussian = models.Gaussian1D(amplitude=amplitude, mean=center, stddev=initial_stddev * u.AA)
        center_value = center.to_value(u.AA)
        gaussian.mean.bounds = (center_value - mean_tolerance, center_value + mean_tolerance)
        gaussian.stddev.bounds = (0.03, half_width)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                fitted = fit_lines(
                    spectrum,
                    gaussian,
                    window=SpectralRegion(center - half_width * u.AA, center + half_width * u.AA),
                )
        except Exception:
            continue

        amplitude_value = _parameter_float(fitted.amplitude)
        mean_value = _parameter_float(fitted.mean)
        stddev_value = abs(_parameter_float(fitted.stddev))
        if not all(np.isfinite(v) for v in (amplitude_value, mean_value, stddev_value)):
            continue
        if stddev_value <= 0 or stddev_value > half_width:
            continue

        fitted_lines.append(SpectrumLineFit(
            line_type=str(row["line_type"]),
            center=float(center_value),
            center_index=center_index,
            significance=_line_significance(flux, uncertainty, center_index),
            amplitude=amplitude_value,
            mean=mean_value,
            stddev=stddev_value,
            model=fitted,
        ))

    return sorted(fitted_lines, key=lambda line_fit: line_fit.center)


def _line_center_quantity(row: Any) -> Any:
    from astropy import units as u

    center = row["line_center"]
    if hasattr(center, "to"):
        return center.to(u.AA)
    return float(center) * u.AA


def _line_center_index(row: Any) -> int:
    try:
        return int(row["line_center_index"])
    except Exception:
        return -1


def _line_significance(flux: np.ndarray, uncertainty: np.ndarray, index: int) -> float:
    if index < 0 or index >= len(flux) or index >= len(uncertainty):
        return np.nan
    sigma = uncertainty[index]
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan
    return float(abs(flux[index]) / sigma)


def _parameter_float(parameter: Any) -> float:
    value = getattr(parameter, "value", parameter)
    return float(np.asarray(value, dtype=np.float64))


def _uncertainty_values(uncertainty: Any, flux_unit: Any) -> np.ndarray:
    quantity = getattr(uncertainty, "quantity", None)
    if quantity is None and getattr(uncertainty, "array", None) is not None:
        quantity = uncertainty.array * flux_unit
    if quantity is not None:
        return np.asarray(quantity.to_value(flux_unit), dtype=np.float64)
    return np.asarray([], dtype=np.float64)


def _strongest_line_fits(line_fits: list[SpectrumLineFit], *, max_draw: int) -> list[SpectrumLineFit]:
    if len(line_fits) <= max_draw:
        return line_fits
    strongest = sorted(line_fits, key=lambda line_fit: line_fit.significance, reverse=True)[:max_draw]
    return sorted(strongest, key=lambda line_fit: line_fit.center)


def _labeled_line_fits(
    line_fits: list[SpectrumLineFit],
    labels: Mapping[float, str] | None,
    *,
    max_labels: int,
    min_spacing_angstrom: float,
) -> list[tuple[SpectrumLineFit, str]]:
    if not labels or max_labels <= 0:
        return []

    label_centers = np.asarray(list(labels.keys()), dtype=np.float64)
    label_values = [str(value).strip() for value in labels.values()]
    if len(label_centers) == 0:
        return []

    selected: list[tuple[SpectrumLineFit, str]] = []
    selected_centers: list[float] = []
    strongest = sorted(line_fits, key=lambda line_fit: line_fit.significance, reverse=True)
    for line_fit in strongest:
        if len(selected) >= max_labels:
            break
        if selected_centers and min(abs(line_fit.center - center) for center in selected_centers) < min_spacing_angstrom:
            continue
        nearest = int(np.nanargmin(np.abs(label_centers - line_fit.center)))
        if abs(label_centers[nearest] - line_fit.center) > 0.1:
            continue
        label = label_values[nearest]
        if not label:
            continue
        selected.append((line_fit, label))
        selected_centers.append(line_fit.center)
    return sorted(selected, key=lambda item: item[0].center)


def _draw_interpanel_line_labels(
    fig: Any,
    upper_ax: Any,
    lower_ax: Any,
    labeled_lines: list[tuple[SpectrumLineFit, str]],
    *,
    color: str,
    fontsize: float,
) -> None:
    if not labeled_lines:
        return

    from matplotlib.transforms import blended_transform_factory

    upper_box = upper_ax.get_position()
    lower_box = lower_ax.get_position()
    gap_bottom = lower_box.y1
    gap_top = upper_box.y0
    if gap_top <= gap_bottom:
        return

    gap_height = gap_top - gap_bottom
    label_y = gap_bottom + 0.50 * gap_height
    transform = blended_transform_factory(lower_ax.transData, fig.transFigure)
    for line_fit, label in labeled_lines:
        lower_ax.plot(
            [line_fit.center, line_fit.center],
            [gap_bottom, gap_top],
            transform=transform,
            color=color,
            lw=0.54,
            alpha=0.32,
            solid_capstyle="butt",
            clip_on=False,
            zorder=5,
        )
        lower_ax.text(
            line_fit.center,
            label_y,
            label,
            transform=transform,
            ha="center",
            va="center",
            rotation=90,
            rotation_mode="anchor",
            fontsize=fontsize,
            color=color,
            alpha=0.95,
            clip_on=False,
            zorder=6,
        )


def _plot_segments(
    ax: Any,
    wavelength: np.ndarray,
    values: np.ndarray,
    *,
    finite: np.ndarray | None,
    max_gap_angstrom: float,
    **plot_kwargs: Any,
) -> None:
    for segment in _plot_segment_indices(
        wavelength,
        values,
        finite=finite,
        max_gap_angstrom=max_gap_angstrom,
    ):
        if len(segment) < 2:
            continue
        ax.plot(wavelength[segment], values[segment], **plot_kwargs)


def _fill_between_segments(
    ax: Any,
    wavelength: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    finite: np.ndarray | None,
    max_gap_angstrom: float,
    **fill_kwargs: Any,
) -> None:
    for segment in _plot_segment_indices(
        wavelength,
        lower,
        upper,
        finite=finite,
        max_gap_angstrom=max_gap_angstrom,
    ):
        if len(segment) < 2:
            continue
        ax.fill_between(wavelength[segment], lower[segment], upper[segment], **fill_kwargs)


def _plot_segment_indices(
    wavelength: np.ndarray,
    *values: np.ndarray,
    finite: np.ndarray | None,
    max_gap_angstrom: float,
) -> list[np.ndarray]:
    wavelength_arr = np.asarray(wavelength, dtype=np.float64)
    valid = np.isfinite(wavelength_arr)
    if finite is not None and len(finite) == len(wavelength_arr):
        valid &= np.asarray(finite, dtype=bool)
    for value in values:
        value_arr = np.asarray(value, dtype=np.float64)
        if len(value_arr) != len(wavelength_arr):
            return []
        valid &= np.isfinite(value_arr)

    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return []
    if len(indices) == 1:
        return [indices]

    dw = np.diff(wavelength_arr[indices])
    index_breaks = np.diff(indices) > 1
    wavelength_breaks = np.isfinite(dw) & (dw > max_gap_angstrom)
    breaks = np.flatnonzero(index_breaks | wavelength_breaks) + 1
    return [segment for segment in np.split(indices, breaks) if len(segment)]


def _plot_gap_threshold(
    wavelength: np.ndarray,
    finite: np.ndarray,
    *,
    gap_factor: float,
    min_gap_angstrom: float,
) -> float:
    wavelength_arr = np.asarray(wavelength, dtype=np.float64)
    valid = np.isfinite(wavelength_arr) & np.asarray(finite, dtype=bool)
    if np.count_nonzero(valid) < 2:
        return float(min_gap_angstrom)

    dw = np.diff(wavelength_arr[valid])
    positive = dw[np.isfinite(dw) & (dw > 0)]
    if len(positive) == 0:
        return float(min_gap_angstrom)

    median_step = float(np.nanmedian(positive))
    if not np.isfinite(median_step) or median_step <= 0:
        return float(min_gap_angstrom)
    return max(float(gap_factor) * median_step, float(min_gap_angstrom))


def _uncertainty_bounds(
    wavelength: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None,
    *,
    finite: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if flux_err is None or len(flux_err) != len(wavelength):
        return None

    wavelength_arr = np.asarray(wavelength, dtype=np.float64)
    flux_arr = np.asarray(flux, dtype=np.float64)
    err_arr = np.asarray(flux_err, dtype=np.float64)
    if len(flux_arr) != len(wavelength_arr):
        return None

    valid = np.isfinite(wavelength_arr) & np.isfinite(flux_arr) & np.isfinite(err_arr)
    if finite is not None and len(finite) == len(wavelength_arr):
        valid &= np.asarray(finite, dtype=bool)
    if not np.any(valid):
        return None

    err_arr = np.abs(err_arr)
    lower = np.full_like(flux_arr, np.nan, dtype=np.float64)
    upper = np.full_like(flux_arr, np.nan, dtype=np.float64)
    lower[valid] = flux_arr[valid] - err_arr[valid]
    upper[valid] = flux_arr[valid] + err_arr[valid]
    return lower, upper


def _robust_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.nan
    median = np.nanmedian(finite)
    mad = np.nanmedian(np.abs(finite - median))
    if np.isfinite(mad) and mad > 0:
        return float(1.4826 * mad)
    return float(np.nanstd(finite))


def _rolling_continuum(flux: np.ndarray, finite: np.ndarray) -> np.ndarray:
    import pandas as pd

    continuum = np.full_like(flux, np.nan, dtype=np.float64)
    idx = np.flatnonzero(finite)
    if len(idx) == 0:
        return continuum

    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    for segment in np.split(idx, breaks):
        if len(segment) < 20:
            continue
        window = min(401, len(segment) if len(segment) % 2 == 1 else len(segment) - 1)
        window = max(21, window)
        median = pd.Series(flux[segment]).rolling(
            window=window,
            center=True,
            min_periods=max(10, window // 5),
        ).median()
        continuum[segment] = median.interpolate(limit_direction="both").to_numpy()
    return continuum


def _robust_limits(values: np.ndarray, lo: float, hi: float, *, min_pad: float) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0, 1.0

    y0, y1 = np.nanpercentile(finite, [lo, hi])
    if not np.isfinite(y0) or not np.isfinite(y1) or y0 == y1:
        center = float(np.nanmedian(finite)) if np.isfinite(np.nanmedian(finite)) else 0.0
        spread = abs(center) * 0.05 or min_pad
        return center - spread, center + spread
    pad = max((y1 - y0) * 0.08, min_pad)
    return y0 - pad, y1 + pad


def _scale_pad(values: np.ndarray, *, fraction: float, minimum: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return minimum
    scale = float(np.nanpercentile(np.abs(finite), 90))
    return max(scale * fraction, minimum)


def _theme_tokens(theme: str) -> dict[str, str]:
    mode = str(theme or "dark").strip().lower()
    if mode == "white":
        return {
            "paper_bg": "#ffffff",
            "plot_bg": "#ffffff",
            "font": "#1c2733",
            "grid": "rgba(104, 128, 149, 0.18)",
            "line_color": "#2563eb",
            "error_fill": "rgba(37, 99, 235, 0.12)",
            "legend_bg": "rgba(255, 255, 255, 0.92)",
            "legend_border": "rgba(120, 140, 158, 0.35)",
        }
    if mode == "gray":
        return {
            "paper_bg": "#2e3440",
            "plot_bg": "#2e3440",
            "font": "#d8dee9",
            "grid": "rgba(216, 222, 233, 0.12)",
            "line_color": "#88c0d0",
            "error_fill": "rgba(136, 192, 208, 0.15)",
            "legend_bg": "rgba(46, 52, 64, 0.92)",
            "legend_border": "rgba(216, 222, 233, 0.2)",
        }
    return {
        "paper_bg": "#0a1628",
        "plot_bg": "#0a1628",
        "font": "#c8d8e6",
        "grid": "rgba(104, 128, 149, 0.18)",
        "line_color": "#5eead4",
        "error_fill": "rgba(94, 234, 212, 0.12)",
        "legend_bg": "rgba(8, 16, 24, 0.92)",
        "legend_border": "rgba(120, 140, 158, 0.25)",
    }
