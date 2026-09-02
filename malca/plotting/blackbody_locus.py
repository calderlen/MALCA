"""Bandpass-integrated blackbody loci for infrared color-color diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D, Bbox

from malca.enrichment.synthetic_photometry import (
    BandpassUnavailableError,
    bandpass_flux_nu_jy,
    fetch_filter_response,
    load_cached_filter_response,
)
from malca.plotting.irac import IRAC_VEGA_ZERO_POINT_JY
from malca.review.sed import bandpass_for


REFERENCE_MARKER_TEMPERATURES_K = (280.0, 400.0, 1000.0, 5000.0)
# Include the supplied paper's Figure 4 fiducials, the requested 3000 K point,
# and round-number temperatures spanning cooler dust through hot emitters.
# Display-space thinning below prevents these candidates from crowding a plot.
DEFAULT_MARKER_TEMPERATURES_K = (
    50.0,
    100.0,
    200.0,
    280.0,
    400.0,
    600.0,
    1000.0,
    2000.0,
    3000.0,
    5000.0,
    10_000.0,
    20_000.0,
    50_000.0,
    100_000.0,
)
_PRIMARY_MARKER_TEMPERATURES_K = (280.0, 400.0, 1000.0, 3000.0, 5000.0)
DEFAULT_TEMPERATURES_K = tuple(
    np.unique(
        np.concatenate(
            [
                # The cool end drives mid-infrared colors beyond the visible
                # axes; the hot end converges to the physical Rayleigh-Jeans
                # color instead of stopping arbitrarily at 10,000 K.
                np.geomspace(20.0, 1.0e8, 640),
                np.asarray(DEFAULT_MARKER_TEMPERATURES_K, dtype=float),
            ]
        )
    )
)

# hc/k in Angstrom K.  Only the spectral shape is required because the
# arbitrary blackbody normalization cancels from every color.
_HC_OVER_K_ANGSTROM_K = 1.438776877e8
_TEMPERATURE_LABEL_OFFSET_POINTS = 18.0
_TEMPERATURE_LABEL_ZORDER = 1_000_000.0


@dataclass(frozen=True)
class _BandDefinition:
    source: str
    band: str
    zero_point_jy: float | None = None


_BANDS: dict[str, _BandDefinition] = {
    "J": _BandDefinition("2MASS", "J"),
    "H": _BandDefinition("2MASS", "H"),
    "Ks": _BandDefinition("2MASS", "Ks"),
    "W1": _BandDefinition("AllWISE", "W1"),
    "W2": _BandDefinition("AllWISE", "W2"),
    "W3": _BandDefinition("AllWISE", "W3"),
    "W4": _BandDefinition("AllWISE", "W4"),
    # SEIP stores IRAC measurements as Jy.  Colors in the plotting scripts use
    # the standard IRAC Vega zero-magnitude flux densities.
    "IRAC1": _BandDefinition("Spitzer SEIP", "IRAC1", IRAC_VEGA_ZERO_POINT_JY["IRAC1"]),
    "IRAC2": _BandDefinition("Spitzer SEIP", "IRAC2", IRAC_VEGA_ZERO_POINT_JY["IRAC2"]),
    "IRAC3": _BandDefinition("Spitzer SEIP", "IRAC3", IRAC_VEGA_ZERO_POINT_JY["IRAC3"]),
    "IRAC4": _BandDefinition("Spitzer SEIP", "IRAC4", IRAC_VEGA_ZERO_POINT_JY["IRAC4"]),
}


@dataclass(frozen=True)
class BlackbodyColorLocus:
    """One unreddened, single-temperature blackbody color-color track."""

    temperature_k: np.ndarray
    x: np.ndarray
    y: np.ndarray


def _visible_endpoint_indices(
    locus: BlackbodyColorLocus,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[int, int] | None:
    """Return the first and last finite locus samples inside the axes limits."""
    finite = np.isfinite(locus.x) & np.isfinite(locus.y)
    visible = (
        finite
        & (locus.x >= min(xlim))
        & (locus.x <= max(xlim))
        & (locus.y >= min(ylim))
        & (locus.y <= max(ylim))
    )
    indices = np.flatnonzero(visible)
    if indices.size < 2:
        indices = np.flatnonzero(finite)
    if indices.size < 2:
        return None
    return int(indices[0]), int(indices[-1])


def _display_rotation(
    ax: plt.Axes,
    locus: BlackbodyColorLocus,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> float:
    """Return one readable display-space angle for the entire visible locus."""
    endpoints_indices = _visible_endpoint_indices(locus, xlim, ylim)
    if endpoints_indices is None:
        return 0.0
    lower, upper = endpoints_indices
    endpoints = ax.transData.transform(
        [
            (float(locus.x[lower]), float(locus.y[lower])),
            (float(locus.x[upper]), float(locus.y[upper])),
        ]
    )
    delta_x, delta_y = endpoints[1] - endpoints[0]
    rotation = float(np.degrees(np.arctan2(delta_y, delta_x)))
    if rotation > 90.0:
        rotation -= 180.0
    elif rotation < -90.0:
        rotation += 180.0
    return rotation


def _frame_line_intersections(
    first: tuple[float, float],
    last: tuple[float, float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> list[tuple[float, float, float]]:
    """Return ``(t, x, y)`` intersections of a line with the axes rectangle."""
    x0, y0 = first
    dx = last[0] - x0
    dy = last[1] - y0
    xmin, xmax = sorted(xlim)
    ymin, ymax = sorted(ylim)
    tolerance = 1.0e-10
    candidates: list[tuple[float, float, float]] = []

    if abs(dx) > tolerance:
        for x in (xmin, xmax):
            t = (x - x0) / dx
            y = y0 + t * dy
            if ymin - tolerance <= y <= ymax + tolerance:
                candidates.append((float(t), float(x), float(np.clip(y, ymin, ymax))))
    if abs(dy) > tolerance:
        for y in (ymin, ymax):
            t = (y - y0) / dy
            x = x0 + t * dx
            if xmin - tolerance <= x <= xmax + tolerance:
                candidates.append((float(t), float(np.clip(x, xmin, xmax)), float(y)))

    unique: list[tuple[float, float, float]] = []
    for candidate in sorted(candidates, key=lambda item: item[0]):
        if not unique or not np.allclose(candidate[1:], unique[-1][1:], atol=tolerance):
            unique.append(candidate)
    return unique


def _extend_locus_to_frame(
    ax: plt.Axes,
    locus: BlackbodyColorLocus,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    color: str,
    linewidth: float,
    zorder: float,
) -> None:
    """Extend both visible ends along the common locus slope to the frame."""
    endpoints_indices = _visible_endpoint_indices(locus, xlim, ylim)
    if endpoints_indices is None:
        return
    lower, upper = endpoints_indices
    first = (float(locus.x[lower]), float(locus.y[lower]))
    last = (float(locus.x[upper]), float(locus.y[upper]))
    intersections = _frame_line_intersections(first, last, xlim, ylim)
    if len(intersections) < 2:
        return

    before, after = intersections[0], intersections[-1]
    if before[0] < 0.0:
        ax.plot(
            [before[1], first[0]],
            [before[2], first[1]],
            color=color,
            linewidth=linewidth,
            zorder=zorder,
            scalex=False,
            scaley=False,
        )
    if after[0] > 1.0:
        ax.plot(
            [last[0], after[1]],
            [last[1], after[2]],
            color=color,
            linewidth=linewidth,
            zorder=zorder,
            scalex=False,
            scaley=False,
        )


def _fixed_above_offset(rotation: float) -> tuple[float, float]:
    """Return one constant display-space offset normal to the local locus."""
    angle = np.radians(rotation)
    return (
        float(-_TEMPERATURE_LABEL_OFFSET_POINTS * np.sin(angle)),
        float(_TEMPERATURE_LABEL_OFFSET_POINTS * np.cos(angle)),
    )


def _padded_bbox(bbox: Bbox, padding_px: float) -> Bbox:
    return Bbox.from_extents(
        bbox.x0 - padding_px,
        bbox.y0 - padding_px,
        bbox.x1 + padding_px,
        bbox.y1 + padding_px,
    )


def _outside_distance(bbox: Bbox, container_bbox: Bbox) -> float:
    return float(
        max(container_bbox.x0 - bbox.x0, 0.0)
        + max(bbox.x1 - container_bbox.x1, 0.0)
        + max(container_bbox.y0 - bbox.y0, 0.0)
        + max(bbox.y1 - container_bbox.y1, 0.0)
    )


def _planck_lambda_shape(wavelength_angstrom: np.ndarray, temperature_k: float) -> np.ndarray:
    """Return a numerically stable Planck ``B_lambda`` shape in arbitrary units."""
    wavelength = np.asarray(wavelength_angstrom, dtype=float)
    exponent = _HC_OVER_K_ANGSTROM_K / (wavelength * float(temperature_k))
    log_expm1 = np.where(exponent > 50.0, exponent, np.log(np.expm1(exponent)))
    log_flux = -5.0 * np.log(wavelength) - log_expm1
    # A single fixed scale factor keeps the values well away from underflow.
    # It must not be normalized independently in each band: the relative
    # normalization between passbands is precisely what defines a color.
    return np.exp(log_flux + 100.0)


def _resolve_response(
    band: str,
    *,
    cache_dir: str | None,
    allow_download: bool,
):
    definition = _BANDS.get(band)
    if definition is None:
        raise KeyError(f"No blackbody-locus band registration for {band!r}.")
    registration = bandpass_for(definition.source, definition.band)
    if registration is None or not registration.svo_filter_id:
        raise BandpassUnavailableError(f"No SVO response is registered for {band}.")
    response = load_cached_filter_response(
        registration.svo_filter_id,
        registration.mag_system,
        cache_dir,
    )
    if response is None and allow_download:
        response = fetch_filter_response(
            registration.svo_filter_id,
            registration.mag_system,
            cache_dir=cache_dir,
        )
    if response is None:
        raise BandpassUnavailableError(
            f"No cached response for {band} ({registration.svo_filter_id})."
        )
    zero_point = definition.zero_point_jy or response.zero_point_jy
    if zero_point is None or not np.isfinite(zero_point) or zero_point <= 0.0:
        raise BandpassUnavailableError(f"No positive Vega zero point is available for {band}.")
    return response, float(zero_point)


@lru_cache(maxsize=64)
def _band_magnitudes_cached(
    band: str,
    temperatures_k: tuple[float, ...],
    cache_dir: str | None,
    allow_download: bool,
) -> tuple[float, ...]:
    response, zero_point_jy = _resolve_response(
        band,
        cache_dir=cache_dir,
        allow_download=allow_download,
    )
    magnitudes = []
    for temperature in temperatures_k:
        spectrum = _planck_lambda_shape(response.wavelength_angstrom, temperature)
        flux_nu_jy = bandpass_flux_nu_jy(
            response.wavelength_angstrom,
            spectrum,
            response,
        )
        magnitudes.append(float(-2.5 * np.log10(flux_nu_jy / zero_point_jy)))
    return tuple(magnitudes)


def blackbody_color_color_locus(
    x_bands: tuple[str, str],
    y_bands: tuple[str, str],
    *,
    temperatures_k: Iterable[float] = DEFAULT_TEMPERATURES_K,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
) -> BlackbodyColorLocus:
    """Calculate two Vega colors for a grid of pure blackbody temperatures."""
    temperatures = np.asarray(tuple(temperatures_k), dtype=float)
    if temperatures.ndim != 1 or temperatures.size < 2:
        raise ValueError("At least two blackbody temperatures are required.")
    if not np.all(np.isfinite(temperatures) & (temperatures > 0.0)):
        raise ValueError("Blackbody temperatures must be finite and positive.")
    order = np.argsort(temperatures)
    temperatures = temperatures[order]
    temperature_key = tuple(float(value) for value in temperatures)
    cache_key = str(Path(cache_dir).expanduser()) if cache_dir is not None else None

    required = set(x_bands + y_bands)
    magnitudes = {
        band: np.asarray(
            _band_magnitudes_cached(band, temperature_key, cache_key, allow_download),
            dtype=float,
        )
        for band in required
    }
    return BlackbodyColorLocus(
        temperature_k=temperatures,
        x=magnitudes[x_bands[0]] - magnitudes[x_bands[1]],
        y=magnitudes[y_bands[0]] - magnitudes[y_bands[1]],
    )


def add_blackbody_locus(
    ax: plt.Axes,
    x_bands: tuple[str, str],
    y_bands: tuple[str, str],
    *,
    marker_temperatures_k: Iterable[float] = DEFAULT_MARKER_TEMPERATURES_K,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
    color: str = "#b2182b",
    label: str = r"Blackbody $T_{\rm BB}$",
    label_placement: Literal["alternating", "above", "below"] = "above",
    label_fontsize: float = 10.5,
) -> BlackbodyColorLocus:
    """Draw a blackbody track without allowing it to change the data limits."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.figure.canvas.draw()
    axes_bbox = ax.get_window_extent(ax.figure.canvas.get_renderer())
    locus = blackbody_color_color_locus(
        x_bands,
        y_bands,
        cache_dir=cache_dir,
        allow_download=allow_download,
    )
    rotation = _display_rotation(ax, locus, xlim, ylim)
    locus_linewidth = 2.9
    locus_zorder = 0.5
    ax.plot(
        locus.x,
        locus.y,
        color=color,
        linewidth=locus_linewidth,
        zorder=locus_zorder,
        label=label,
    )
    # Plotting the full physical locus can temporarily autoscale far beyond the
    # requested color-color view.  Restore the caller's limits before measuring
    # annotations so valid in-panel labels are not falsely rejected at an edge.
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _extend_locus_to_frame(
        ax,
        locus,
        xlim,
        ylim,
        color=color,
        linewidth=locus_linewidth,
        zorder=locus_zorder,
    )

    marker_temperatures = tuple(float(value) for value in marker_temperatures_k)
    candidates = []
    for index, temperature in enumerate(marker_temperatures):
        nearest = int(np.argmin(np.abs(locus.temperature_k - temperature)))
        x = float(locus.x[nearest])
        y = float(locus.y[nearest])
        if not (
            np.isfinite(x)
            and np.isfinite(y)
            and xlim[0] <= x <= xlim[1]
            and ylim[0] <= y <= ylim[1]
        ):
            continue
        display_x, display_y = ax.transData.transform((x, y))
        candidates.append(
            (
                index,
                temperature,
                x,
                y,
                nearest,
                float(display_x),
                float(display_y),
            )
        )

    # Thin candidates in display rather than data coordinates so the result is
    # stable across very different color ranges and aspect ratios.  Established
    # paper fiducials retain priority over supplemental round-number markers.
    priority_order = {
        temperature: index
        for index, temperature in enumerate(_PRIMARY_MARKER_TEMPERATURES_K)
    }
    minimum_marker_spacing_px = 42.0 * float(ax.figure.dpi) / 72.0
    selected = []

    def _add_if_separated(candidate) -> bool:
        *_, display_x, display_y = candidate
        if any(
            np.hypot(display_x - other[-2], display_y - other[-1])
            < minimum_marker_spacing_px
            for other in selected
        ):
            return False
        selected.append(candidate)
        return True

    primary_candidates = sorted(
        (item for item in candidates if item[1] in priority_order),
        key=lambda item: priority_order[item[1]],
    )
    for candidate in primary_candidates:
        selected.append(candidate)

    supplemental_candidates = [item for item in candidates if item[1] not in priority_order]
    if primary_candidates:
        visible_primary_temperatures = [item[1] for item in primary_candidates]
        lower = sorted(
            (
                item
                for item in supplemental_candidates
                if item[1] < min(visible_primary_temperatures)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        higher = sorted(
            (
                item
                for item in supplemental_candidates
                if item[1] > max(visible_primary_temperatures)
            ),
            key=lambda item: item[1],
        )
        supplemental_order = []
        while lower or higher:
            if lower:
                supplemental_order.append(lower.pop(0))
            if higher:
                supplemental_order.append(higher.pop(0))
    else:
        supplemental_order = sorted(supplemental_candidates, key=lambda item: item[1])

    supplemental_added = 0
    for candidate in supplemental_order:
        if _add_if_separated(candidate):
            supplemental_added += 1
        if supplemental_added >= 2:
            break

    placed_labels: list[Bbox] = []
    for draw_index, (_, temperature, x, y, nearest, _, _) in enumerate(
        sorted(
            selected,
            key=lambda item: (
                0 if item[1] in priority_order else 1,
                item[1],
            ),
        )
    ):
        marker = MarkerStyle("s").transformed(Affine2D().rotate_deg(rotation))
        if label_placement in {"above", "below"}:
            horizontal_alignment = "center"
            vertical_alignment = "center"
        elif label_placement == "alternating":
            angle_radians = np.radians(rotation)
            normal_distance = 15.0 if draw_index % 2 == 0 else -15.0
            offset = (
                float(-normal_distance * np.sin(angle_radians)),
                float(normal_distance * np.cos(angle_radians)),
            )
            horizontal_alignment = "left"
            vertical_alignment = "bottom" if offset[1] > 0 else "top"
        else:
            raise ValueError(f"Unknown blackbody label placement: {label_placement}")
        annotation = ax.annotate(
            f"{temperature:g} K",
            xy=(x, y),
            xytext=(0.0, 0.0) if label_placement in {"above", "below"} else offset,
            textcoords="offset points",
            ha=horizontal_alignment,
            va=vertical_alignment,
            fontsize=label_fontsize,
            fontweight=900,
            color=color,
            path_effects=[
                path_effects.withStroke(linewidth=0.7, foreground=color),
            ],
            rotation=rotation,
            rotation_mode="anchor",
            annotation_clip=False,
            clip_on=False,
            zorder=_TEMPERATURE_LABEL_ZORDER,
        )
        annotation.set_in_layout(False)
        label_is_visible = True
        if label_placement in {"above", "below"}:
            ax.figure.canvas.draw()
            offset = _fixed_above_offset(rotation)
            if label_placement == "below":
                offset = (-offset[0], -offset[1])
            annotation.set_position(offset)
            ax.figure.canvas.draw()
            actual_bbox = annotation.get_window_extent(ax.figure.canvas.get_renderer())
            padded_actual = _padded_bbox(actual_bbox, 8.0)
            overlaps_label = any(
                padded_actual.overlaps(_padded_bbox(other, 8.0))
                for other in placed_labels
            )
            if _outside_distance(padded_actual, axes_bbox) > 0.0 or overlaps_label:
                # Omit the complete label-marker pair when the label cannot be
                # placed cleanly inside the axes.
                annotation.remove()
                label_is_visible = False
            else:
                placed_labels.append(actual_bbox)
        if label_is_visible:
            ax.scatter(
                [x],
                [y],
                marker=marker,
                s=36,
                facecolor=color,
                edgecolor=color,
                linewidth=1.0,
                zorder=999,
            )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return locus
