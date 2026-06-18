"""Native Matplotlib publication exports for review light curves."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from malca.config import REVIEW_RESIDUAL_FRACTION
from malca.review.lightcurve_pdf import _axis_label_for_offset


def build_review_lightcurve_publication_pdf(
    payload: dict,
    *,
    plot_dir: str | Path | None,
    selected_cameras: list[str] | None,
    selected_bands: list[str] | None,
    filter_bad_cameras: bool,
    show_baseline: bool,
    show_event_markers: bool,
    show_residuals: bool,
    show_phase_fold: bool,
    show_raw_mag: bool,
    phase_panel_mode: Literal["fold", "time"] = "fold",
    override_period: float | None,
    override_period_source: str = "manual/search",
    phase_period_pending: bool = False,
    suppress_catalog_phase_period: bool = False,
    show_diagnostics: bool,
    confidence_colors: bool,
    run_params: dict | None,
    residual_fraction: float = REVIEW_RESIDUAL_FRACTION,
    baseline_opacity: float = 0.55,
    yaxis_mode: Literal["mag", "flux"] = "mag",
    external_lcs: dict[str, Path] | None = None,
    external_source_view: str | list[str] = "asassn",
    external_panel_mode: Literal["overlay", "split"] = "overlay",
    candidate_id: str | None = None,
    native_color_mode: Literal["camera", "band"] = "camera",
) -> bytes:
    """Render the review light-curve view as a publication PDF via the unified assembler."""
    from malca.review.lightcurve_assembly import ReviewPlotRequest, assemble_review_lightcurve_plot
    from malca.review.lightcurve_pdf import render_review_lightcurve_pdf

    request = ReviewPlotRequest.from_kwargs(
        payload,
        plot_dir=Path(plot_dir) if plot_dir else None,
        selected_cameras=selected_cameras,
        filter_bad_cameras=filter_bad_cameras,
        show_baseline=show_baseline,
        show_event_markers=show_event_markers,
        show_residuals=show_residuals,
        show_phase_fold=show_phase_fold,
        phase_panel_mode=phase_panel_mode,
        show_raw_mag=show_raw_mag,
        override_period=override_period,
        override_period_source=override_period_source,
        phase_period_pending=phase_period_pending,
        suppress_catalog_phase_period=suppress_catalog_phase_period,
        show_diagnostics=show_diagnostics,
        confidence_colors=confidence_colors,
        run_params=run_params,
        residual_fraction=residual_fraction,
        baseline_opacity=baseline_opacity,
        yaxis_mode=yaxis_mode,
        external_lcs=external_lcs,
        external_source_view=external_source_view,
        external_panel_mode=external_panel_mode,
        selected_bands=selected_bands,
        native_color_mode=native_color_mode,
        candidate_id=candidate_id,
        discover_external=True,
    )
    spec = assemble_review_lightcurve_plot(request)
    return render_review_lightcurve_pdf(spec)
