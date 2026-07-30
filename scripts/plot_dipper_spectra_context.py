#!/usr/bin/env python
"""Build publication-quality spectrum/context PDFs for reviewed dippers.

Each PDF page is self-contained and contains:

1. Median-centered ASAS-SN V/g light curves.
2. The current Review SED photometry and current stored atmosphere model.
3. One archival spectrum, its pseudo-continuum, and its normalized residual.

The default cohort is the run's paper-candidate table.  Candidates represented
by more than one spectral survey receive one page per survey in the same PDF.
The script reads the Review database in SQLite read-only mode, reuses existing
spectrum caches, and only downloads missing spectra when ``--cache-only`` is
not supplied.

Example
-------
conda run -n malca python scripts/plot_dipper_spectra_context.py \
  --run-dir output/runs/dat3-full-extended_2026-07-01-v4
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from malca.enrich.spectrum_fetch import (
    FetchStatus,
    SpectrumData,
    fetch_spectrum,
    load_spectrum_cache,
)
from malca.plotting.lightcurve_publication import (
    PUBLICATION_STYLE,
    filter_lightcurve,
    load_lightcurve,
)
from malca.review.sed import load_sed_rows, render_sed_matplotlib
from malca.enrichment.sed_model import (
    load_sed_model_curves,
    load_sed_model_fits,
    load_sed_model_points,
)
from malca.review.spectrum_plot import analyze_spectrum


DEFAULT_RUN_DIR = Path("output/runs/dat3-full-extended_2026-07-01-v4")
DEFAULT_OUTPUT_DIR = Path("output/pdf/dipper_spectra_context_publication")
DEFAULT_PAPER_TABLE_RELATIVE = Path("results/paper_candidate_tables/dipper_candidates.csv")
DEFAULT_SPECTRA_LONG_RELATIVE = Path("results/spectra_enrichment/spectra_long.parquet")
DEFAULT_SPECTRA_SUMMARY_RELATIVE = Path("results/spectra_enrichment/spectra_summary.parquet")
DEFAULT_SPECTRUM_CACHE_RELATIVE = Path("results/spectra_enrichment/spectra")

PAGE_WIDTH_IN = 7.25
PAGE_HEIGHT_IN = 9.70
SOLAR_LUMINOSITY_ERG_S = 3.828e33

SOURCE_LABELS = {
    "apogee_dr16": "APOGEE DR16",
    "apogee_dr17": "APOGEE DR17",
    "desi_dr1": "DESI DR1",
    "galah_dr3": "GALAH DR3",
    "galah_dr4": "GALAH DR4",
    "lamost_dr7": "LAMOST DR7",
    "rave_dr6": "RAVE DR6",
    "sdss2_sn": "SDSS DR16",
    "sdss_dr16_spec": "SDSS DR16",
    "sdss_boss": "SDSS/BOSS",
}

LIGHTCURVE_STYLES = {
    "V": {"color": "#6a51a3", "marker": "^"},
    "g": {"color": "#238b45", "marker": "o"},
}

CONTEXT_STYLE = {
    **PUBLICATION_STYLE,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "legend.fontsize": 6.8,
    "lines.linewidth": 0.85,
}


@dataclass
class CandidateContext:
    """Run-local data needed to draw one candidate's context panels."""

    run_dir: Path
    candidate_rows: pd.DataFrame
    sed_rows: dict[str, pd.DataFrame]
    sed_curves: dict[str, pd.DataFrame]
    sed_fits: dict[str, pd.DataFrame]
    sed_points: dict[str, pd.DataFrame]

    def candidate_row(self, candidate_id: str) -> pd.Series:
        row = self.candidate_rows.loc[str(candidate_id)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    def payload(self, candidate_id: str) -> dict[str, object]:
        row = self.candidate_row(candidate_id)
        payload: dict[str, object] = {}
        raw = row.get("payload_json")
        if isinstance(raw, str) and raw.strip():
            try:
                decoded = json.loads(raw)
                if isinstance(decoded, dict):
                    payload.update(decoded)
            except json.JSONDecodeError:
                pass
        for key, value in row.items():
            cleaned = _clean_value(value)
            if cleaned is not None:
                payload[str(key)] = cleaned
        payload["candidate_id"] = str(candidate_id)
        return payload


def _clean_value(value: object) -> object | None:
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return None
    except Exception:
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _safe_filename(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _latex_text(value: object) -> str:
    """Escape plain metadata for Matplotlib's LaTeX text renderer."""

    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (str(name),),
    ).fetchone()
    return row is not None


def _query_candidate_ids(
    conn: sqlite3.Connection,
    *,
    cohort: str,
    paper_table: Path,
    explicit_ids: Iterable[str] | None,
) -> list[str]:
    explicit = [str(value).strip() for value in (explicit_ids or []) if str(value).strip()]
    if explicit:
        return list(dict.fromkeys(explicit))

    if cohort == "paper":
        if not paper_table.exists():
            raise FileNotFoundError(
                f"Paper cohort requested but candidate table is missing: {paper_table}"
            )
        paper = pd.read_csv(paper_table)
        if "candidate_id" not in paper.columns:
            raise ValueError(f"{paper_table} has no candidate_id column")
        return list(dict.fromkeys(paper["candidate_id"].dropna().astype(str)))

    rows = conn.execute(
        """
        SELECT r.candidate_id
        FROM reviews AS r
        WHERE lower(trim(coalesce(r.event_class, ''))) = 'dipper'
          AND lower(trim(coalesce(r.status, ''))) = 'reviewed'
          AND lower(trim(coalesce(r.disposition, ''))) = 'keep'
        ORDER BY r.updated_at DESC, r.candidate_id
        """
    ).fetchall()
    return [str(row[0]) for row in rows]


def load_candidate_context(
    run_dir: Path,
    *,
    candidate_ids: list[str],
) -> CandidateContext:
    db_path = run_dir / "review" / "review.db"
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    if not candidate_ids:
        raise ValueError("No candidate IDs were selected.")

    placeholders = ",".join("?" for _ in candidate_ids)
    with _connect_readonly(db_path) as conn:
        quick_check = conn.execute("PRAGMA quick_check").fetchone()
        if not quick_check or str(quick_check[0]).lower() != "ok":
            raise RuntimeError(f"Review database quick_check failed: {quick_check}")

        candidates = pd.read_sql_query(
            f"""
            SELECT c.*,
                   r.event_class,
                   r.status AS review_status,
                   r.workflow_status,
                   r.disposition,
                   r.morphology_primary,
                   r.morphology_secondary,
                   r.physical_primary,
                   r.physical_secondary,
                   r.classification_confidence,
                   r.notes AS review_notes,
                   r.updated_at AS review_updated_at
            FROM candidates AS c
            LEFT JOIN reviews AS r USING(candidate_id)
            WHERE c.candidate_id IN ({placeholders})
            """,
            conn,
            params=candidate_ids,
        )
        candidates["candidate_id"] = candidates["candidate_id"].astype(str)
        found = set(candidates["candidate_id"])
        missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in found]
        if missing:
            raise KeyError(f"Candidate IDs absent from Review DB: {', '.join(missing)}")

        sed_rows: dict[str, pd.DataFrame] = {}
        sed_curves: dict[str, pd.DataFrame] = {}
        sed_fits: dict[str, pd.DataFrame] = {}
        sed_points: dict[str, pd.DataFrame] = {}
        for candidate_id in candidate_ids:
            sed_rows[candidate_id] = load_sed_rows(conn, candidate_id)
            sed_curves[candidate_id] = load_sed_model_curves(conn, candidate_id)
            sed_fits[candidate_id] = load_sed_model_fits(conn, candidate_id)
            sed_points[candidate_id] = load_sed_model_points(conn, candidate_id)

    return CandidateContext(
        run_dir=run_dir,
        candidate_rows=candidates.set_index("candidate_id", drop=False),
        sed_rows=sed_rows,
        sed_curves=sed_curves,
        sed_fits=sed_fits,
        sed_points=sed_points,
    )


def select_spectrum_rows(
    spectra_long: pd.DataFrame,
    candidate_ids: list[str],
    *,
    surveys: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Choose the nearest unique spectrum row per candidate and survey."""

    if "candidate_id" not in spectra_long.columns or "survey" not in spectra_long.columns:
        raise ValueError("spectra_long must contain candidate_id and survey columns")
    selected = spectra_long.copy()
    selected["candidate_id"] = selected["candidate_id"].astype(str)
    selected["survey"] = selected["survey"].fillna("").astype(str).str.strip()
    selected = selected[
        selected["candidate_id"].isin(set(map(str, candidate_ids)))
        & selected["survey"].ne("")
    ].copy()
    if "spectrum_record_status" in selected.columns:
        selected = selected[
            selected["spectrum_record_status"].fillna("available").astype(str).eq("available")
        ].copy()
    survey_filter = {str(item).strip() for item in (surveys or []) if str(item).strip()}
    if survey_filter:
        selected = selected[selected["survey"].isin(survey_filter)].copy()
    if selected.empty:
        return selected

    selected["_sep_sort"] = pd.to_numeric(
        selected.get("sep_arcsec", pd.Series(np.nan, index=selected.index)),
        errors="coerce",
    ).fillna(np.inf)
    selected["_candidate_order"] = pd.Categorical(
        selected["candidate_id"],
        categories=list(dict.fromkeys(map(str, candidate_ids))),
        ordered=True,
    )
    selected = (
        selected.sort_values(
            ["_candidate_order", "survey", "_sep_sort"],
            kind="mergesort",
        )
        .drop_duplicates(["candidate_id", "survey"], keep="first")
        .drop(columns=["_sep_sort", "_candidate_order"])
        .reset_index(drop=True)
    )
    return selected


def _candidate_coordinate_name(row: pd.Series) -> str:
    ra = _finite_float(row.get("ra"))
    dec = _finite_float(row.get("dec"))
    if ra is None or dec is None:
        return str(row.get("candidate_id") or "unknown")
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
        ra_text = coord.ra.to_string(unit=u.hourangle, sep="", precision=0, pad=True)
        dec_text = coord.dec.to_string(
            unit=u.deg,
            sep="",
            precision=0,
            pad=True,
            alwayssign=True,
        )
        return f"J{ra_text}{dec_text}"
    except Exception:
        sign = "+" if dec >= 0 else "-"
        return f"RA {ra:.5f}, Dec {sign}{abs(dec):.5f}"


def _resolve_lightcurve_path(path_value: object, run_dir: Path) -> Path | None:
    text = str(path_value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((run_dir / path, Path.cwd() / path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _asassn_jd_minus_2458000(frame: pd.DataFrame) -> pd.Series:
    time = pd.to_numeric(frame["time"], errors="coerce")
    finite = time[np.isfinite(time)]
    median = float(finite.median()) if not finite.empty else np.nan
    if np.isfinite(median) and median > 2_000_000:
        return time - 2_458_000.0
    if np.isfinite(median) and median > 40_000:
        return time - 57_999.5
    if np.isfinite(median) and median > 5_000:
        return time - 8_000.0
    return time


def _median_center(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["centered_mag"] = (
        out["mag"]
        - out.groupby("band", observed=True)["mag"].transform("median")
    )
    return out


def _empty_axis(ax: Any, message: str) -> None:
    ax.text(
        0.5,
        0.5,
        _latex_text(message),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color="0.35",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("black")


def draw_lightcurve_context(
    ax: Any,
    context: CandidateContext,
    candidate_id: str,
) -> dict[str, object]:
    """Draw median-centered ASAS-SN V/g photometry."""

    row = context.candidate_row(candidate_id)
    optical_path = _resolve_lightcurve_path(row.get("lc_path"), context.run_dir)
    optical = pd.DataFrame()
    if optical_path is not None:
        lightcurve = load_lightcurve(optical_path)
        plot_frame = filter_lightcurve(lightcurve, max_error=1.0)
        if not plot_frame.empty:
            optical = pd.DataFrame(
                {
                    "jd_minus_2458000": _asassn_jd_minus_2458000(plot_frame),
                    "mag": pd.to_numeric(plot_frame["value"], errors="coerce"),
                    "mag_err": pd.to_numeric(plot_frame["value_error"], errors="coerce"),
                    "band": plot_frame["band"].astype(str),
                    "survey": "ASAS-SN",
                }
            )
            good = (
                np.isfinite(optical["jd_minus_2458000"])
                & np.isfinite(optical["mag"])
                & optical["band"].isin(["V", "g"])
            )
            optical = optical.loc[good].copy()

    if optical.empty:
        _empty_axis(ax, "No ASAS-SN light curve available")
        return {
            "asassn_status": "missing",
            "asassn_n_points": 0,
            "asassn_path": str(optical_path or ""),
        }

    combined = _median_center(optical)
    for band in ("V", "g"):
        subset = combined[combined["band"].astype(str).eq(band)]
        if subset.empty:
            continue
        style = LIGHTCURVE_STYLES[band]
        x = pd.to_numeric(subset["jd_minus_2458000"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(subset["centered_mag"], errors="coerce").to_numpy(dtype=float)
        err = pd.to_numeric(subset["mag_err"], errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        if not bool(good.any()):
            continue
        with_err = good & np.isfinite(err) & (err > 0)
        if bool(with_err.any()):
            ax.errorbar(
                x[with_err],
                y[with_err],
                yerr=err[with_err],
                fmt="none",
                ecolor=style["color"],
                elinewidth=0.35,
                alpha=0.42,
                capsize=0,
                zorder=2,
            )
        ax.scatter(
            x[good],
            y[good],
            s=7.0,
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor="black",
            linewidth=0.18,
            alpha=0.82,
            label=band,
            zorder=3,
        )

    ax.axhline(0.0, color="0.55", linewidth=0.55, linestyle=":", zorder=1)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\mathrm{JD} - 2458000\ [\mathrm{d}]$")
    ax.set_ylabel(r"$\Delta m$ [mag]")
    ax.set_title("ASAS-SN (bands median centered)", loc="left")
    ax.legend(
        loc="upper right",
        ncol=2,
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        borderpad=0.25,
        handletextpad=0.25,
        columnspacing=0.65,
    )
    _style_axis(ax)
    return {
        "asassn_status": "ok" if not optical.empty else "missing",
        "asassn_n_points": int(len(optical)),
        "asassn_path": str(optical_path or ""),
    }


def draw_sed_context(
    ax: Any,
    context: CandidateContext,
    candidate_id: str,
) -> dict[str, object]:
    rows = context.sed_rows.get(candidate_id, pd.DataFrame())
    curves = context.sed_curves.get(candidate_id, pd.DataFrame())
    fits = context.sed_fits.get(candidate_id, pd.DataFrame())
    points = context.sed_points.get(candidate_id, pd.DataFrame())
    warnings = render_sed_matplotlib(
        ax,
        context.payload(candidate_id),
        candidate_id=candidate_id,
        external_rows=rows,
        model_curve_rows=curves,
        model_fit_rows=fits,
        model_point_rows=points,
        extinction_mode="observed",
        theme="white",
        y_axis_side="left",
    )
    ax.set_title("Spectral energy distribution", loc="left")
    ax.yaxis.set_label_coords(-0.085, 0.5)
    _style_axis(ax)
    return {
        "sed_status": "ok" if not rows.empty else "missing",
        "sed_n_points": int(len(rows)),
        "sed_fit_version": (
            str(fits.iloc[0].get("fit_version") or "") if not fits.empty else ""
        ),
        "sed_warnings": " ".join(map(str, warnings)),
    }


def _style_axis(ax: Any) -> None:
    ax.set_facecolor("white")
    ax.grid(False, which="both")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.75)
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=3.6,
        width=0.75,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=2.0,
        width=0.55,
    )


def _spectrum_segments(
    wavelength: np.ndarray,
    mask: np.ndarray,
    *,
    gap_factor: float = 5.0,
    minimum_gap_angstrom: float = 2.0,
) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    finite_wave = wavelength[indices]
    steps = np.diff(finite_wave)
    positive = steps[np.isfinite(steps) & (steps > 0)]
    typical = float(np.nanmedian(positive)) if positive.size else 0.0
    threshold = max(float(minimum_gap_angstrom), float(gap_factor) * typical)
    breaks = np.flatnonzero((np.diff(indices) > 1) | (steps > threshold))
    return [segment for segment in np.split(indices, breaks + 1) if segment.size > 1]


def _robust_limits(
    values: np.ndarray,
    *,
    lower: float = 0.5,
    upper: float = 99.5,
    pad_fraction: float = 0.06,
) -> tuple[float, float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    lo, hi = np.nanpercentile(finite, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi <= lo:
        pad = max(abs(float(lo)) * pad_fraction, 1e-3)
    else:
        pad = max(float(hi - lo) * pad_fraction, 1e-3)
    return float(lo - pad), float(hi + pad)


def draw_spectrum_context(
    raw_ax: Any,
    residual_ax: Any,
    spectrum: SpectrumData,
    *,
    survey: str,
) -> dict[str, object]:
    """Draw the standard raw/pseudo-continuum and normalized spectrum panels."""

    # Zero-valued masked pixels are common in survey products.  The shared
    # analysis code deliberately turns their relative errors into NaN; silence
    # the corresponding harmless NumPy divide warnings for batch exports.
    with np.errstate(divide="ignore", invalid="ignore"):
        analysis = analyze_spectrum(spectrum)
    wavelength = np.asarray(analysis.wavelength, dtype=float)
    flux = np.asarray(analysis.flux, dtype=float)
    continuum = np.asarray(analysis.continuum, dtype=float)
    residual = np.asarray(analysis.normalized_residual_flux, dtype=float)
    finite = np.asarray(analysis.finite, dtype=bool)
    finite &= np.isfinite(wavelength) & np.isfinite(flux)
    segments = _spectrum_segments(wavelength, finite)

    if analysis.flux_err is not None:
        flux_err = np.asarray(analysis.flux_err, dtype=float)
        for segment in segments:
            err_good = np.isfinite(flux_err[segment]) & (flux_err[segment] >= 0)
            if not bool(err_good.any()):
                continue
            segment = segment[err_good]
            raw_ax.fill_between(
                wavelength[segment],
                flux[segment] - flux_err[segment],
                flux[segment] + flux_err[segment],
                color="0.20",
                alpha=0.09,
                linewidth=0,
                zorder=1,
            )

    for index, segment in enumerate(segments):
        raw_ax.plot(
            wavelength[segment],
            flux[segment],
            color="black",
            linewidth=0.55,
            label="spectrum" if index == 0 else "_nolegend_",
            zorder=2,
        )
        continuum_good = segment[np.isfinite(continuum[segment])]
        if continuum_good.size > 1:
            raw_ax.plot(
                wavelength[continuum_good],
                continuum[continuum_good],
                color="#b2182b",
                linewidth=0.78,
                alpha=0.82,
                label="pseudo-continuum" if index == 0 else "_nolegend_",
                zorder=3,
            )
        residual_good = segment[np.isfinite(residual[segment])]
        if residual_good.size > 1:
            residual_ax.plot(
                wavelength[residual_good],
                residual[residual_good],
                color="black",
                linewidth=0.52,
                zorder=2,
            )

    strongest = sorted(
        analysis.line_fits,
        key=lambda item: abs(float(item.significance)),
        reverse=True,
    )[:12]
    for line in strongest:
        color = "#2166ac" if str(line.line_type) == "absorption" else "#b2182b"
        residual_ax.axvline(
            float(line.center),
            color=color,
            linewidth=0.42,
            alpha=0.30,
            zorder=1,
        )

    raw_bounds = _robust_limits(
        np.concatenate(
            (
                flux[finite],
                continuum[finite & np.isfinite(continuum)],
            )
        )
    )
    residual_bounds = _robust_limits(residual, lower=0.5, upper=99.5)
    if raw_bounds is not None:
        raw_ax.set_ylim(*raw_bounds)
    if residual_bounds is not None:
        residual_ax.set_ylim(*residual_bounds)
    if bool(finite.any()):
        x = wavelength[finite]
        x_pad = max(1.0, 0.005 * float(np.nanmax(x) - np.nanmin(x)))
        residual_ax.set_xlim(float(np.nanmin(x) - x_pad), float(np.nanmax(x) + x_pad))

    label = SOURCE_LABELS.get(str(survey), str(survey).replace("_", " ").upper())
    raw_ax.set_title(f"Spectrum - {_latex_text(label)}", loc="left")
    raw_ax.set_ylabel(r"$F_\lambda$ [native units]")
    raw_ax.tick_params(labelbottom=False)
    raw_ax.legend(
        loc="upper right",
        frameon=False,
        ncol=2,
        handlelength=1.6,
        columnspacing=0.8,
    )
    residual_ax.axhline(0.0, color="0.45", linewidth=0.55, linestyle=":", zorder=1)
    residual_ax.set_xlabel(r"Observed wavelength [$\AA$]")
    residual_ax.set_ylabel(r"$F_\lambda/F_{\lambda,\mathrm{cont}} - 1$")
    _style_axis(raw_ax)
    _style_axis(residual_ax)

    good_wave = wavelength[finite]
    return {
        "spectrum_n_points": int(np.count_nonzero(finite)),
        "spectrum_wavelength_min_angstrom": (
            float(np.nanmin(good_wave)) if good_wave.size else np.nan
        ),
        "spectrum_wavelength_max_angstrom": (
            float(np.nanmax(good_wave)) if good_wave.size else np.nan
        ),
        "spectrum_n_detected_lines": int(len(analysis.line_fits)),
    }


def build_candidate_page(
    context: CandidateContext,
    candidate_id: str,
    spectrum: SpectrumData,
    *,
    survey: str,
    redshift: float | None = None,
) -> tuple[Any, dict[str, object]]:
    row = context.candidate_row(candidate_id)
    coordinate_name = _candidate_coordinate_name(row)
    source_label = SOURCE_LABELS.get(str(survey), str(survey).replace("_", " ").upper())

    with plt.rc_context(CONTEXT_STYLE):
        fig = plt.figure(figsize=(PAGE_WIDTH_IN, PAGE_HEIGHT_IN), facecolor="white")
        grid = fig.add_gridspec(
            4,
            1,
            height_ratios=(1.20, 1.38, 1.42, 0.95),
            hspace=0.37,
        )
        lightcurve_ax = fig.add_subplot(grid[0, 0])
        sed_ax = fig.add_subplot(grid[1, 0])
        spectrum_ax = fig.add_subplot(grid[2, 0])
        residual_ax = fig.add_subplot(grid[3, 0], sharex=spectrum_ax)
        fig.subplots_adjust(left=0.105, right=0.975, bottom=0.070, top=0.945)

        redshift_text = (
            rf", $z={float(redshift):.4f}$"
            if redshift is not None and np.isfinite(redshift)
            else ""
        )
        fig.suptitle(
            (
                f"{_latex_text(coordinate_name)} "
                rf"(\texttt{{{_latex_text(candidate_id)}}}) - "
                f"{_latex_text(source_label)}{redshift_text}"
            ),
            fontsize=10.5,
            y=0.982,
        )

        lightcurve_status = draw_lightcurve_context(
            lightcurve_ax,
            context,
            candidate_id,
        )
        sed_status = draw_sed_context(sed_ax, context, candidate_id)
        spectrum_status = draw_spectrum_context(
            spectrum_ax,
            residual_ax,
            spectrum,
            survey=survey,
        )
        status = {
            "candidate_id": str(candidate_id),
            "coordinate_name": coordinate_name,
            "survey": str(survey),
            **lightcurve_status,
            **sed_status,
            **spectrum_status,
        }
        return fig, status


def _cache_path(cache_dir: Path, candidate_id: str, survey: str) -> Path:
    return cache_dir / f"{candidate_id}_{survey}.npz"


def load_or_fetch_spectrum(
    row: pd.Series,
    *,
    cache_dir: Path,
    cache_only: bool,
) -> tuple[str, SpectrumData | None, str, str]:
    candidate_id = str(row.get("candidate_id") or "")
    survey = str(row.get("survey") or "")
    cache_path = _cache_path(cache_dir, candidate_id, survey)
    cached = load_spectrum_cache(cache_path)
    if cached is not None:
        return "cached", cached, "", str(cache_path)
    if cache_only:
        return "not_cached", None, "Spectrum is not present in the local cache.", str(cache_path)

    result = fetch_spectrum(
        row,
        survey_key=survey,
        cache_dir=cache_dir,
    )
    if result.status == FetchStatus.OK and result.data is not None:
        return "downloaded", result.data, str(result.message or ""), str(cache_path)
    message = str(result.message or result.link or result.status.value)
    return result.status.value, None, message, str(cache_path)


def generate_pdfs(
    context: CandidateContext,
    spectrum_rows: pd.DataFrame,
    *,
    output_dir: Path,
    cache_dir: Path,
    cache_only: bool,
    merge_atlas: bool,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = output_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    atlas_path = output_dir / "dipper_spectra_context_atlas.pdf"
    manifest_records: list[dict[str, object]] = []

    atlas_writer = (
        PdfPages(atlas_path, metadata={"Creator": "MALCA", "Title": "Dipper spectra context atlas"})
        if merge_atlas
        else None
    )
    try:
        for candidate_id, candidate_group in spectrum_rows.groupby("candidate_id", sort=False):
            candidate_path = pdf_dir / (
                f"dipper_spectra_context_{_safe_filename(candidate_id)}.pdf"
            )
            candidate_writer: PdfPages | None = None
            candidate_page = 0
            try:
                for _, spectrum_row in candidate_group.iterrows():
                    survey = str(spectrum_row.get("survey") or "")
                    fetch_status, spectrum, fetch_message, cache_path = load_or_fetch_spectrum(
                        spectrum_row,
                        cache_dir=cache_dir,
                        cache_only=cache_only,
                    )
                    base_record = {
                        "candidate_id": str(candidate_id),
                        "survey": survey,
                        "spectrum_fetch_status": fetch_status,
                        "spectrum_fetch_message": fetch_message,
                        "spectrum_cache_path": cache_path,
                        "output_pdf": "",
                        "candidate_pdf_page": np.nan,
                        "atlas_pdf_page": np.nan,
                    }
                    if spectrum is None:
                        manifest_records.append(base_record)
                        continue

                    redshift = _finite_float(spectrum_row.get("spectrum_redshift"))
                    try:
                        fig, page_status = build_candidate_page(
                            context,
                            str(candidate_id),
                            spectrum,
                            survey=survey,
                            redshift=redshift,
                        )
                    except Exception as exc:
                        manifest_records.append(
                            {
                                **base_record,
                                "spectrum_fetch_status": "plot_error",
                                "spectrum_fetch_message": str(exc),
                            }
                        )
                        continue

                    if candidate_writer is None:
                        candidate_writer = PdfPages(
                            candidate_path,
                            metadata={
                                "Creator": "MALCA",
                                "Title": f"Dipper spectrum context: {candidate_id}",
                            },
                        )
                    candidate_page += 1
                    # Matplotlib resolves ``text.usetex`` and its LaTeX
                    # preamble at draw/save time.  Keep the publication style
                    # active here so returned figures still embed cmbright.
                    with plt.rc_context(CONTEXT_STYLE):
                        candidate_writer.savefig(fig, dpi=300)
                        if atlas_writer is not None:
                            atlas_writer.savefig(fig, dpi=300)
                    plt.close(fig)
                    manifest_records.append(
                        {
                            **base_record,
                            **page_status,
                            "output_pdf": str(candidate_path),
                            "candidate_pdf_page": candidate_page,
                            "atlas_pdf_page": sum(
                                bool(record.get("output_pdf"))
                                for record in manifest_records
                            )
                            + 1,
                        }
                    )
            finally:
                if candidate_writer is not None:
                    candidate_writer.close()
    finally:
        if atlas_writer is not None:
            atlas_writer.close()

    manifest = pd.DataFrame(manifest_records)
    manifest_path = output_dir / "dipper_spectra_context_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    if merge_atlas and (
        manifest.empty
        or "output_pdf" not in manifest.columns
        or not bool(manifest["output_pdf"].fillna("").astype(str).str.len().gt(0).any())
    ):
        atlas_path.unlink(missing_ok=True)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create one publication-quality ASAS-SN/SED/spectrum PDF "
            "per dipper with a matched spectral survey."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="New copy-only output root (default: %(default)s).",
    )
    parser.add_argument(
        "--cohort",
        choices=("paper", "review"),
        default="paper",
        help="Paper candidate table (default) or all current reviewed/kept dippers.",
    )
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help="Restrict to one candidate; repeat to select multiple candidates.",
    )
    parser.add_argument(
        "--survey",
        action="append",
        default=[],
        help="Restrict to a survey key such as apogee_dr16; repeat as needed.",
    )
    parser.add_argument(
        "--paper-table",
        type=Path,
        default=None,
        help="Candidate table for --cohort paper.",
    )
    parser.add_argument(
        "--spectra-long",
        type=Path,
        default=None,
        help="Run-local spectra_long parquet.",
    )
    parser.add_argument(
        "--spectrum-cache-dir",
        type=Path,
        default=None,
        help="NPZ spectrum cache directory.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Do not access remote archives; plot only locally cached spectra.",
    )
    parser.add_argument(
        "--no-atlas",
        action="store_true",
        help="Do not create the merged multi-candidate atlas PDF.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit selected candidates for a development smoke run.",
    )
    return parser


def run(args: argparse.Namespace) -> pd.DataFrame:
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    paper_table = (
        Path(args.paper_table).expanduser().resolve()
        if args.paper_table is not None
        else run_dir / DEFAULT_PAPER_TABLE_RELATIVE
    )
    spectra_long_path = (
        Path(args.spectra_long).expanduser().resolve()
        if args.spectra_long is not None
        else run_dir / DEFAULT_SPECTRA_LONG_RELATIVE
    )
    cache_dir = (
        Path(args.spectrum_cache_dir).expanduser().resolve()
        if args.spectrum_cache_dir is not None
        else run_dir / DEFAULT_SPECTRUM_CACHE_RELATIVE
    )
    db_path = run_dir / "review" / "review.db"
    if not spectra_long_path.exists():
        raise FileNotFoundError(spectra_long_path)

    with _connect_readonly(db_path) as conn:
        candidate_ids = _query_candidate_ids(
            conn,
            cohort=str(args.cohort),
            paper_table=paper_table,
            explicit_ids=args.candidate_id,
        )
    if args.limit is not None:
        if int(args.limit) <= 0:
            raise ValueError("--limit must be positive")
        candidate_ids = candidate_ids[: int(args.limit)]

    spectra_long = pd.read_parquet(spectra_long_path)
    spectrum_rows = select_spectrum_rows(
        spectra_long,
        candidate_ids,
        surveys=args.survey,
    )
    if spectrum_rows.empty:
        raise RuntimeError("No spectral catalogue rows matched the selected candidates.")
    plotted_candidate_ids = list(dict.fromkeys(spectrum_rows["candidate_id"].astype(str)))
    context = load_candidate_context(run_dir, candidate_ids=plotted_candidate_ids)
    manifest = generate_pdfs(
        context,
        spectrum_rows,
        output_dir=output_dir,
        cache_dir=cache_dir,
        cache_only=bool(args.cache_only),
        merge_atlas=not bool(args.no_atlas),
    )

    plotted = int(
        manifest.get("output_pdf", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.len()
        .gt(0)
        .sum()
    )
    summary = {
        "run_dir": str(run_dir),
        "cohort": str(args.cohort),
        "selected_candidates": int(len(candidate_ids)),
        "candidates_with_catalogue_rows": int(len(plotted_candidate_ids)),
        "selected_spectrum_sources": int(len(spectrum_rows)),
        "rendered_pages": plotted,
        "rendered_candidates": int(
            manifest.loc[
                manifest.get("output_pdf", pd.Series(dtype=str)).fillna("").astype(str).str.len().gt(0),
                "candidate_id",
            ].nunique()
            if plotted
            else 0
        ),
        "cache_only": bool(args.cache_only),
        "manifest": str(output_dir / "dipper_spectra_context_manifest.csv"),
        "atlas": str(output_dir / "dipper_spectra_context_atlas.pdf")
        if not args.no_atlas and plotted
        else "",
    }
    (output_dir / "dipper_spectra_context_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
