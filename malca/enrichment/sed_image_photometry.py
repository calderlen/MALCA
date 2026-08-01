"""Resumable image-level SED photometry worker.

The worker deliberately emits ``pending_validation`` measurements.  It never
promotes a forced flux or upper limit into an R24 input; that requires a
separate versioned validation record.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
import json
import math
from pathlib import Path
import socket
import tarfile
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from malca.enrichment.sed_archive import download_archive_product
from malca.review.sed import _direct_flux_sed_row, upsert_sed_rows
from malca.review.sed_storage import (
    ensure_sed_storage_schema,
    load_sed_archive_coverage,
    load_sed_archive_products,
    load_sed_image_jobs,
    prepare_canonical_sed_rows,
    update_sed_image_job,
    upsert_sed_archive_products,
)
from malca.review.store import db_connect


IMAGE_MEASUREMENT_VERSION = "sed-image-aperture-v1"


class NoUsableCoverageError(RuntimeError):
    """Raised when the target is not on valid pixels in the retrieved map."""


class UnsupportedImageUnitError(RuntimeError):
    """Raised rather than guessing how a map unit should be converted to Jy."""


@dataclass(frozen=True)
class ApertureMeasurement:
    flux_jy: float
    flux_error_jy: float
    is_upper_limit: bool
    aperture_radius_arcsec: float
    annulus_inner_arcsec: float
    annulus_outer_arcsec: float
    beam_fwhm_arcsec: float
    n_aperture_pixels: int
    n_annulus_pixels: int
    background_native_per_pixel: float
    noise_native_per_pixel: float
    native_unit: str
    conversion_jy_per_native_pixel: float


BEAM_FWHM_ARCSEC = {
    "IRAC1": 1.66,
    "IRAC2": 1.72,
    "IRAC3": 1.88,
    "IRAC4": 1.98,
    "MIPS24": 6.0,
    "PACS70": 5.6,
    "PACS100": 6.8,
    "PACS160": 11.4,
    "SPIRE250": 18.2,
    "SPIRE350": 24.9,
    "SPIRE500": 36.3,
    "SABOCA350": 7.8,
    "LABOCA870": 19.2,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malca sed-image-photometry",
        description=(
            "Resume archive-product downloads and create provisional forced "
            "photometry/upper limits from covered images."
        ),
    )
    parser.add_argument("review_db", type=Path, help="Review SQLite database")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/sed_archive"),
        help="Downloaded archive-product cache (default: cache/sed_archive)",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=25,
        help="Maximum queued/retry jobs to process in this invocation (default: 25)",
    )
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help="Limit the worker to this candidate ID; repeat as needed.",
    )
    parser.add_argument(
        "--worker-id",
        default=f"{socket.gethostname()}:{IMAGE_MEASUREMENT_VERSION}",
        help="Worker identity written to job leases/provenance.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download products but leave measurement jobs pending manual review.",
    )
    parser.add_argument(
        "--include-reduction-required",
        action="store_true",
        help=(
            "Download reduction-required APEX products too. They remain queued "
            "for instrument-specific reduction and are never aperture measured."
        ),
    )
    return parser


def _json_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _extract_fits_archive(path: Path) -> list[Path]:
    destination = path.with_name(f"{path.name}.extracted")
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(path, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            member_path = (destination / member.name).resolve()
            if destination_root not in member_path.parents:
                raise ValueError(f"Unsafe archive member path: {member.name!r}")
            member_path.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, member_path.open("wb") as handle:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
    return sorted(
        path
        for path in destination.rglob("*")
        if path.is_file() and path.name.lower().endswith((".fits", ".fit", ".fits.gz"))
    )


def _fits_paths(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(
            item
            for item in path.rglob("*")
            if item.is_file()
            and item.name.lower().endswith((".fits", ".fit", ".fits.gz", ".fz"))
        )
    lower = path.name.lower()
    if lower.endswith((".tar", ".tar.gz", ".tgz")):
        return _extract_fits_archive(path)
    if lower.endswith((".fits", ".fit", ".fits.gz", ".fz")):
        return [path]
    return []


def _classify_fits_product(path: Path) -> str:
    text = path.name.lower()
    if any(token in text for token in ("unc", "error", "sigma", "std", "noise")):
        return "uncertainty_map"
    if any(token in text for token in ("cov", "coverage", "hits", "nhit")):
        return "coverage_map"
    if any(token in text for token in ("mask", "flag")):
        return "mask"
    return "science_image"


def _band_from_header_or_path(
    header: Mapping[str, Any],
    path: Path,
    *,
    fallback: str | None,
) -> str | None:
    if fallback:
        return str(fallback)
    tokens = " ".join(
        str(header.get(key, "")) for key in ("FILTER", "BAND", "CAMERA", "INSTRUME")
    )
    tokens = f"{tokens} {path.name}".upper().replace("_", "").replace("-", "")
    aliases = (
        (("IRAC1", "3.6", "I1"), "IRAC1"),
        (("IRAC2", "4.5", "I2"), "IRAC2"),
        (("IRAC3", "5.8", "I3"), "IRAC3"),
        (("IRAC4", "8.0", "I4"), "IRAC4"),
        (("MIPS24", "MIPS1", "24UM"), "MIPS24"),
        (("PACS70", "BLUE", "70UM"), "PACS70"),
        (("PACS100", "GREEN", "100UM"), "PACS100"),
        (("PACS160", "RED", "160UM"), "PACS160"),
        (("SPIRE250", "PSW"), "SPIRE250"),
        (("SPIRE350", "PMW"), "SPIRE350"),
        (("SPIRE500", "PLW"), "SPIRE500"),
    )
    for names, band in aliases:
        if any(name in tokens for name in names):
            return band
    return None


def _image_plane(path: Path) -> tuple[np.ndarray, Any, Mapping[str, Any]]:
    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(path, memmap=False) as hdus:
        for hdu in hdus:
            data = getattr(hdu, "data", None)
            if data is None or np.ndim(data) < 2:
                continue
            array = np.asarray(data, dtype=float)
            while array.ndim > 2:
                array = array[0]
            try:
                wcs = WCS(hdu.header).celestial
                if not wcs.has_celestial:
                    continue
            except Exception:
                continue
            return array, wcs, dict(hdu.header)
    raise ValueError(f"No two-dimensional celestial FITS image in {path}")


def _target_pixel(wcs: Any, ra_deg: float, dec_deg: float) -> tuple[float, float]:
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    x, y = wcs.world_to_pixel(SkyCoord(ra_deg * u.deg, dec_deg * u.deg))
    return float(x), float(y)


def _pixel_scale_arcsec(wcs: Any) -> float:
    from astropy.wcs.utils import proj_plane_pixel_scales

    scales = np.asarray(proj_plane_pixel_scales(wcs), dtype=float) * 3600.0
    valid = scales[np.isfinite(scales) & (scales > 0)]
    if valid.size == 0:
        raise ValueError("FITS WCS has no finite celestial pixel scale")
    return float(np.sqrt(np.prod(valid[:2])))


def _conversion_jy_per_native_pixel(
    native_unit: str,
    *,
    pixel_scale_arcsec: float,
    beam_fwhm_arcsec: float,
) -> float:
    unit = (
        str(native_unit or "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("steradian", "sr")
        .replace("pixels", "pix")
        .replace("pixel", "pix")
    )
    pixel_area_sr = (pixel_scale_arcsec / 206264.80624709636) ** 2
    beam_area_arcsec2 = math.pi * beam_fwhm_arcsec**2 / (4.0 * math.log(2.0))
    pixel_area_arcsec2 = pixel_scale_arcsec**2
    if unit in {"mjy/sr", "mjysr-1", "mjyper_sr"}:
        return 1.0e6 * pixel_area_sr
    if unit in {"jy/sr", "jysr-1"}:
        return pixel_area_sr
    if unit in {"jy/pix", "jy/pix-1", "jy"}:
        return 1.0
    if unit in {"mjy/pix", "mjy/pix-1", "mjy"}:
        return 1.0e-3
    if unit in {"jy/beam", "jybeam-1"}:
        return pixel_area_arcsec2 / beam_area_arcsec2
    if unit in {"mjy/beam", "mjybeam-1"}:
        return 1.0e-3 * pixel_area_arcsec2 / beam_area_arcsec2
    raise UnsupportedImageUnitError(
        f"Unsupported BUNIT {native_unit!r}; refusing an implicit flux conversion"
    )


def _coverage_is_valid(
    coverage_path: Path | None,
    *,
    ra_deg: float,
    dec_deg: float,
) -> bool:
    if coverage_path is None:
        return True
    try:
        coverage, wcs, _header = _image_plane(coverage_path)
        x, y = _target_pixel(wcs, ra_deg, dec_deg)
        ix, iy = int(round(x)), int(round(y))
        return (
            0 <= iy < coverage.shape[0]
            and 0 <= ix < coverage.shape[1]
            and np.isfinite(coverage[iy, ix])
            and coverage[iy, ix] > 0
        )
    except Exception:
        return False


def measure_fits_aperture(
    science_path: Path,
    *,
    ra_deg: float,
    dec_deg: float,
    band: str,
    coverage_path: Path | None = None,
) -> ApertureMeasurement:
    """Measure a background-subtracted aperture flux or a local 3-sigma limit."""

    if not _coverage_is_valid(coverage_path, ra_deg=ra_deg, dec_deg=dec_deg):
        raise NoUsableCoverageError("target is not on positive coverage-map pixels")
    image, wcs, header = _image_plane(science_path)
    x, y = _target_pixel(wcs, ra_deg, dec_deg)
    if not np.isfinite(x) or not np.isfinite(y):
        raise NoUsableCoverageError("target WCS position is not finite")
    if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
        raise NoUsableCoverageError("target lies outside the science image")

    pixel_scale = _pixel_scale_arcsec(wcs)
    beam = BEAM_FWHM_ARCSEC.get(str(band))
    if beam is None:
        raise ValueError(f"No aperture/beam policy is registered for band {band!r}")
    aperture_radius = max(beam, 2.0 * pixel_scale)
    annulus_inner = max(1.75 * beam, aperture_radius + pixel_scale)
    annulus_outer = max(2.75 * beam, annulus_inner + 2.0 * pixel_scale)
    y_grid, x_grid = np.indices(image.shape, dtype=float)
    radius_arcsec = np.hypot(x_grid - x, y_grid - y) * pixel_scale
    aperture_mask = radius_arcsec <= aperture_radius
    annulus_mask = (radius_arcsec >= annulus_inner) & (radius_arcsec <= annulus_outer)
    aperture_values = image[aperture_mask & np.isfinite(image)]
    annulus_values = image[annulus_mask & np.isfinite(image)]
    if aperture_values.size < 3 or annulus_values.size < 12:
        raise NoUsableCoverageError("too few finite aperture/background pixels")

    background = float(np.median(annulus_values))
    mad = float(np.median(np.abs(annulus_values - background)))
    noise_native = 1.4826 * mad
    if not np.isfinite(noise_native) or noise_native <= 0:
        noise_native = float(np.std(annulus_values, ddof=1))
    if not np.isfinite(noise_native) or noise_native <= 0:
        raise NoUsableCoverageError("local background noise is not measurable")

    native_flux = float(np.sum(aperture_values - background))
    native_error = float(
        noise_native
        * np.sqrt(
            aperture_values.size
            * (1.0 + aperture_values.size / max(annulus_values.size, 1))
        )
    )
    native_unit = str(header.get("BUNIT", "")).strip()
    conversion = _conversion_jy_per_native_pixel(
        native_unit,
        pixel_scale_arcsec=pixel_scale,
        beam_fwhm_arcsec=beam,
    )
    flux_jy = native_flux * conversion
    error_jy = native_error * conversion
    is_upper_limit = not np.isfinite(flux_jy) or flux_jy <= 3.0 * error_jy
    reported_flux = 3.0 * error_jy if is_upper_limit else flux_jy
    return ApertureMeasurement(
        flux_jy=float(reported_flux),
        flux_error_jy=float(error_jy),
        is_upper_limit=bool(is_upper_limit),
        aperture_radius_arcsec=float(aperture_radius),
        annulus_inner_arcsec=float(annulus_inner),
        annulus_outer_arcsec=float(annulus_outer),
        beam_fwhm_arcsec=float(beam),
        n_aperture_pixels=int(aperture_values.size),
        n_annulus_pixels=int(annulus_values.size),
        background_native_per_pixel=background,
        noise_native_per_pixel=noise_native,
        native_unit=native_unit,
        conversion_jy_per_native_pixel=float(conversion),
    )


def _source_for_band(band: str) -> tuple[str, str]:
    if band.startswith(("IRAC", "MIPS")):
        return "Spitzer SEIP", "SEIP image"
    if band.startswith(("PACS", "SPIRE")):
        return "Herschel", "HSA image"
    if band.startswith(("LABOCA", "SABOCA")):
        return "APEX", "ESO image"
    raise ValueError(f"Unrecognized archive-image band {band!r}")


def _measurement_row(
    *,
    coverage: Mapping[str, Any],
    product: Mapping[str, Any],
    science_path: Path,
    band: str,
    result: ApertureMeasurement,
    worker_id: str,
) -> dict[str, Any]:
    source, catalog_release = _source_for_band(band)
    flags = [
        "image_forced_photometry",
        "pending_validation",
        "aperture_correction_not_applied",
        "local_mad_background",
    ]
    if result.is_upper_limit:
        flags.append("three_sigma_upper_limit")
    provenance = {
        "measurement_version": IMAGE_MEASUREMENT_VERSION,
        "worker_id": worker_id,
        "coverage_id": coverage.get("coverage_id"),
        "product_id": product.get("product_id"),
        "science_path": str(science_path),
        "aperture_radius_arcsec": result.aperture_radius_arcsec,
        "annulus_inner_arcsec": result.annulus_inner_arcsec,
        "annulus_outer_arcsec": result.annulus_outer_arcsec,
        "beam_fwhm_arcsec": result.beam_fwhm_arcsec,
        "n_aperture_pixels": result.n_aperture_pixels,
        "n_annulus_pixels": result.n_annulus_pixels,
        "background_native_per_pixel": result.background_native_per_pixel,
        "noise_native_per_pixel": result.noise_native_per_pixel,
        "native_image_unit": result.native_unit,
        "conversion_jy_per_native_pixel": result.conversion_jy_per_native_pixel,
        "requires_instrument_aperture_correction_review": True,
    }
    row = _direct_flux_sed_row(
        candidate_id=str(coverage["candidate_id"]),
        payload={},
        source=source,
        band=band,
        flux_jy=result.flux_jy,
        flux_error_jy=result.flux_error_jy,
        separation_arcsec=0.0,
        catalog_release=f"{catalog_release}:{IMAGE_MEASUREMENT_VERSION}",
        source_object_id=str(product.get("product_id")),
        observation_id=str(coverage.get("observation_id") or ""),
        instrument=str(coverage.get("instrument") or ""),
        quality_flags=flags,
        provenance=provenance,
    )
    if row is None:
        raise ValueError("Image measurement could not be converted to a canonical SED row")
    row["is_upper_limit"] = int(result.is_upper_limit)
    row["quality_status"] = "pending_validation"
    row["fit_policy"] = "diagnostic_only"
    row["normalization_version"] = IMAGE_MEASUREMENT_VERSION
    measurements, _normalizations = prepare_canonical_sed_rows(
        [row],
        normalization_version=IMAGE_MEASUREMENT_VERSION,
        ingestion_version=IMAGE_MEASUREMENT_VERSION,
    )
    row["measurement_id"] = measurements[0]["measurement_id"]
    return row


def _download_job_products(
    conn: Any,
    products: pd.DataFrame,
    *,
    cache_dir: Path,
) -> pd.DataFrame:
    updated: list[dict[str, Any]] = []
    for _, product in products.iterrows():
        local_path = str(product.get("local_path") or "").strip()
        if local_path and Path(local_path).expanduser().exists():
            updated.append(product.to_dict())
            continue
        downloaded = download_archive_product(product, cache_dir=cache_dir)
        upsert_sed_archive_products(conn, downloaded)
        updated.append(downloaded)
    return pd.DataFrame(updated)


def _expanded_product_paths(products: pd.DataFrame) -> list[tuple[dict[str, Any], Path, str]]:
    expanded: list[tuple[dict[str, Any], Path, str]] = []
    for _, product in products.iterrows():
        local = str(product.get("local_path") or "").strip()
        if not local:
            continue
        for path in _fits_paths(Path(local).expanduser()):
            declared = str(product.get("product_type") or "")
            product_type = (
                declared
                if declared in {"science_image", "uncertainty_map", "coverage_map", "mask"}
                else _classify_fits_product(path)
            )
            expanded.append((product.to_dict(), path, product_type))
    return expanded


def _measure_job(
    coverage: Mapping[str, Any],
    products: pd.DataFrame,
    *,
    worker_id: str,
) -> list[dict[str, Any]]:
    expanded = _expanded_product_paths(products)
    coverage_paths = [path for _row, path, kind in expanded if kind == "coverage_map"]
    science = [(row, path) for row, path, kind in expanded if kind == "science_image"]
    if not science:
        return []
    ra = float(coverage["target_ra_deg"])
    dec = float(coverage["target_dec_deg"])
    output: list[dict[str, Any]] = []
    completed_bands: set[str] = set()
    for product, path in science:
        try:
            _image, _wcs, header = _image_plane(path)
            band = _band_from_header_or_path(
                header,
                path,
                fallback=str(coverage.get("band") or "").strip() or None,
            )
            if band is None or band in completed_bands:
                continue
            result = measure_fits_aperture(
                path,
                ra_deg=ra,
                dec_deg=dec,
                band=band,
                coverage_path=coverage_paths[0] if coverage_paths else None,
            )
            output.append(
                _measurement_row(
                    coverage=coverage,
                    product=product,
                    science_path=path,
                    band=band,
                    result=result,
                    worker_id=worker_id,
                )
            )
            completed_bands.add(band)
        except (NoUsableCoverageError, UnsupportedImageUnitError, ValueError):
            continue
    return output


def run(args: argparse.Namespace) -> Path:
    review_db = args.review_db.expanduser()
    cache_dir = args.cache_dir.expanduser()
    statuses = ["queued", "retry"]
    if bool(args.include_reduction_required):
        statuses.append("reduction_required")
    with closing(db_connect(review_db)) as conn:
        ensure_sed_storage_schema(conn)
        conn.commit()
        jobs = load_sed_image_jobs(
            conn,
            statuses=statuses,
            candidate_ids=args.candidate_id,
            limit=max(int(args.max_jobs), 0),
        )
        coverage = load_sed_archive_coverage(conn)
        products = load_sed_archive_products(conn)
        coverage_by_id = {
            str(row["coverage_id"]): row.to_dict()
            for _, row in coverage.iterrows()
        }
        print(f"Loaded {len(jobs)} resumable SED image jobs")
        for index, (_, job) in enumerate(jobs.iterrows(), start=1):
            job_id = str(job["job_id"])
            coverage_id = str(job["coverage_id"])
            item = coverage_by_id.get(coverage_id)
            if item is None:
                update_sed_image_job(
                    conn,
                    job_id,
                    status="failed",
                    last_error=f"missing coverage row {coverage_id}",
                    increment_attempt=True,
                )
                continue
            if str(job.get("job_type")) == "apex_bolometer_classify_reduce":
                if bool(args.include_reduction_required):
                    job_products = products[
                        products["coverage_id"].astype(str) == coverage_id
                    ].copy()
                    try:
                        _download_job_products(conn, job_products, cache_dir=cache_dir)
                    except Exception as exc:
                        update_sed_image_job(
                            conn,
                            job_id,
                            status="reduction_required",
                            last_error=f"{type(exc).__name__}: {exc}",
                            increment_attempt=True,
                        )
                        continue
                update_sed_image_job(
                    conn,
                    job_id,
                    status="reduction_required",
                    last_error=(
                        "APEXBOL product requires LABOCA/SABOCA classification and "
                        "instrument-specific reduction"
                    ),
                )
                continue

            print(
                f"[SED image] {index}/{len(jobs)} "
                f"{job.get('source_key')} {job.get('band') or ''} {job_id}"
            )
            update_sed_image_job(
                conn,
                job_id,
                status="running",
                lease_owner=str(args.worker_id),
                increment_attempt=True,
            )
            try:
                job_products = products[
                    products["coverage_id"].astype(str) == coverage_id
                ].copy()
                if job_products.empty:
                    raise FileNotFoundError("coverage has no product ledger rows")
                downloaded = _download_job_products(
                    conn,
                    job_products,
                    cache_dir=cache_dir,
                )
                if bool(args.download_only):
                    update_sed_image_job(
                        conn,
                        job_id,
                        status="downloaded",
                        last_error=None,
                    )
                    continue
                measurement_rows = _measure_job(
                    item,
                    downloaded,
                    worker_id=str(args.worker_id),
                )
                if not measurement_rows:
                    update_sed_image_job(
                        conn,
                        job_id,
                        status="manual_review",
                        last_error=(
                            "products downloaded, but no supported covered image "
                            "yielded a calibrated provisional measurement"
                        ),
                    )
                    continue
                upsert_sed_rows(conn, pd.DataFrame(measurement_rows))
                measurement_ids = [str(row["measurement_id"]) for row in measurement_rows]
                update_sed_image_job(
                    conn,
                    job_id,
                    status="pending_validation",
                    output_measurement_id=measurement_ids[0],
                    last_error=None,
                )
                print(
                    f"  wrote {len(measurement_rows)} provisional point(s): "
                    + ", ".join(measurement_ids)
                )
            except Exception as exc:
                attempts = int(job.get("attempt_count") or 0) + 1
                max_attempts = int(job.get("max_attempts") or 3)
                update_sed_image_job(
                    conn,
                    job_id,
                    status="failed" if attempts >= max_attempts else "retry",
                    last_error=f"{type(exc).__name__}: {exc}",
                )
    return review_db


def main(argv: list[str] | None = None) -> None:
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
