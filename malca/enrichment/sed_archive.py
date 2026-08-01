"""Archive-product discovery and download support for SED enrichment.

Catalog photometry and image products answer different questions.  The
catalog adapters in :mod:`malca.review.sed` produce provisional measurements;
this module records whether an archive observed the target, which products are
available, and which image-level operation still needs to be performed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import socket
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from typing import Callable

import numpy as np
import pandas as pd

from malca.review.sed import (
    _archive_query_position,
    _candidate_id_for_row,
    _clean_text,
    _row_value,
)
from malca.review.sed_storage import (
    SED_ARCHIVE_COVERAGE_COLUMNS,
    SED_ARCHIVE_PRODUCT_COLUMNS,
    SED_IMAGE_JOB_COLUMNS,
    make_sed_archive_coverage_id,
    make_sed_archive_product_id,
    make_sed_image_job_id,
    stable_sed_hash,
)


ARCHIVE_DISCOVERY_VERSION = "sed-archive-discovery-v1"
ARCHIVE_QUERY_TIMEOUT_SECONDS = 60.0
SPITZER_SEIP_COLLECTION = "spitzer_seip"
APEX_DISCOVERY_RADIUS_ARCSEC = 40.0
ARCHIVE_DISCOVERY_SOURCE_KEYS = frozenset(
    {"spitzer", "herschel", "apex_laboca", "apex_saboca", "apex_bolometer"}
)
ProgressCallback = Callable[[str], None]
ArchiveCheckpointCallback = Callable[
    [
        str,
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ],
    None,
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _default_requests_timeout(session: object, timeout_seconds: float):
    """Add connect/read timeouts to clients that omit ``requests`` timeouts.

    PyVO's synchronous SIA/TAP query methods currently call the shared
    ``requests.Session`` without a timeout.  Archive discovery is synchronous,
    so temporarily wrapping that session is safe and prevents an unresponsive
    socket from holding an entire SED run indefinitely.
    """

    timeout = max(float(timeout_seconds), 0.0)
    original_request = getattr(session, "request", None)
    if timeout <= 0.0 or not callable(original_request):
        yield
        return

    connect_timeout = min(timeout, 15.0)

    def request_with_timeout(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", (connect_timeout, timeout))
        return original_request(*args, **kwargs)

    setattr(session, "request", request_with_timeout)
    try:
        yield
    finally:
        setattr(session, "request", original_request)


@contextmanager
def _default_socket_timeout(timeout_seconds: float):
    """Bound legacy ``http.client`` archive calls that expose no timeout."""

    timeout = max(float(timeout_seconds), 0.0)
    if timeout <= 0.0:
        yield
        return
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


def resolve_archive_discovery_source_keys(sources: Iterable[str]) -> tuple[str, ...]:
    """Resolve catalog-facing source aliases to archive-ledger source keys."""

    requested = {str(value).strip().lower() for value in sources}
    resolved: list[str] = []
    if "spitzer" in requested:
        resolved.append("spitzer")
    if "herschel" in requested:
        resolved.append("herschel")
    if requested & {"apex_laboca", "apex_saboca", "apex_bolometer"}:
        resolved.append("apex_bolometer")
    return tuple(resolved)


def _emit_checkpoint(
    callback: ArchiveCheckpointCallback | None,
    source_key: str,
    coverage_rows: list[dict[str, Any]],
    product_rows: list[dict[str, Any]],
    job_rows: list[dict[str, Any]],
    starts: tuple[int, int, int],
) -> None:
    if callback is None:
        return
    coverage_start, product_start, job_start = starts
    callback(
        source_key,
        coverage_rows[coverage_start:],
        product_rows[product_start:],
        job_rows[job_start:],
    )


def _present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _value(row: Mapping[str, Any] | pd.Series, *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if _present(value):
            return value
        folded = str(name).casefold()
        for column in row.keys():
            if str(column).casefold() == folded:
                value = row.get(column)
                if _present(value):
                    return value
    return None


def _float_value(row: Mapping[str, Any] | pd.Series, *names: str) -> float | None:
    value = _value(row, *names)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _frame(value: object) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    if hasattr(value, "to_table"):
        table = value.to_table()
        return table.to_pandas() if hasattr(table, "to_pandas") else pd.DataFrame(table)
    return pd.DataFrame(value)


def _json_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not _present(value):
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _discovery_signature(
    *,
    source_key: str,
    adapter: str,
    target_ra_deg: float | None,
    target_dec_deg: float | None,
    policy: Mapping[str, Any],
) -> str:
    return stable_sed_hash(
        {
            "version": ARCHIVE_DISCOVERY_VERSION,
            "source_key": source_key,
            "adapter": adapter,
            "target_ra_deg": target_ra_deg,
            "target_dec_deg": target_dec_deg,
            "policy": dict(policy),
        }
    )


def _coverage_row(
    *,
    candidate_id: str,
    source_key: str,
    archive: str,
    collection: str | None,
    instrument: str | None,
    band: str | None,
    observation_id: str | None,
    coverage_status: str,
    target_ra_deg: float | None,
    target_dec_deg: float | None,
    coordinate_epoch_jyear: float | None,
    coordinate_method: str,
    discovery_signature: str,
    product_count: int = 0,
    observation_start_mjd: float | None = None,
    observation_end_mjd: float | None = None,
    exposure_seconds: float | None = None,
    coverage_fraction: float | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    row: dict[str, Any] = {
        "candidate_id": str(candidate_id),
        "source_key": source_key,
        "archive": archive,
        "collection": collection,
        "instrument": instrument,
        "band": band,
        "observation_id": observation_id,
        "coverage_status": coverage_status,
        "target_ra_deg": target_ra_deg,
        "target_dec_deg": target_dec_deg,
        "coordinate_epoch_jyear": coordinate_epoch_jyear,
        "coordinate_method": coordinate_method,
        "observation_start_mjd": observation_start_mjd,
        "observation_end_mjd": observation_end_mjd,
        "exposure_seconds": exposure_seconds,
        "coverage_fraction": coverage_fraction,
        "product_count": int(product_count),
        "discovery_signature": discovery_signature,
        "provenance_json": dict(provenance or {}),
        "discovered_at": now,
        "updated_at": now,
    }
    row["coverage_id"] = make_sed_archive_coverage_id(row)
    return row


def _product_row(
    coverage: Mapping[str, Any],
    *,
    product_type: str,
    processing_level: str | None,
    access_url: str | None,
    access_format: str | None,
    product_status: str = "discovered",
    provenance: Mapping[str, Any] | None = None,
    size_bytes: int | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    row: dict[str, Any] = {
        "coverage_id": coverage["coverage_id"],
        "candidate_id": coverage["candidate_id"],
        "source_key": coverage["source_key"],
        "archive": coverage["archive"],
        "collection": coverage.get("collection"),
        "observation_id": coverage.get("observation_id"),
        "instrument": coverage.get("instrument"),
        "band": coverage.get("band"),
        "product_type": str(product_type),
        "processing_level": processing_level,
        "access_url": access_url,
        "access_format": access_format,
        "local_path": None,
        "content_hash": None,
        "size_bytes": size_bytes,
        "product_status": product_status,
        "provenance_json": dict(provenance or {}),
        "discovered_at": now,
        "downloaded_at": None,
        "updated_at": now,
    }
    row["product_id"] = make_sed_archive_product_id(row)
    return row


def _job_row(
    coverage: Mapping[str, Any],
    *,
    job_type: str,
    job_status: str = "queued",
    priority: int = 100,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    row: dict[str, Any] = {
        "candidate_id": coverage["candidate_id"],
        "coverage_id": coverage["coverage_id"],
        "source_key": coverage["source_key"],
        "archive": coverage["archive"],
        "instrument": coverage.get("instrument"),
        "band": coverage.get("band"),
        "job_type": job_type,
        "job_status": job_status,
        "priority": int(priority),
        "attempt_count": 0,
        "max_attempts": 3,
        "lease_owner": None,
        "lease_expires_at": None,
        "last_error": None,
        "output_measurement_id": None,
        "provenance_json": dict(provenance or {}),
        "created_at": now,
        "updated_at": now,
    }
    row["job_id"] = make_sed_image_job_id(row)
    return row


def _spitzer_band(value: object) -> str | None:
    text = _clean_text(value).upper().replace("_", "").replace(" ", "")
    aliases = {
        "IRAC1": "IRAC1",
        "IRAC3.6": "IRAC1",
        "3.6UM": "IRAC1",
        "IRAC2": "IRAC2",
        "IRAC4.5": "IRAC2",
        "4.5UM": "IRAC2",
        "IRAC3": "IRAC3",
        "IRAC5.8": "IRAC3",
        "5.8UM": "IRAC3",
        "IRAC4": "IRAC4",
        "IRAC8.0": "IRAC4",
        "8.0UM": "IRAC4",
        "MIPS24": "MIPS24",
        "MIPS1": "MIPS24",
        "24UM": "MIPS24",
    }
    if text in aliases:
        return aliases[text]
    for token, band in (
        ("IRAC1", "IRAC1"),
        ("IRAC2", "IRAC2"),
        ("IRAC3", "IRAC3"),
        ("IRAC4", "IRAC4"),
        ("MIPS24", "MIPS24"),
        ("24", "MIPS24"),
    ):
        if token in text:
            return band
    return None


def _seip_product_type(row: Mapping[str, Any] | pd.Series) -> str:
    subtype = _clean_text(_value(row, "dataproduct_subtype"))
    url = _clean_text(_value(row, "access_url")).lower()
    text = f"{subtype} {url}".lower()
    if any(token in text for token in ("unc", "sigma", "std", "noise")):
        return "uncertainty_map"
    if any(token in text for token in ("cov", "coverage")):
        return "coverage_map"
    if any(token in text for token in ("mask", "flag")):
        return "mask"
    return "science_image"


def discover_spitzer_seip(
    candidates: pd.DataFrame,
    *,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: ArchiveCheckpointCallback | None = None,
    query_timeout_seconds: float = ARCHIVE_QUERY_TIMEOUT_SECONDS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Discover SEIP mosaics that cover each propagated Gaia position."""

    from astroquery.ipac.irsa import Irsa

    coverage_rows: list[dict[str, Any]] = []
    product_rows: list[dict[str, Any]] = []
    job_rows: list[dict[str, Any]] = []
    total = len(candidates)
    policy = {"collection": SPITZER_SEIP_COLLECTION, "radius_arcsec": 1.0, "maxrec": 500}
    for index, (_, candidate) in enumerate(candidates.iterrows(), start=1):
        starts = (len(coverage_rows), len(product_rows), len(job_rows))
        cid = _candidate_id_for_row(candidate)
        if progress_callback and (index == 1 or index % 50 == 0 or index == total):
            progress_callback(f"[SED archive] Spitzer SEIP {index}/{total}")
        ra, dec, coordinate_method = _archive_query_position(candidate, epoch_jyear=2008.0)
        signature = _discovery_signature(
            source_key="spitzer",
            adapter="irsa-sia2-spitzer-seip-v1",
            target_ra_deg=ra,
            target_dec_deg=dec,
            policy=policy,
        )
        if ra is None or dec is None:
            coverage_rows.append(
                _coverage_row(
                    candidate_id=cid,
                    source_key="spitzer",
                    archive="IRSA",
                    collection=SPITZER_SEIP_COLLECTION,
                    instrument=None,
                    band=None,
                    observation_id=None,
                    coverage_status="query_error",
                    target_ra_deg=ra,
                    target_dec_deg=dec,
                    coordinate_epoch_jyear=2008.0,
                    coordinate_method=coordinate_method,
                    discovery_signature=signature,
                    provenance={"reason": "missing_coordinates"},
                )
            )
            _emit_checkpoint(
                checkpoint_callback,
                "spitzer",
                coverage_rows,
                product_rows,
                job_rows,
                starts,
            )
            continue
        try:
            with _default_requests_timeout(Irsa._session, query_timeout_seconds):
                result = _frame(
                    Irsa.query_sia(
                        pos=(float(ra), float(dec), 1.0 / 3600.0),
                        collection=SPITZER_SEIP_COLLECTION,
                        data_type="image",
                        maxrec=500,
                    )
                )
        except Exception as exc:
            coverage_rows.append(
                _coverage_row(
                    candidate_id=cid,
                    source_key="spitzer",
                    archive="IRSA",
                    collection=SPITZER_SEIP_COLLECTION,
                    instrument=None,
                    band=None,
                    observation_id=None,
                    coverage_status="query_error",
                    target_ra_deg=ra,
                    target_dec_deg=dec,
                    coordinate_epoch_jyear=2008.0,
                    coordinate_method=coordinate_method,
                    discovery_signature=signature,
                    provenance={"error": f"{type(exc).__name__}: {exc}"},
                )
            )
            _emit_checkpoint(
                checkpoint_callback,
                "spitzer",
                coverage_rows,
                product_rows,
                job_rows,
                starts,
            )
            continue
        if result.empty:
            coverage_rows.append(
                _coverage_row(
                    candidate_id=cid,
                    source_key="spitzer",
                    archive="IRSA",
                    collection=SPITZER_SEIP_COLLECTION,
                    instrument=None,
                    band=None,
                    observation_id=None,
                    coverage_status="not_observed",
                    target_ra_deg=ra,
                    target_dec_deg=dec,
                    coordinate_epoch_jyear=2008.0,
                    coordinate_method=coordinate_method,
                    discovery_signature=signature,
                )
            )
            _emit_checkpoint(
                checkpoint_callback,
                "spitzer",
                coverage_rows,
                product_rows,
                job_rows,
                starts,
            )
            continue

        grouped: dict[tuple[str | None, str, str], list[pd.Series]] = {}
        for _, product in result.iterrows():
            band = _spitzer_band(_value(product, "energy_bandpassname", "instrument_name"))
            obs_id = _clean_text(_value(product, "obs_id", "obs_publisher_did")) or "unknown"
            instrument = _clean_text(_value(product, "instrument_name")) or (
                "MIPS" if band == "MIPS24" else "IRAC"
            )
            grouped.setdefault((band, obs_id, instrument), []).append(product)

        for (band, obs_id, instrument), group in grouped.items():
            coverage = _coverage_row(
                candidate_id=cid,
                source_key="spitzer",
                archive="IRSA",
                collection=SPITZER_SEIP_COLLECTION,
                instrument=instrument,
                band=band,
                observation_id=obs_id,
                coverage_status="covered_product",
                target_ra_deg=ra,
                target_dec_deg=dec,
                coordinate_epoch_jyear=2008.0,
                coordinate_method=coordinate_method,
                discovery_signature=signature,
                product_count=len(group),
                observation_start_mjd=min(
                    (value for value in (_float_value(item, "t_min") for item in group) if value is not None),
                    default=None,
                ),
                observation_end_mjd=max(
                    (value for value in (_float_value(item, "t_max") for item in group) if value is not None),
                    default=None,
                ),
                exposure_seconds=max(
                    (value for value in (_float_value(item, "t_exptime") for item in group) if value is not None),
                    default=None,
                ),
                provenance={
                    "query_collection": SPITZER_SEIP_COLLECTION,
                    "bandpass_name": _clean_text(_value(group[0], "energy_bandpassname")),
                },
            )
            coverage_rows.append(coverage)
            for product in group:
                access_url = _clean_text(_value(product, "access_url")) or None
                size_kib = _float_value(product, "access_estsize")
                product_rows.append(
                    _product_row(
                        coverage,
                        product_type=_seip_product_type(product),
                        processing_level=_clean_text(_value(product, "calib_level")) or None,
                        access_url=access_url,
                        access_format=_clean_text(_value(product, "access_format")) or None,
                        size_bytes=int(size_kib * 1024) if size_kib is not None else None,
                        provenance={
                            "obs_publisher_did": _clean_text(_value(product, "obs_publisher_did"))
                            or None,
                            "dataproduct_subtype": _clean_text(
                                _value(product, "dataproduct_subtype")
                            )
                            or None,
                            "s_region": _clean_text(_value(product, "s_region")) or None,
                            "spatial_resolution_arcsec": _float_value(product, "s_resolution"),
                        },
                    )
                )
            if band is not None:
                job_rows.append(
                    _job_row(
                        coverage,
                        job_type="spitzer_seip_forced_photometry",
                        priority=40,
                        provenance={
                            "required_products": [
                                "science_image",
                                "uncertainty_map",
                                "coverage_map",
                            ],
                            "result_requires_validation": True,
                        },
                    )
                )
        _emit_checkpoint(
            checkpoint_callback,
            "spitzer",
            coverage_rows,
            product_rows,
            job_rows,
            starts,
        )
    return coverage_rows, product_rows, job_rows


def _herschel_instrument(row: Mapping[str, Any] | pd.Series) -> str | None:
    text = _clean_text(_value(row, "instrument", "instrument_name")).upper()
    for instrument in ("PACS", "SPIRE"):
        if instrument in text:
            return instrument
    instrument_oid = _float_value(row, "instrument_oid")
    if instrument_oid is not None:
        return {1: "PACS", 2: "SPIRE"}.get(int(round(instrument_oid)))
    return text or None


def discover_herschel_hsa(
    candidates: pd.DataFrame,
    *,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: ArchiveCheckpointCallback | None = None,
    query_timeout_seconds: float = ARCHIVE_QUERY_TIMEOUT_SECONDS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Discover HSA observation bundles, leaving footprints for map validation."""

    from astroquery.esa.hsa import HSA
    coverage_rows: list[dict[str, Any]] = []
    product_rows: list[dict[str, Any]] = []
    job_rows: list[dict[str, Any]] = []
    total = len(candidates)
    policy = {
        "table": "hsa.v_active_observation",
        "spatial_predicate": "target_point_inside_polygon_fov",
        "instrument_oids": [1, 2],
        "max_records": 500,
    }
    for index, (_, candidate) in enumerate(candidates.iterrows(), start=1):
        starts = (len(coverage_rows), len(product_rows), len(job_rows))
        cid = _candidate_id_for_row(candidate)
        if progress_callback and (index == 1 or index % 50 == 0 or index == total):
            progress_callback(f"[SED archive] Herschel HSA {index}/{total}")
        ra, dec, coordinate_method = _archive_query_position(candidate, epoch_jyear=2011.0)
        signature = _discovery_signature(
            source_key="herschel",
            adapter="hsa-region-observation-v1",
            target_ra_deg=ra,
            target_dec_deg=dec,
            policy=policy,
        )
        if ra is None or dec is None:
            result = pd.DataFrame()
            error = "missing_coordinates"
        else:
            try:
                query = (
                    "SELECT TOP 500 observation_id, observation_oid, instrument_oid, "
                    "ra, dec, polygon_fov, start_time, end_time, duration, "
                    "target_name, proposal_id, image_location, image_2_5_location "
                    "FROM hsa.v_active_observation "
                    "WHERE instrument_oid IN (1, 2) AND 1=CONTAINS("
                    f"POINT('ICRS', {float(ra):.10f}, {float(dec):.10f}), polygon_fov)"
                )
                with _default_socket_timeout(query_timeout_seconds):
                    result = _frame(HSA.query_hsa_tap(query))
                error = None
            except Exception as exc:
                result = pd.DataFrame()
                error = f"{type(exc).__name__}: {exc}"
        if result.empty:
            coverage_rows.append(
                _coverage_row(
                    candidate_id=cid,
                    source_key="herschel",
                    archive="HSA",
                    collection="Herschel",
                    instrument=None,
                    band=None,
                    observation_id=None,
                    coverage_status="query_error" if error else "not_observed",
                    target_ra_deg=ra,
                    target_dec_deg=dec,
                    coordinate_epoch_jyear=2011.0,
                    coordinate_method=coordinate_method,
                    discovery_signature=signature,
                    provenance={"error": error} if error else {},
                )
            )
            _emit_checkpoint(
                checkpoint_callback,
                "herschel",
                coverage_rows,
                product_rows,
                job_rows,
                starts,
            )
            continue
        seen: set[tuple[str, str | None]] = set()
        for _, observation in result.iterrows():
            obs_id = _clean_text(
                _value(observation, "observation_id", "obsid", "observationid")
            )
            instrument = _herschel_instrument(observation)
            if not obs_id or not instrument or (obs_id, instrument) in seen:
                continue
            seen.add((obs_id, instrument))
            coverage = _coverage_row(
                candidate_id=cid,
                source_key="herschel",
                archive="HSA",
                collection="Herschel",
                instrument=instrument,
                band=None,
                observation_id=obs_id,
                coverage_status="covered_product",
                target_ra_deg=ra,
                target_dec_deg=dec,
                coordinate_epoch_jyear=2011.0,
                coordinate_method=coordinate_method,
                discovery_signature=signature,
                product_count=1,
                observation_start_mjd=_float_value(
                    observation, "start_time", "start_time_mjd", "t_min"
                ),
                observation_end_mjd=_float_value(
                    observation, "end_time", "end_time_mjd", "t_max"
                ),
                exposure_seconds=_float_value(
                    observation, "duration", "duration_seconds", "t_exptime"
                ),
                provenance={
                    "archive_polygon_contains_target": True,
                    "map_pixel_validation_required": True,
                    "observation": {
                        str(key): value
                        for key, value in observation.to_dict().items()
                        if str(key).casefold()
                        in {
                            "observation_id",
                            "obsid",
                            "instrument",
                            "instrument_name",
                            "target_name",
                            "proposal",
                            "proposal_id",
                            "polygon_fov",
                        }
                    },
                },
            )
            coverage_rows.append(coverage)
            product_rows.append(
                _product_row(
                    coverage,
                    product_type="observation_bundle",
                    processing_level="LEVEL2_5",
                    access_url=None,
                    access_format="application/x-tar",
                    provenance={
                        "retrieval": "astroquery.esa.hsa.HSA.get_observation",
                        "coverage_unconfirmed_until_map_opened": True,
                    },
                )
            )
            job_rows.append(
                _job_row(
                    coverage,
                    job_type="herschel_map_validate_photometry",
                    priority=60,
                    provenance={
                        "preferred_product_level": "LEVEL2_5",
                        "result_requires_validation": True,
                    },
                )
            )
        _emit_checkpoint(
            checkpoint_callback,
            "herschel",
            coverage_rows,
            product_rows,
            job_rows,
            starts,
        )
    return coverage_rows, product_rows, job_rows


def discover_apex_bolometer(
    candidates: pd.DataFrame,
    *,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: ArchiveCheckpointCallback | None = None,
    query_timeout_seconds: float = ARCHIVE_QUERY_TIMEOUT_SECONDS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Find APEXBOL observations and queue instrument-specific reduction.

    ESO raw metadata may identify only the umbrella ``APEXBOL`` instrument.
    These records are therefore discovered for all selected targets, but are
    intentionally not bulk aperture-photometered as if they were reduced maps.
    """

    import pyvo

    service = pyvo.dal.TAPService("https://archive.eso.org/tap_obs")
    coverage_rows: list[dict[str, Any]] = []
    product_rows: list[dict[str, Any]] = []
    job_rows: list[dict[str, Any]] = []
    total = len(candidates)
    policy = {
        "telescope": "APEX-12m",
        "instrument": "APEXBOL",
        "radius_arcsec": APEX_DISCOVERY_RADIUS_ARCSEC,
    }
    for index, (_, candidate) in enumerate(candidates.iterrows(), start=1):
        starts = (len(coverage_rows), len(product_rows), len(job_rows))
        cid = _candidate_id_for_row(candidate)
        if progress_callback and (index == 1 or index % 50 == 0 or index == total):
            progress_callback(f"[SED archive] APEX bolometer {index}/{total}")
        ra, dec, coordinate_method = _archive_query_position(candidate, epoch_jyear=2010.0)
        signature = _discovery_signature(
            source_key="apex_bolometer",
            adapter="eso-apexbol-obscore-v1",
            target_ra_deg=ra,
            target_dec_deg=dec,
            policy=policy,
        )
        result = pd.DataFrame()
        error: str | None = None
        if ra is None or dec is None:
            error = "missing_coordinates"
        else:
            radius_deg = APEX_DISCOVERY_RADIUS_ARCSEC / 3600.0
            query = (
                "SELECT TOP 100 dp_id, datalink_url, access_url, "
                "date_obs, mjd_obs, exposure, prog_id, prog_title, instrument, telescope "
                "FROM dbo.raw WHERE telescope = 'APEX-12m' AND instrument = 'APEXBOL' "
                "AND 1=INTERSECTS("
                "s_region, "
                f"CIRCLE('ICRS', {float(ra):.10f}, {float(dec):.10f}, {radius_deg:.10f})"
                ")"
            )
            try:
                with _default_requests_timeout(
                    getattr(service, "_session", None),
                    query_timeout_seconds,
                ):
                    result = _frame(service.search(query))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if result.empty:
            coverage_rows.append(
                _coverage_row(
                    candidate_id=cid,
                    source_key="apex_bolometer",
                    archive="ESO",
                    collection="APEX raw",
                    instrument="APEXBOL",
                    band=None,
                    observation_id=None,
                    coverage_status="query_error" if error else "not_observed",
                    target_ra_deg=ra,
                    target_dec_deg=dec,
                    coordinate_epoch_jyear=2010.0,
                    coordinate_method=coordinate_method,
                    discovery_signature=signature,
                    provenance={"error": error} if error else {},
                )
            )
            _emit_checkpoint(
                checkpoint_callback,
                "apex_bolometer",
                coverage_rows,
                product_rows,
                job_rows,
                starts,
            )
            continue
        for _, product in result.iterrows():
            obs_id = _clean_text(_value(product, "dp_id"))
            if not obs_id:
                continue
            coverage = _coverage_row(
                candidate_id=cid,
                source_key="apex_bolometer",
                archive="ESO",
                collection="APEX raw",
                instrument="APEXBOL",
                band=None,
                observation_id=obs_id,
                coverage_status="reduction_required",
                target_ra_deg=ra,
                target_dec_deg=dec,
                coordinate_epoch_jyear=2010.0,
                coordinate_method=coordinate_method,
                discovery_signature=signature,
                product_count=1,
                observation_start_mjd=_float_value(product, "mjd_obs"),
                exposure_seconds=_float_value(product, "exposure"),
                provenance={
                    "program_id": _clean_text(_value(product, "prog_id")) or None,
                    "program_title": _clean_text(_value(product, "prog_title")) or None,
                    "instrument_classification_required": True,
                },
            )
            coverage_rows.append(coverage)
            access_url = (
                _clean_text(_value(product, "datalink_url", "access_url")) or None
            )
            product_rows.append(
                _product_row(
                    coverage,
                    product_type="raw_bolometer_observation",
                    processing_level="raw",
                    access_url=access_url,
                    access_format=None,
                    product_status="reduction_required",
                    provenance={
                        "datalink_url": _clean_text(_value(product, "datalink_url")) or None,
                        "ordinary_fits_aperture_photometry_forbidden": True,
                    },
                )
            )
            job_rows.append(
                _job_row(
                    coverage,
                    job_type="apex_bolometer_classify_reduce",
                    job_status="reduction_required",
                    priority=90,
                    provenance={
                        "classify_as": ["LABOCA", "SABOCA", "other"],
                        "prefer_released_phase3_product": True,
                        "target_by_target_reduction": True,
                    },
                )
            )
        _emit_checkpoint(
            checkpoint_callback,
            "apex_bolometer",
            coverage_rows,
            product_rows,
            job_rows,
            starts,
        )
    return coverage_rows, product_rows, job_rows


def discover_sed_archive_products(
    candidates: pd.DataFrame,
    *,
    sources: Iterable[str],
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: ArchiveCheckpointCallback | None = None,
    completed_candidate_ids_by_source: Mapping[str, set[str]] | None = None,
    query_timeout_seconds: float = ARCHIVE_QUERY_TIMEOUT_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Discover archive products for the selected archive-backed sources."""

    requested = resolve_archive_discovery_source_keys(sources)
    completed_by_source = completed_candidate_ids_by_source or {}
    coverage: list[dict[str, Any]] = []
    products: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    discoverers: list[
        tuple[
            str,
            Callable[
                ...,
                tuple[
                    list[dict[str, Any]],
                    list[dict[str, Any]],
                    list[dict[str, Any]],
                ],
            ],
        ]
    ] = []
    if "spitzer" in requested:
        discoverers.append(("spitzer", discover_spitzer_seip))
    if "herschel" in requested:
        discoverers.append(("herschel", discover_herschel_hsa))
    if "apex_bolometer" in requested:
        discoverers.append(("apex_bolometer", discover_apex_bolometer))
    for source_key, discover in discoverers:
        completed = {
            str(value) for value in completed_by_source.get(source_key, set())
        }
        if completed:
            pending_mask = [
                _candidate_id_for_row(candidate) not in completed
                for _, candidate in candidates.iterrows()
            ]
            pending_candidates = candidates.loc[pending_mask]
        else:
            pending_candidates = candidates
        if progress_callback and len(pending_candidates) != len(candidates):
            progress_callback(
                f"[SED archive] {source_key}: checkpoint has "
                f"{len(candidates) - len(pending_candidates)}/{len(candidates)} targets"
            )
        if pending_candidates.empty:
            continue
        discovered_coverage, discovered_products, discovered_jobs = discover(
            pending_candidates,
            progress_callback=progress_callback,
            checkpoint_callback=checkpoint_callback,
            query_timeout_seconds=query_timeout_seconds,
        )
        coverage.extend(discovered_coverage)
        products.extend(discovered_products)
        jobs.extend(discovered_jobs)
    return (
        pd.DataFrame(coverage, columns=SED_ARCHIVE_COVERAGE_COLUMNS),
        pd.DataFrame(products, columns=SED_ARCHIVE_PRODUCT_COLUMNS),
        pd.DataFrame(jobs, columns=SED_IMAGE_JOB_COLUMNS),
    )


def _safe_suffix(access_url: str | None, access_format: str | None) -> str:
    path = urlparse(str(access_url or "")).path
    lower = path.lower()
    for suffix in (".fits.gz", ".fit.gz", ".fits", ".fit", ".fz", ".tar.gz", ".tar"):
        if lower.endswith(suffix):
            return suffix
    if "fits" in str(access_format or "").lower():
        return ".fits"
    return ".bin"


def _safe_component(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")
    return token[:120] or "unknown"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_archive_product(
    product: Mapping[str, Any] | pd.Series,
    *,
    cache_dir: Path,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Download one product and return its updated ledger row.

    Existing files are reused only after a content hash is computed.  HSA
    observation bundles are retrieved with the archive client when no direct
    URL was exposed by discovery.
    """

    row = dict(product)
    archive = _clean_text(row.get("archive")).upper()
    candidate_dir = (
        Path(cache_dir).expanduser()
        / _safe_component(archive.lower())
        / _safe_component(row.get("candidate_id"))
    )
    candidate_dir.mkdir(parents=True, exist_ok=True)
    access_url = _clean_text(row.get("access_url")) or None
    local_path: Path
    if access_url:
        suffix = _safe_suffix(access_url, _clean_text(row.get("access_format")))
        local_path = candidate_dir / f"{_safe_component(row.get('product_id'))}{suffix}"
        if not local_path.exists() or local_path.stat().st_size <= 0:
            temporary = local_path.with_name(f".{local_path.name}.partial")
            request = Request(access_url, headers={"User-Agent": "MALCA-SED/1"})
            with urlopen(request, timeout=float(timeout_seconds)) as response, temporary.open(
                "wb"
            ) as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
            temporary.replace(local_path)
    elif archive == "HSA":
        from astroquery.esa.hsa import HSA

        retrieved = HSA.get_observation(
            observation_id=str(row.get("observation_id")),
            instrument_name=str(row.get("instrument")),
            product_level=str(row.get("processing_level") or "LEVEL2_5"),
            download_dir=str(candidate_dir),
            cache=True,
        )
        local_path = Path(str(retrieved)).expanduser()
        if not local_path.exists():
            candidates = sorted(candidate_dir.rglob("*"), key=lambda item: item.stat().st_mtime)
            files = [item for item in candidates if item.is_file()]
            if not files:
                raise FileNotFoundError("HSA retrieval returned no local product")
            local_path = files[-1]
    else:
        raise ValueError(
            f"Archive product {row.get('product_id')!r} has no downloadable access URL"
        )

    row["local_path"] = str(local_path)
    row["size_bytes"] = int(local_path.stat().st_size) if local_path.is_file() else None
    row["content_hash"] = _file_sha256(local_path) if local_path.is_file() else None
    row["product_status"] = "downloaded"
    row["downloaded_at"] = _utc_now()
    row["updated_at"] = row["downloaded_at"]
    provenance = _json_dict(row.get("provenance_json"))
    provenance["download_version"] = "sed-archive-download-v1"
    row["provenance_json"] = provenance
    return {column: row.get(column) for column in SED_ARCHIVE_PRODUCT_COLUMNS}
