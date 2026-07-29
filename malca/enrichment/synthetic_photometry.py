"""Bandpass response caching and synthetic-photometry primitives for MALCA."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping
from urllib.parse import urlencode

import numpy as np
import requests
from astropy import units as u
from astropy.io.votable import parse as parse_votable

from malca.config import DEFAULT_CACHE_DIR
from malca.enrichment.photometric_calibration import (
    CONTRACT_AB_FLAT_FNU_COUNT_RATIO,
    CONTRACT_RESPONSE_MATCHED_VEGA_ZERO_POINT,
    OBSERVABLE_AB_MAG,
    OBSERVABLE_QUOTED_FNU,
    OBSERVABLE_VEGA_MAG,
    PhotometricCalibration,
    ab_calibration,
    reference_fnu_jy,
    vega_zero_point_calibration,
)


SVO_FILTER_SERVICE_URL = "https://svo2.cab.inta-csic.es/svo/theory/fps/fps.php"
SED_BANDPASS_CACHE_DIR = DEFAULT_CACHE_DIR.expanduser() / "sed" / "bandpasses"
C_ANGSTROM_PER_S = 2.99792458e18
JY_CGS = 1.0e-23
_TRAPEZOID = getattr(np, "trapezoid", np.trapz)
CACHE_FORMAT_VERSION = 3
HASH_VALIDATED_CACHE_FORMAT_VERSION = 2
SYNTHETIC_PHOTOMETRY_VERSION = "bandpass-union-grid-v2-detector-pivot"
RESPONSE_MANIFEST_VERSION = "response-science-manifest-v3"
RESPONSE_AUDIT_MANIFEST_VERSION = "response-audit-manifest-v1"


class BandpassUnavailableError(RuntimeError):
    """Raised when an exact response curve cannot be loaded."""


@dataclass(frozen=True)
class FilterResponse:
    """One system-throughput curve plus an explicitly identified calibration view.

    ``wavelength_ref_angstrom`` is retained as a compatibility alias for the
    SVO PhotCal ``WavelengthRef`` value.  It is *not* a mission catalog's
    quoted-flux reference wavelength.  New callers should use
    ``svo_calibration_wavelength_ref_angstrom`` and obtain mission reference
    wavelengths from :class:`PhotometricCalibration` instead.
    """

    filter_id: str
    wavelength_angstrom: np.ndarray
    throughput: np.ndarray
    detector_type: str = "photon"
    mag_system: str = ""
    zero_point_jy: float | None = None
    wavelength_ref_angstrom: float | None = None
    svo_calibration_wavelength_ref_angstrom: float | None = None
    zero_point_contract: str = ""
    source_url: str = ""
    response_hash: str = ""
    retrieved_at_utc: str = ""
    cached_at_utc: str = ""
    upstream_query_id: str = ""
    upstream_query_json: str = ""
    refresh_provenance: str = ""
    cache_format_version: int = 0

    def __post_init__(self) -> None:
        wave = np.asarray(self.wavelength_angstrom, dtype=float)
        throughput = np.asarray(self.throughput, dtype=float)
        good = np.isfinite(wave) & np.isfinite(throughput) & (wave > 0) & (throughput >= 0)
        wave = wave[good]
        throughput = throughput[good]
        if wave.size < 2 or not np.any(throughput > 0):
            raise ValueError(f"Filter {self.filter_id!r} has no usable throughput samples.")
        order = np.argsort(wave)
        wave = wave[order]
        throughput = throughput[order]
        unique, unique_idx = np.unique(wave, return_index=True)
        object.__setattr__(self, "wavelength_angstrom", unique)
        object.__setattr__(self, "throughput", throughput[unique_idx])
        computed_hash = _response_hash(unique, throughput[unique_idx])
        if self.response_hash and not hmac.compare_digest(str(self.response_hash), computed_hash):
            raise ValueError(f"Filter {self.filter_id!r} response_hash does not match its arrays.")
        object.__setattr__(self, "response_hash", computed_hash)
        svo_reference = _safe_float(self.svo_calibration_wavelength_ref_angstrom)
        legacy_reference = _safe_float(self.wavelength_ref_angstrom)
        explicit_reference = svo_reference or legacy_reference
        object.__setattr__(self, "svo_calibration_wavelength_ref_angstrom", explicit_reference)
        # Preserve the old attribute as an exact compatibility alias; keeping
        # both names equal prevents two ambiguous values entering the cache.
        object.__setattr__(self, "wavelength_ref_angstrom", explicit_reference)
        contract = str(self.zero_point_contract or "").strip()
        if not contract and str(self.mag_system or "").strip().casefold() == "vega" and _safe_float(self.zero_point_jy):
            contract = CONTRACT_RESPONSE_MATCHED_VEGA_ZERO_POINT
        object.__setattr__(self, "zero_point_contract", contract)


ResponseLoader = Callable[[str, str], FilterResponse]


def _response_hash(wavelength: np.ndarray, throughput: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(wavelength, dtype="<f8").tobytes())
    digest.update(np.asarray(throughput, dtype="<f8").tobytes())
    return digest.hexdigest()


def _cache_stem(filter_id: str, mag_system: str = "") -> str:
    """Return the v2 curve-cache identity.

    ``mag_system`` remains an accepted argument for compatibility with older
    callers, but deliberately does not participate in the cache key.  AB,
    Vega, and quoted-Jy products that use the same physical filter must share
    one throughput artifact.
    """
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(filter_id)).strip("_")[:80]
    digest = hashlib.sha256(str(filter_id).encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"


def _legacy_cache_stem(filter_id: str, mag_system: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(filter_id)).strip("_")[:80]
    digest = hashlib.sha256(f"{filter_id}|{mag_system}".encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"


def _normalized_system(value: object) -> str:
    folded = str(value or "").strip().casefold()
    aliases = {"ab": "AB", "vega": "Vega", "st": "ST", "jy": "Jy"}
    return aliases.get(folded, str(value or "").strip())


def _metadata_hash(metadata: Mapping[str, object]) -> str:
    content = {key: value for key, value in metadata.items() if key != "metadata_hash"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _upstream_query_identity(
    filter_id: str,
    requested_mag_system: str,
    params: Mapping[str, object],
) -> tuple[str, str]:
    """Return stable JSON and a digest identifying the exact upstream query."""
    payload = {
        "service": "SVO Filter Profile Service",
        "service_url": SVO_FILTER_SERVICE_URL,
        "filter_id": str(filter_id),
        "requested_mag_system": _normalized_system(requested_mag_system),
        "parameters": {str(key): str(value) for key, value in sorted(params.items())},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    """Atomically replace one cache component in its destination directory."""
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _param_value(table: object, name: str) -> object | None:
    wanted = name.strip().lower()
    containers = [getattr(table, "params", None)]
    resource = getattr(table, "_resource", None)
    containers.append(getattr(resource, "params", None))
    for params in containers:
        for param in params or []:
            key = str(getattr(param, "name", "") or getattr(param, "ID", "")).strip().lower()
            if key == wanted:
                return getattr(param, "value", None)
    return None


def _parse_svo_response(
    payload: bytes,
    filter_id: str,
    mag_system: str,
    source_url: str,
    *,
    retrieved_at_utc: str = "",
    upstream_query_id: str = "",
    upstream_query_json: str = "",
    refresh_provenance: str = "",
) -> FilterResponse:
    try:
        votable = parse_votable(io.BytesIO(payload))
    except Exception as exc:
        raise BandpassUnavailableError(f"SVO returned an unreadable response for {filter_id}: {exc}") from exc

    for resource in votable.resources:
        query_status = next(
            (str(info.value) for info in resource.infos if str(getattr(info, "name", "")).upper() == "QUERY_STATUS"),
            "",
        )
        if query_status.upper() == "ERROR":
            raise BandpassUnavailableError(f"SVO rejected filter identifier {filter_id!r}.")
        for table in resource.tables:
            names = {str(field.name).strip().lower(): str(field.name) for field in table.fields}
            wave_name = names.get("wavelength")
            transmission_name = names.get("transmission") or names.get("trasmission")
            if not wave_name or not transmission_name or table.array is None:
                continue
            wave = np.asarray(table.array[wave_name], dtype=float)
            transmission = np.asarray(table.array[transmission_name], dtype=float)
            detector_raw = _param_value(table, "DetectorType")
            detector_text = str(detector_raw or "1").strip().lower()
            detector_type = "photon" if detector_text in {"1", "photon", "photon counter"} else "energy"
            zero_point = _safe_float(_param_value(table, "ZeroPoint"))
            wave_ref = _safe_float(_param_value(table, "WavelengthRef"))
            returned_mag_system = str(_param_value(table, "MagSys") or mag_system or "")
            zero_point_contract = (
                CONTRACT_RESPONSE_MATCHED_VEGA_ZERO_POINT
                if returned_mag_system.strip().casefold() == "vega" and zero_point is not None
                else ""
            )
            return FilterResponse(
                filter_id=filter_id,
                wavelength_angstrom=wave,
                throughput=transmission,
                detector_type=detector_type,
                mag_system=returned_mag_system,
                zero_point_jy=zero_point,
                wavelength_ref_angstrom=wave_ref,
                svo_calibration_wavelength_ref_angstrom=wave_ref,
                zero_point_contract=zero_point_contract,
                source_url=source_url,
                retrieved_at_utc=retrieved_at_utc,
                upstream_query_id=upstream_query_id,
                upstream_query_json=upstream_query_json,
                refresh_provenance=refresh_provenance,
                cache_format_version=CACHE_FORMAT_VERSION,
            )
    raise BandpassUnavailableError(f"SVO response for {filter_id!r} contained no transmission curve.")


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def save_filter_response(
    response: FilterResponse,
    cache_dir: str | Path | None = None,
    *,
    requested_mag_system: str | None = None,
    refresh_provenance: str | None = None,
) -> tuple[Path, Path]:
    """Atomically cache one physical response and its calibration metadata.

    Throughput is keyed only by ``filter_id``.  Repeated saves for different
    magnitude systems merge their calibration metadata into the same cache
    record.  ``requested_mag_system`` records aliases such as a native quoted
    Jy catalog registration even when an SVO ID query returns Vega metadata.
    """
    root = Path(cache_dir or SED_BANDPASS_CACHE_DIR).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    stem = _cache_stem(response.filter_id)
    data_path = root / f"{stem}.npz"
    meta_path = root / f"{stem}.json"

    calibrations: dict[str, dict[str, object]] = {}
    requested_systems: set[str] = set()
    previous: dict[str, object] = {}
    previous_response = _load_cached_pair(
        data_path,
        meta_path,
        filter_id=response.filter_id,
        requested_mag_system="",
    )
    if previous_response is not None and previous_response.response_hash == response.response_hash:
        try:
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
            calibrations.update(dict(previous.get("calibrations") or {}))
            requested_systems.update(str(item) for item in previous.get("requested_systems") or [])
        except Exception:
            # A corrupt previous entry is replaced below; it is never trusted.
            pass

    returned_system = _normalized_system(response.mag_system)
    if returned_system.casefold() in {"ab", "vega", "st"}:
        calibrations[returned_system.casefold()] = {
            "mag_system": returned_system,
            "zero_point_jy": response.zero_point_jy,
            "zero_point_contract": response.zero_point_contract,
            "wavelength_ref_angstrom": response.wavelength_ref_angstrom,
            "svo_calibration_wavelength_ref_angstrom": (
                response.svo_calibration_wavelength_ref_angstrom
            ),
            "source_url": response.source_url,
        }
    requested_system = _normalized_system(
        response.mag_system if requested_mag_system is None else requested_mag_system
    )
    if requested_system:
        requested_systems.add(requested_system)

    retrieved_at_utc = str(
        response.retrieved_at_utc or previous.get("retrieved_at_utc") or ""
    )
    upstream_query_id = str(
        response.upstream_query_id or previous.get("upstream_query_id") or ""
    )
    upstream_query_json = str(
        response.upstream_query_json or previous.get("upstream_query_json") or ""
    )
    resolved_refresh_provenance = str(
        refresh_provenance
        or response.refresh_provenance
        or previous.get("refresh_provenance")
        or "manual_cache_write"
    )
    cached_at_utc = _utc_now()

    generation = uuid.uuid4().hex
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        wavelength_angstrom=response.wavelength_angstrom,
        throughput=response.throughput,
        cache_generation=np.asarray(generation),
    )
    metadata: dict[str, object] = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "cache_generation": generation,
        "filter_id": response.filter_id,
        "detector_type": response.detector_type,
        "zero_point_contract": response.zero_point_contract,
        "wavelength_ref_angstrom": response.wavelength_ref_angstrom,
        "svo_calibration_wavelength_ref_angstrom": (
            response.svo_calibration_wavelength_ref_angstrom
        ),
        "source_url": response.source_url,
        "response_hash": response.response_hash,
        "retrieved_at_utc": retrieved_at_utc,
        "cached_at_utc": cached_at_utc,
        "upstream_query_id": upstream_query_id,
        "upstream_query_json": upstream_query_json,
        "refresh_provenance": resolved_refresh_provenance,
        "calibrations": calibrations,
        "requested_systems": sorted(requested_systems),
    }
    metadata["metadata_hash"] = _metadata_hash(metadata)

    # The metadata is the commit marker.  If a process stops after replacing
    # the NPZ but before replacing JSON, the generation mismatch makes the
    # incomplete pair unreadable rather than silently mixing versions.
    _atomic_write(data_path, buffer.getvalue())
    _atomic_write(meta_path, json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"))
    return data_path, meta_path


def load_cached_filter_response(
    filter_id: str,
    mag_system: str = "",
    cache_dir: str | Path | None = None,
) -> FilterResponse | None:
    root = Path(cache_dir or SED_BANDPASS_CACHE_DIR).expanduser()
    if not root.exists():
        return None

    # Try the calibration-independent v2 identity first, then the exact legacy
    # identity.  Finally scan legacy metadata by filter ID: this recovers the
    # old Jy bug where an ID request was saved under SVO's returned Vega key.
    primary_stems = [_cache_stem(filter_id), _legacy_cache_stem(filter_id, mag_system)]

    def usable_response(stem: str) -> FilterResponse | None:
        response = _load_cached_pair(
            root / f"{stem}.npz",
            root / f"{stem}.json",
            filter_id=str(filter_id),
            requested_mag_system=str(mag_system or ""),
        )
        if response is not None:
            if str(mag_system or "").strip().casefold() == "vega" and response.zero_point_jy is None:
                # A v2 curve may have been created from a native-Jy migration
                # while a usable Vega calibration still exists in a legacy
                # sibling.  Keep searching before declaring it absent.
                return None
            return response
        return None

    for stem in primary_stems:
        response = usable_response(stem)
        if response is not None:
            return response

    # The slow metadata scan is only needed for legacy aliases such as a Jy
    # registration whose SVO ID response was stored under a Vega-derived key.
    seen = set(primary_stems)
    for meta_path in sorted(root.glob("*.json")):
        if meta_path.stem in seen:
            continue
        try:
            legacy_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(legacy_metadata.get("filter_id") or "") != str(filter_id):
            continue
        response = usable_response(meta_path.stem)
        if response is not None:
            return response
    return None


def _load_cached_pair(
    data_path: Path,
    meta_path: Path,
    *,
    filter_id: str,
    requested_mag_system: str,
) -> FilterResponse | None:
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if str(metadata.get("filter_id") or filter_id) != filter_id:
            return None
        cache_version = int(metadata.get("cache_format_version") or 1)
        if cache_version >= HASH_VALIDATED_CACHE_FORMAT_VERSION:
            recorded_metadata_hash = str(metadata.get("metadata_hash") or "")
            if not recorded_metadata_hash or not hmac.compare_digest(
                recorded_metadata_hash,
                _metadata_hash(metadata),
            ):
                return None
        with np.load(data_path, allow_pickle=False) as arrays:
            wavelength = np.asarray(arrays["wavelength_angstrom"], dtype=float)
            throughput = np.asarray(arrays["throughput"], dtype=float)
            if cache_version >= HASH_VALIDATED_CACHE_FORMAT_VERSION:
                generation = str(np.asarray(arrays["cache_generation"]).item())
                if generation != str(metadata.get("cache_generation") or ""):
                    return None
        computed_hash = _response_hash(wavelength, throughput)
        recorded_hash = str(metadata.get("response_hash") or "")
        if recorded_hash and not hmac.compare_digest(recorded_hash, computed_hash):
            return None

        calibration = _cached_calibration(metadata, requested_mag_system)
        explicit_svo_reference = (
            _safe_float(calibration.get("svo_calibration_wavelength_ref_angstrom"))
            or _safe_float(metadata.get("svo_calibration_wavelength_ref_angstrom"))
            or _safe_float(calibration.get("wavelength_ref_angstrom"))
            or _safe_float(metadata.get("wavelength_ref_angstrom"))
        )
        provenance_default = (
            "legacy_cache_v1"
            if cache_version < HASH_VALIDATED_CACHE_FORMAT_VERSION
            else "pre_provenance_cache_v2"
            if cache_version < CACHE_FORMAT_VERSION
            else ""
        )
        return FilterResponse(
            filter_id=filter_id,
            wavelength_angstrom=wavelength,
            throughput=throughput,
            detector_type=str(metadata.get("detector_type") or "photon"),
            mag_system=str(calibration.get("mag_system") or ""),
            zero_point_jy=_safe_float(calibration.get("zero_point_jy")),
            wavelength_ref_angstrom=explicit_svo_reference,
            svo_calibration_wavelength_ref_angstrom=explicit_svo_reference,
            zero_point_contract=str(calibration.get("zero_point_contract") or ""),
            source_url=str(calibration.get("source_url") or metadata.get("source_url") or ""),
            response_hash=computed_hash,
            retrieved_at_utc=str(metadata.get("retrieved_at_utc") or ""),
            cached_at_utc=str(metadata.get("cached_at_utc") or ""),
            upstream_query_id=str(metadata.get("upstream_query_id") or ""),
            upstream_query_json=str(metadata.get("upstream_query_json") or ""),
            refresh_provenance=str(metadata.get("refresh_provenance") or provenance_default),
            cache_format_version=cache_version,
        )
    except Exception:
        return None


def _cached_calibration(metadata: Mapping[str, object], requested_mag_system: str) -> dict[str, object]:
    requested = _normalized_system(requested_mag_system)
    folded = requested.casefold()
    calibrations = dict(metadata.get("calibrations") or {})

    if folded == "ab":
        selected = dict(calibrations.get("ab") or {})
        selected.update({
            "mag_system": "AB",
            "zero_point_jy": 3631.0,
            "zero_point_contract": CONTRACT_AB_FLAT_FNU_COUNT_RATIO,
        })
        return selected
    if folded == "jy":
        # Jy is the catalog's native observable, not an SVO magnitude
        # calibration.  Throughput and reference wavelength are still useful.
        return {"mag_system": "Jy", "zero_point_jy": None, "zero_point_contract": ""}
    if folded and folded in calibrations:
        return dict(calibrations[folded])

    # Legacy v1 metadata stored exactly one returned calibration at top level.
    legacy_system = _normalized_system(metadata.get("mag_system"))
    legacy = {
        "mag_system": requested or legacy_system,
        "zero_point_jy": metadata.get("zero_point_jy"),
        "zero_point_contract": metadata.get("zero_point_contract"),
        "wavelength_ref_angstrom": metadata.get("wavelength_ref_angstrom"),
        "source_url": metadata.get("source_url"),
    }
    if folded == "vega" and legacy_system.casefold() != "vega":
        legacy["zero_point_jy"] = None
    if not requested and calibrations:
        return dict(next(iter(calibrations.values())))
    return legacy


def fetch_filter_response(
    filter_id: str,
    mag_system: str = "",
    *,
    cache_dir: str | Path | None = None,
    timeout: float = 30.0,
    session: requests.Session | None = None,
    force: bool = False,
) -> FilterResponse:
    """Load an SVO filter response, using a persistent cache when possible."""
    clean_id = str(filter_id or "").strip()
    clean_system = str(mag_system or "").strip()
    if not clean_id:
        raise BandpassUnavailableError("The SED row has no SVO filter identifier.")
    if not force:
        cached = load_cached_filter_response(clean_id, clean_system, cache_dir)
        calibration_ready = not (
            clean_system.casefold() == "vega" and cached is not None and cached.zero_point_jy is None
        )
        if cached is not None and calibration_ready:
            return cached

    params = {"VERB": "2"}
    if clean_system.lower() in {"ab", "vega", "st"}:
        params["PhotCalID"] = f"{clean_id}/{clean_system.upper() if clean_system.lower() != 'vega' else 'Vega'}"
    else:
        params["ID"] = clean_id
    upstream_query_json, upstream_query_id = _upstream_query_identity(
        clean_id,
        clean_system,
        params,
    )
    url = f"{SVO_FILTER_SERVICE_URL}?{urlencode(params)}"
    client = session or requests
    try:
        response = client.get(url, timeout=float(timeout))
        response.raise_for_status()
    except Exception as exc:
        raise BandpassUnavailableError(f"Could not retrieve {clean_id} from SVO: {exc}") from exc
    retrieved_at_utc = _utc_now()
    refresh_provenance = "forced_upstream_refresh" if force else "cache_miss_upstream_fetch"
    parsed = _parse_svo_response(
        response.content,
        clean_id,
        clean_system,
        url,
        retrieved_at_utc=retrieved_at_utc,
        upstream_query_id=upstream_query_id,
        upstream_query_json=upstream_query_json,
        refresh_provenance=refresh_provenance,
    )
    save_filter_response(
        parsed,
        cache_dir,
        requested_mag_system=clean_system,
        refresh_provenance=refresh_provenance,
    )
    # Return the same calibration view that a subsequent cache load will
    # provide.  In particular an ID request registered as native Jy must not
    # leak SVO's default Vega zero point into the catalog interpretation.
    cached = load_cached_filter_response(clean_id, clean_system, cache_dir)
    return cached if cached is not None else parsed


def build_response_map(
    filters: Iterable[tuple[str, str]],
    *,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
    progress_callback: Callable[[str], None] | None = None,
    response_loader: ResponseLoader | None = None,
) -> tuple[dict[tuple[str, str], FilterResponse], dict[tuple[str, str], str]]:
    """Resolve each unique ``(filter_id, magnitude system)`` exactly once."""
    unique = sorted({(str(fid or "").strip(), str(ms or "").strip()) for fid, ms in filters if str(fid or "").strip()})
    responses: dict[tuple[str, str], FilterResponse] = {}
    failures: dict[tuple[str, str], str] = {}
    for index, key in enumerate(unique, start=1):
        filter_id, mag_system = key
        if progress_callback and (index == 1 or index % 10 == 0 or index == len(unique)):
            progress_callback(f"[SED bandpass] loading {index}/{len(unique)}: {filter_id}")
        try:
            if response_loader is not None:
                response = response_loader(filter_id, mag_system)
            else:
                response = load_cached_filter_response(filter_id, mag_system, cache_dir)
                if response is None and allow_download:
                    response = fetch_filter_response(filter_id, mag_system, cache_dir=cache_dir)
                if response is None:
                    raise BandpassUnavailableError(f"No cached response for {filter_id}.")
            responses[key] = response
        except Exception as exc:
            failures[key] = str(exc)
    return responses, failures


def apply_extinction(
    wavelength_angstrom: np.ndarray,
    flux_lambda: np.ndarray,
    av: float,
    *,
    rv: float = 3.1,
    law: str = "G23",
) -> np.ndarray:
    """Apply a wavelength-dependent extinction curve to an intrinsic spectrum."""
    wave = np.asarray(wavelength_angstrom, dtype=float)
    flux = np.asarray(flux_lambda, dtype=float)
    av_value = max(float(av), 0.0)
    if av_value == 0:
        return flux.copy()
    if str(law).strip().upper() != "G23":
        raise ValueError(f"Unsupported extinction law: {law}")
    from dust_extinction.parameter_averages import G23

    extinction = G23(Rv=float(rv))
    factor = np.ones_like(wave, dtype=float)
    micron = wave * 1.0e-4
    valid = np.isfinite(micron) & (micron >= 1.0 / extinction.x_range[1]) & (micron <= 1.0 / extinction.x_range[0])
    if np.any(valid):
        axav = np.asarray(extinction(wave[valid] * u.AA), dtype=float)
        factor[valid] = np.power(10.0, -0.4 * av_value * axav)
    return flux * factor


def bandpass_flux_nu_jy(
    wavelength_angstrom: np.ndarray,
    flux_lambda: np.ndarray,
    response: FilterResponse,
) -> float:
    """Return the catalog-equivalent mean flux density for one response curve.

    The result is normalized by the response to a flat 1 Jy reference spectrum.
    For photon counters this is equivalent to taking the ratio of photon count
    rates; energy-integrating responses omit the photon wavelength weighting.
    """
    wave, flux, throughput = _integration_samples(wavelength_angstrom, flux_lambda, response)
    flat_one_jy_flam = JY_CGS * C_ANGSTROM_PER_S / np.square(wave)
    numerator = _bandpass_signal(wave, flux, throughput, response.detector_type)
    denominator = _bandpass_signal(wave, flat_one_jy_flam, throughput, response.detector_type)
    return _positive_signal_ratio(numerator, denominator, response.filter_id)


def response_matched_zero_point_jy(
    reference_wavelength_angstrom: np.ndarray,
    reference_flux_lambda: np.ndarray,
    response: FilterResponse,
) -> float:
    """Return ``C(reference) / C(flat 1 Jy)`` for one response.

    Supplying a pinned Vega reference spectrum produces the scalar required by
    :func:`vega_zero_point_calibration`.  Once computed, model magnitudes are
    exactly count-rate ratios because the shared flat-one-Jy denominator
    cancels algebraically.  This helper intentionally accepts an explicit
    spectrum and never invents a blackbody approximation to Vega.
    """
    return bandpass_flux_nu_jy(
        reference_wavelength_angstrom,
        reference_flux_lambda,
        response,
    )


def bandpass_quoted_flux_nu_jy(
    wavelength_angstrom: np.ndarray,
    flux_lambda: np.ndarray,
    response: FilterResponse,
    calibration: PhotometricCalibration,
) -> float:
    """Predict a mission's quoted monochromatic flux density in Jy.

    The reference spectrum is normalized to one Jy at the mission reference
    wavelength, then passed through exactly the same response integral as the
    model.  This implements color-convention-aware forward modeling rather
    than treating direct IR catalog fluxes as pseudo-AB measurements.
    """
    if calibration.observable_kind != OBSERVABLE_QUOTED_FNU:
        raise ValueError("bandpass_quoted_flux_nu_jy requires a quoted_fnu calibration.")
    wave, flux, throughput = _integration_samples(wavelength_angstrom, flux_lambda, response)
    reference_fnu = reference_fnu_jy(wave, calibration)
    reference_flam = reference_fnu * JY_CGS * C_ANGSTROM_PER_S / np.square(wave)
    numerator = _bandpass_signal(wave, flux, throughput, response.detector_type)
    denominator = _bandpass_signal(wave, reference_flam, throughput, response.detector_type)
    return _positive_signal_ratio(numerator, denominator, response.filter_id)


def predict_native_observable(
    wavelength_angstrom: np.ndarray,
    flux_lambda: np.ndarray,
    response: FilterResponse,
    calibration: PhotometricCalibration,
) -> float:
    """Predict one catalog-native AB/Vega magnitude or quoted Jy value."""
    if calibration.observable_kind == OBSERVABLE_QUOTED_FNU:
        return bandpass_quoted_flux_nu_jy(
            wavelength_angstrom,
            flux_lambda,
            response,
            calibration,
        )
    equivalent_fnu_jy = bandpass_flux_nu_jy(wavelength_angstrom, flux_lambda, response)
    if calibration.observable_kind in {OBSERVABLE_AB_MAG, OBSERVABLE_VEGA_MAG}:
        if (
            calibration.observable_kind == OBSERVABLE_VEGA_MAG
            and calibration.forward_contract != CONTRACT_RESPONSE_MATCHED_VEGA_ZERO_POINT
        ):
            raise ValueError(
                "Vega magnitude prediction requires a response-matched zero-point count-ratio contract."
            )
        zero_point_jy = float(calibration.zero_point_jy or 0.0)
        if zero_point_jy <= 0:
            raise ValueError("Magnitude prediction requires a positive zero point.")
        return float(-2.5 * math.log10(equivalent_fnu_jy / zero_point_jy))
    raise ValueError(f"Unsupported native observable: {calibration.observable_kind}")


def calibration_for_response(
    response: FilterResponse,
    mag_system: str | None = None,
) -> PhotometricCalibration:
    """Construct the backward-compatible AB/Vega calibration view."""
    system = _normalized_system(response.mag_system if mag_system is None else mag_system)
    if system.casefold() == "ab":
        return ab_calibration(calibration_id=f"{response.filter_id}/AB/3631Jy")
    if system.casefold() == "vega":
        if response.zero_point_jy is None or response.zero_point_jy <= 0:
            raise BandpassUnavailableError(
                f"{response.filter_id} has no response-matched Vega zero point."
            )
        return vega_zero_point_calibration(
            response.zero_point_jy,
            calibration_id=f"{response.filter_id}/Vega/SVO-zero-point",
        )
    raise ValueError(
        f"{system or 'blank'} is not an AB/Vega calibration; configure native quoted Jy explicitly."
    )


def response_pivot_wavelength_angstrom(response: FilterResponse) -> float:
    """Return the source-independent pivot wavelength of a response curve.

    The pivot wavelength is the appropriate response-derived x coordinate for
    broadband flux-density conversion and display.  It must not be confused
    with a mission's quoted-flux reference wavelength or a source-dependent
    isophotal wavelength.  Photon-counting responses carry the detector's
    additional wavelength weighting; energy-integrating responses do not, so
    their pivot definitions are necessarily different.
    """
    wave = response.wavelength_angstrom
    throughput = response.throughput
    if str(response.detector_type or "").lower().startswith("photon"):
        numerator = float(_TRAPEZOID(throughput * wave, wave))
        denominator = float(_TRAPEZOID(throughput / wave, wave))
    else:
        numerator = float(_TRAPEZOID(throughput, wave))
        denominator = float(_TRAPEZOID(throughput / np.square(wave), wave))
    if not (
        math.isfinite(numerator)
        and math.isfinite(denominator)
        and numerator > 0
        and denominator > 0
    ):
        raise BandpassUnavailableError(f"Pivot wavelength failed for {response.filter_id}.")
    return float(math.sqrt(numerator / denominator))


def svo_calibration_reference_wavelength_angstrom(response: FilterResponse) -> float | None:
    """Return SVO PhotCal ``WavelengthRef``, never a mission Jy reference.

    Mission-quoted flux reference wavelengths live on
    :class:`PhotometricCalibration`.  The explicit name prevents callers from
    accidentally using SVO's default Vega calibration wavelength when forward
    modeling IRAC, AKARI, IRAS, MIPS, or PACS quoted fluxes.
    """
    return _safe_float(response.svo_calibration_wavelength_ref_angstrom)


def _integration_samples(
    wavelength_angstrom: np.ndarray,
    flux_lambda: np.ndarray,
    response: FilterResponse,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample model and response on their union grid.

    The previous implementation sampled the model only at SVO response nodes,
    which can miss narrow model structure by several percent.  Retaining every
    model node inside the passband makes the numerical integral converge with
    the supplied model resolution while exactly preserving response vertices.
    """
    model_wave = np.asarray(wavelength_angstrom, dtype=float)
    model_flux = np.asarray(flux_lambda, dtype=float)
    if model_wave.shape != model_flux.shape:
        raise ValueError("Model wavelength and flux arrays must have the same shape.")
    good = np.isfinite(model_wave) & np.isfinite(model_flux) & (model_wave > 0)
    model_wave = model_wave[good]
    model_flux = model_flux[good]
    if model_wave.size < 2:
        raise BandpassUnavailableError("Model spectrum has fewer than two usable samples.")
    order = np.argsort(model_wave)
    model_wave = model_wave[order]
    model_flux = model_flux[order]
    model_wave, unique_index = np.unique(model_wave, return_index=True)
    model_flux = model_flux[unique_index]

    response_wave = response.wavelength_angstrom
    if response_wave[0] < model_wave[0] or response_wave[-1] > model_wave[-1]:
        raise BandpassUnavailableError(
            f"Model spectrum does not fully cover {response.filter_id} "
            f"({response_wave[0]:.0f}-{response_wave[-1]:.0f} A)."
        )
    inside = (model_wave >= response_wave[0]) & (model_wave <= response_wave[-1])
    integration_wave = np.union1d(response_wave, model_wave[inside])
    sampled_flux = np.interp(integration_wave, model_wave, model_flux)
    sampled_throughput = np.interp(
        integration_wave,
        response_wave,
        response.throughput,
        left=0.0,
        right=0.0,
    )
    if np.any((sampled_flux < 0) & (sampled_throughput > 0)):
        raise BandpassUnavailableError(
            f"Model spectrum is negative inside the {response.filter_id} response."
        )
    valid = (
        np.isfinite(sampled_flux)
        & (sampled_flux >= 0)
        & np.isfinite(sampled_throughput)
        & (sampled_throughput >= 0)
    )
    if np.count_nonzero(valid & (sampled_throughput > 0)) < 2:
        raise BandpassUnavailableError(f"No valid model samples overlap {response.filter_id}.")
    return integration_wave[valid], sampled_flux[valid], sampled_throughput[valid]


def _bandpass_signal(
    wavelength_angstrom: np.ndarray,
    flux_lambda: np.ndarray,
    throughput: np.ndarray,
    detector_type: str,
) -> float:
    detector_weight = (
        wavelength_angstrom
        if str(detector_type or "").lower().startswith("photon")
        else np.ones_like(wavelength_angstrom)
    )
    return float(_TRAPEZOID(flux_lambda * throughput * detector_weight, wavelength_angstrom))


def _positive_signal_ratio(numerator: float, denominator: float, filter_id: str) -> float:
    if not (
        math.isfinite(numerator)
        and math.isfinite(denominator)
        and numerator > 0
        and denominator > 0
    ):
        raise BandpassUnavailableError(f"Synthetic integration failed for {filter_id}.")
    return float(numerator / denominator)


def response_manifest_hash(responses: Mapping[tuple[str, str], FilterResponse]) -> str:
    """Hash only response state capable of changing synthetic photometry.

    Retrieval time, cache-write time, refresh reason, and cache format are
    operational provenance.  They remain available in the cache metadata and
    :func:`response_audit_manifest`, but deliberately do not make an otherwise
    identical atmosphere fit scientifically stale.
    """
    digest = hashlib.sha256()
    digest.update(f"{RESPONSE_MANIFEST_VERSION}\n".encode("utf-8"))
    for key, response in sorted(responses.items()):
        digest.update(
            (
                f"{key[0]}|{key[1]}|{response.response_hash}|{response.detector_type}|"
                f"{response.zero_point_jy}|{response.zero_point_contract}|"
                f"{response.svo_calibration_wavelength_ref_angstrom}|"
                f"{response.upstream_query_id}|"
                f"{SYNTHETIC_PHOTOMETRY_VERSION}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def response_audit_manifest(
    responses: Mapping[tuple[str, str], FilterResponse],
) -> dict[str, object]:
    """Return complete operational provenance for a resolved response set.

    This payload is intentionally broader than the scientific manifest hash.
    It supports cache/retrieval audits without coupling fit staleness to a wall
    clock or a metadata-only cache migration.
    """
    records: list[dict[str, object]] = []
    for key, response in sorted(responses.items()):
        records.append(
            {
                "requested_filter_id": key[0],
                "requested_mag_system": key[1],
                "filter_id": response.filter_id,
                "response_hash": response.response_hash,
                "detector_type": response.detector_type,
                "zero_point_jy": response.zero_point_jy,
                "zero_point_contract": response.zero_point_contract,
                "svo_calibration_wavelength_ref_angstrom": (
                    response.svo_calibration_wavelength_ref_angstrom
                ),
                "source_url": response.source_url,
                "upstream_query_id": response.upstream_query_id,
                "upstream_query_json": response.upstream_query_json,
                "retrieved_at_utc": response.retrieved_at_utc,
                "cached_at_utc": response.cached_at_utc,
                "refresh_provenance": response.refresh_provenance,
                "cache_format_version": response.cache_format_version,
            }
        )
    return {
        "manifest_version": RESPONSE_AUDIT_MANIFEST_VERSION,
        "scientific_manifest_hash": response_manifest_hash(responses),
        "synthetic_photometry_version": SYNTHETIC_PHOTOMETRY_VERSION,
        "responses": records,
    }


def response_audit_manifest_hash(
    responses: Mapping[tuple[str, str], FilterResponse],
) -> str:
    """Hash the complete operational audit payload."""
    encoded = json.dumps(
        response_audit_manifest(responses),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def top_hat_response(
    filter_id: str,
    center_angstrom: float,
    width_angstrom: float,
    *,
    mag_system: str = "AB",
    detector_type: str = "photon",
) -> FilterResponse:
    """Construct a deterministic response for tests and analytic validation."""
    center = float(center_angstrom)
    half_width = 0.5 * float(width_angstrom)
    wave = np.array([center - half_width, center - 0.49 * half_width, center + 0.49 * half_width, center + half_width])
    throughput = np.array([0.0, 1.0, 1.0, 0.0])
    return FilterResponse(
        filter_id=filter_id,
        wavelength_angstrom=wave,
        throughput=throughput,
        detector_type=detector_type,
        mag_system=mag_system,
        wavelength_ref_angstrom=center,
    )
