from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any

import numpy as np

from malca.review.classification_labels import resolve_catalog_class

try:  # Keep the review app importable when optional live-query deps are absent.
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier
except Exception:  # pragma: no cover - exercised by runtime fallback behavior.
    u = None
    SkyCoord = None
    Vizier = None


VSX_VIZIER_CATALOG = "B/vsx/vsx"
VSX_DETAIL_URL = "https://vsx.aavso.org/index.php?view=detail.top&oid={oid}"
VSX_NEIGHBOR_COLUMNS = ["+_r", "OID", "Name", "RAJ2000", "DEJ2000", "Type", "Period"]


@dataclass(frozen=True)
class VsxNeighbor:
    """A normalized nearby VSX catalog entry returned by VizieR."""

    sep_arcsec: float
    oid: str
    name: str
    ra_deg: float | None
    dec_deg: float | None
    vsx_type: str
    type_label: str
    period_days: float | None
    url: str | None


def _float_or_none(value: Any) -> float | None:
    try:
        if np.ma.is_masked(value):
            return None
    except Exception:
        pass
    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _string_or_empty(value: Any) -> str:
    try:
        if np.ma.is_masked(value):
            return ""
    except Exception:
        pass
    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value or "").strip()
    if text.lower() in {"--", "nan", "<na>", "none"}:
        return ""
    return text


def _short_vsx_type_label(value: object) -> str:
    resolved = resolve_catalog_class("vsx_class", value)
    if not resolved.value:
        return ""
    suffix = f" [{resolved.source}]" if resolved.source else ""
    if suffix and resolved.label.endswith(suffix):
        return resolved.label[: -len(suffix)]
    return resolved.label


def _row_value(row: Any, column: str) -> Any:
    try:
        return row[column]
    except Exception:
        return None


def _row_sep_arcsec(row: Any, target: Any) -> float | None:
    ra = _float_or_none(_row_value(row, "RAJ2000"))
    dec = _float_or_none(_row_value(row, "DEJ2000"))
    if ra is not None and dec is not None and SkyCoord is not None and u is not None:
        try:
            coord = SkyCoord(ra * u.deg, dec * u.deg)
            return float(target.separation(coord).arcsec)
        except Exception:
            pass
    return _float_or_none(_row_value(row, "_r"))


def _neighbor_from_row(row: Any, target: Any) -> VsxNeighbor | None:
    sep = _row_sep_arcsec(row, target)
    if sep is None:
        return None
    oid = _string_or_empty(_row_value(row, "OID"))
    vsx_type = _string_or_empty(_row_value(row, "Type"))
    url = VSX_DETAIL_URL.format(oid=oid) if oid else None
    return VsxNeighbor(
        sep_arcsec=sep,
        oid=oid,
        name=_string_or_empty(_row_value(row, "Name")),
        ra_deg=_float_or_none(_row_value(row, "RAJ2000")),
        dec_deg=_float_or_none(_row_value(row, "DEJ2000")),
        vsx_type=vsx_type,
        type_label=_short_vsx_type_label(vsx_type),
        period_days=_float_or_none(_row_value(row, "Period")),
        url=url,
    )


@lru_cache(maxsize=512)
def _find_nearby_vsx_cached(
    ra_deg: float,
    dec_deg: float,
    limit: int,
    radius_arcsec: float,
    timeout_sec: float,
) -> tuple[VsxNeighbor, ...]:
    if Vizier is None or SkyCoord is None or u is None:
        return tuple()
    try:
        target = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
        vizier = Vizier(columns=VSX_NEIGHBOR_COLUMNS, row_limit=limit)
        vizier.TIMEOUT = timeout_sec
        tables = vizier.query_region(
            target,
            radius=radius_arcsec * u.arcsec,
            catalog=VSX_VIZIER_CATALOG,
        )
    except Exception:
        return tuple()
    if not tables:
        return tuple()

    neighbors: list[VsxNeighbor] = []
    try:
        rows = tables[0]
    except Exception:
        return tuple()
    for row in rows:
        neighbor = _neighbor_from_row(row, target)
        if neighbor is not None and neighbor.sep_arcsec <= radius_arcsec:
            neighbors.append(neighbor)
    neighbors.sort(key=lambda item: item.sep_arcsec)
    return tuple(neighbors[:limit])


def find_nearby_vsx(
    ra_deg: object,
    dec_deg: object,
    *,
    limit: int = 3,
    radius_arcsec: float = 60.0,
    timeout_sec: float = 5.0,
) -> list[VsxNeighbor]:
    """Return nearby VSX entries from VizieR, hiding network/query failures."""
    ra = _float_or_none(ra_deg)
    dec = _float_or_none(dec_deg)
    if ra is None or dec is None:
        return []
    if limit <= 0 or radius_arcsec <= 0:
        return []
    if not (0.0 <= ra <= 360.0 and -90.0 <= dec <= 90.0):
        return []
    return list(
        _find_nearby_vsx_cached(
            float(ra),
            float(dec),
            int(limit),
            float(radius_arcsec),
            float(timeout_sec),
        )
    )
