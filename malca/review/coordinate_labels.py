"""Coordinate header labels for review publication figures."""

from __future__ import annotations

import math

from astropy.coordinates import SkyCoord
import astropy.units as u


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def payload_ra_dec(payload: dict) -> tuple[float, float] | None:
    """Return J2000 RA/Dec in degrees from a candidate payload, if available."""
    pairs = (
        ("ra", "dec"),
        ("ra_deg", "dec_deg"),
        ("ra_j2000", "dec_j2000"),
    )
    for ra_key, dec_key in pairs:
        ra = _finite_float(payload.get(ra_key))
        dec = _finite_float(payload.get(dec_key))
        if ra is not None and dec is not None:
            return ra, dec
    return None


def format_j_designation(ra_deg: float, dec_deg: float) -> str:
    """Format IAU-style Jhhmmss±ddmmss designation."""
    coord = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
    ra_h, ra_m, ra_s = coord.ra.hms
    dec_d, dec_m, dec_s = coord.dec.dms
    ra_str = f"{int(ra_h):02d}{int(ra_m):02d}{int(round(ra_s)):02d}"
    dec_sign = "+" if float(dec_deg) >= 0.0 else "-"
    dec_abs = abs(float(dec_d))
    dec_str = f"{int(dec_abs):02d}{int(abs(dec_m)):02d}{int(round(abs(dec_s))):02d}"
    return f"J{ra_str}{dec_sign}{dec_str}"


def format_ra_dec_degrees_label(ra_deg: float, dec_deg: float) -> str:
    """LaTeX label for decimal-degree coordinates."""
    return rf"$(\alpha,\,\delta)=({float(ra_deg):.5f}^\circ,\,{float(dec_deg):+.5f}^\circ)$"


def publication_coordinate_headers(payload: dict) -> tuple[str | None, str | None]:
    """Return left (J designation) and right (decimal RA/Dec) header labels."""
    coords = payload_ra_dec(payload)
    if coords is None:
        return None, None
    ra_deg, dec_deg = coords
    return format_j_designation(ra_deg, dec_deg), format_ra_dec_degrees_label(ra_deg, dec_deg)
