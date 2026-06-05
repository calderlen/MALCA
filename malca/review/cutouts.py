from __future__ import annotations

from dataclasses import dataclass
import math
import re
from urllib.parse import urlencode

from malca.nuclear.targets import DEC_ALIASES, RA_ALIASES


HIPS2FITS_URL = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
LEGACY_CUTOUT_URL = "https://www.legacysurvey.org/viewer/jpeg-cutout"
SKYVIEW_URL = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"

DEFAULT_CUTOUT_FOV_ARCSEC = 60.0
DEFAULT_CUTOUT_SIZE_PX = 512
DEFAULT_CUTOUT_SURVEY_KEY = "panstarrs-dr1-color"


@dataclass(frozen=True)
class CutoutSurvey:
    key: str
    label: str
    provider: str
    hips_id: str | None = None
    layer: str | None = None
    bands: str | None = None
    skyview_survey: str | None = None
    default_format: str = "jpg"
    coverage_note: str | None = None


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def _hips(label: str, hips_id: str, *, image_format: str = "jpg", note: str | None = None) -> CutoutSurvey:
    return CutoutSurvey(
        key=_slug(label),
        label=label,
        provider="hips2fits",
        hips_id=hips_id,
        default_format=image_format,
        coverage_note=note,
    )


CUTOUT_SURVEYS: tuple[CutoutSurvey, ...] = (
    _hips("eROSITA DR1", "erosita/dr1/exposure/021", note="eROSITA DR1 exposure band 021"),
    _hips("Fermi", "CDS/P/Fermi/color"),
    _hips("XMM PN", "xcatdb/P/XMM/PN/color"),
    _hips("Chandra", "cxc.harvard.edu/P/cda/hips/allsky/rgb"),
    _hips("GALEXGR6_7", "CDS/P/GALEXGR6_7/color"),
    _hips("DSS2 blue", "CDS/P/DSS2/blue"),
    _hips("DSS2", "CDS/P/DSS2/color"),
    _hips("Mellinger", "CDS/P/Mellinger/color"),
    _hips("Finkbeiner", "CDS/P/Finkbeiner"),
    _hips("SDSS9", "CDS/P/SDSS9/color"),
    _hips("DSS2 red", "CDS/P/DSS2/red"),
    _hips("VTSS Ha", "CDS/P/VTSS/HaCC"),
    _hips("PanSTARRS DR1 color", "CDS/P/PanSTARRS/DR1/color-i-r-g"),
    CutoutSurvey(
        key="desi-legacy-dr10",
        label="DESI Legacy DR10",
        provider="legacy",
        layer="ls-dr10",
        bands="grz",
        default_format="jpg",
    ),
    _hips("DECaPS DR2", "CDS/P/DECaPS/DR2/color"),
    _hips("2MASS", "CDS/P/2MASS/color"),
    _hips("GLIMPSE360", "IPAC/P/GLIMPSE360"),
    _hips("SPITZER", "CDS/P/SPITZER/color"),
    _hips("allWISE", "CDS/P/allWISE/color"),
    _hips("IRIS", "CDS/P/IRIS/color"),
    _hips("AKARI FIS", "CDS/P/AKARI/FIS/Color"),
)
CUTOUT_SURVEY_BY_KEY: dict[str, CutoutSurvey] = {survey.key: survey for survey in CUTOUT_SURVEYS}


def available_cutout_options() -> list[dict[str, str]]:
    """Return Dash dropdown-compatible survey options."""
    return [{"label": survey.label, "value": survey.key} for survey in CUTOUT_SURVEYS]


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _first_payload_number(payload: dict, aliases: tuple[str, ...]) -> float | None:
    lower_lookup = {str(key).lower(): key for key in payload.keys()}
    for alias in aliases:
        key = alias if alias in payload else lower_lookup.get(alias.lower())
        if key is None:
            continue
        value = _safe_float(payload.get(key))
        if value is not None:
            return value
    return None


def candidate_coordinates(payload: dict | None) -> tuple[float, float] | None:
    """Extract normalized candidate coordinates in decimal degrees."""
    if not isinstance(payload, dict):
        return None
    ra = _first_payload_number(payload, RA_ALIASES)
    dec = _first_payload_number(payload, DEC_ALIASES)
    if ra is None or dec is None:
        return None
    if not (0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0):
        return None
    return ra, dec


def normalize_cutout_survey_key(survey_key: str | None) -> str:
    key = str(survey_key or "").strip()
    return key if key in CUTOUT_SURVEY_BY_KEY else DEFAULT_CUTOUT_SURVEY_KEY


def _image_size(size_px: int | float) -> int:
    try:
        size = int(size_px)
    except (TypeError, ValueError):
        size = DEFAULT_CUTOUT_SIZE_PX
    return max(128, min(2048, size))


def _fov_arcsec(fov_arcsec: int | float) -> float:
    value = _safe_float(fov_arcsec)
    if value is None or value <= 0:
        return DEFAULT_CUTOUT_FOV_ARCSEC
    return max(1.0, min(7200.0, value))


def build_hips2fits_url(
    survey: CutoutSurvey,
    ra: float,
    dec: float,
    *,
    fov_arcsec: float = DEFAULT_CUTOUT_FOV_ARCSEC,
    size_px: int = DEFAULT_CUTOUT_SIZE_PX,
) -> str:
    if not survey.hips_id:
        raise ValueError(f"Survey {survey.label!r} is missing a HiPS ID")
    size = _image_size(size_px)
    fov_deg = _fov_arcsec(fov_arcsec) / 3600.0
    params = {
        "hips": survey.hips_id,
        "ra": f"{float(ra):.8f}",
        "dec": f"{float(dec):.8f}",
        "fov": f"{fov_deg:.10g}",
        "width": str(size),
        "height": str(size),
        "projection": "TAN",
        "coordsys": "icrs",
        "format": survey.default_format,
    }
    return f"{HIPS2FITS_URL}?{urlencode(params)}"


def build_legacy_cutout_url(
    survey: CutoutSurvey,
    ra: float,
    dec: float,
    *,
    fov_arcsec: float = DEFAULT_CUTOUT_FOV_ARCSEC,
    size_px: int = DEFAULT_CUTOUT_SIZE_PX,
) -> str:
    size = _image_size(size_px)
    pixscale = _fov_arcsec(fov_arcsec) / float(size)
    params = {
        "ra": f"{float(ra):.8f}",
        "dec": f"{float(dec):.8f}",
        "layer": survey.layer or "ls-dr10",
        "pixscale": f"{pixscale:.8g}",
        "bands": survey.bands or "grz",
        "size": str(size),
    }
    return f"{LEGACY_CUTOUT_URL}?{urlencode(params)}"


def build_skyview_url(
    survey: CutoutSurvey,
    ra: float,
    dec: float,
    *,
    fov_arcsec: float = DEFAULT_CUTOUT_FOV_ARCSEC,
    size_px: int = DEFAULT_CUTOUT_SIZE_PX,
) -> str:
    size = _image_size(size_px)
    fov_deg = _fov_arcsec(fov_arcsec) / 3600.0
    params = {
        "Position": f"{float(ra):.8f},{float(dec):.8f}",
        "Survey": survey.skyview_survey or survey.label,
        "Coordinates": "J2000",
        "Return": "JPEG",
        "Pixels": str(size),
        "Size": f"{fov_deg:.10g}",
    }
    return f"{SKYVIEW_URL}?{urlencode(params)}"


def build_cutout_url(
    survey_key: str | None,
    ra: float,
    dec: float,
    *,
    fov_arcsec: float = DEFAULT_CUTOUT_FOV_ARCSEC,
    size_px: int = DEFAULT_CUTOUT_SIZE_PX,
) -> str:
    key = normalize_cutout_survey_key(survey_key)
    survey = CUTOUT_SURVEY_BY_KEY[key]
    if survey.provider == "hips2fits":
        return build_hips2fits_url(survey, ra, dec, fov_arcsec=fov_arcsec, size_px=size_px)
    if survey.provider == "legacy":
        return build_legacy_cutout_url(survey, ra, dec, fov_arcsec=fov_arcsec, size_px=size_px)
    if survey.provider == "skyview":
        return build_skyview_url(survey, ra, dec, fov_arcsec=fov_arcsec, size_px=size_px)
    raise ValueError(f"Unknown cutout provider {survey.provider!r}")


def cutout_payload_for_candidate(
    payload: dict | None,
    *,
    selected_key: str | None = None,
    fov_arcsec: float = DEFAULT_CUTOUT_FOV_ARCSEC,
    size_px: int = DEFAULT_CUTOUT_SIZE_PX,
) -> dict[str, object]:
    key = normalize_cutout_survey_key(selected_key)
    survey = CUTOUT_SURVEY_BY_KEY[key]
    fov = _fov_arcsec(fov_arcsec)
    size = _image_size(size_px)
    coords = candidate_coordinates(payload)
    if coords is None:
        return {
            "has_coordinates": False,
            "ra": None,
            "dec": None,
            "selected_key": key,
            "selected_label": survey.label,
            "image_url": "",
            "source_url": "",
            "fov_arcsec": fov,
            "size_px": size,
            "message": "No RA/Dec available for survey cutout.",
        }
    ra, dec = coords
    image_url = build_cutout_url(key, ra, dec, fov_arcsec=fov, size_px=size)
    return {
        "has_coordinates": True,
        "ra": ra,
        "dec": dec,
        "selected_key": key,
        "selected_label": survey.label,
        "image_url": image_url,
        "source_url": image_url,
        "fov_arcsec": fov,
        "size_px": size,
        "message": f"{survey.label} | {fov:g}\" FOV | RA {ra:.6f}, Dec {dec:.6f}",
    }
