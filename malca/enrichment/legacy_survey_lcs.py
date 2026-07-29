"""Fetch and normalize legacy-survey light curves used by ``external-lcs``.

The functions in this module intentionally preserve archive-level provenance:
SuperWASP raw and SysRem magnitudes and KELT raw and TFA products remain
separate rows identified by ``proc_type``.
"""

from __future__ import annotations

import json
import os
import re
import tarfile
import threading
from html.parser import HTMLParser
from io import BytesIO, StringIO
from pathlib import Path

from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import numpy as np
import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError


NASA_TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
SUPERWASP_DATA_ROOT = "https://exoplanetarchive.ipac.caltech.edu/data/ETSS/SuperWASP/TBL/DR1"
KELT_INDEX_URL = "https://exoplanetarchive.ipac.caltech.edu/bulk_data_download/KELT_wget.tar.gz"
ASAS_SEARCH_URL = "https://www.astrouw.edu.pl/cgi-asas/asas_cat_input"
ASAS_DATA_URL = "https://www.astrouw.edu.pl/cgi-asas/asas_cgi_get_data"
DASCH_API_ROOT = "https://api.starglass.cfa.harvard.edu/public/dasch/dr7"
NSVS_SEARCH_URL = (
    "https://data.kasi.re.kr/vo/stardb/NSVS/conesearch/"
    "nsvs_conesearch_interactive.php"
)
NSVS_DATA_URL = (
    "https://data.kasi.re.kr/vo/stardb/lightcurve/"
    "nsvs_lightcurve_data.php"
)

SUPERWASP_MATCH_RADIUS_ARCSEC = 20.0
KELT_MATCH_RADIUS_ARCSEC = 35.0
NSVS_MATCH_RADIUS_ARCSEC = 20.0
ASAS3_MATCH_RADIUS_ARCSEC = 30.0
DASCH_MATCH_RADIUS_ARCSEC = 10.0
DASCH_SOURCE_CONTEXT_RADIUS_ARCSEC = 30.0

# These are the five AFLAG bits rejected by daschlab's documented
# ``Lightcurve.apply_standard_rejections()`` behavior. Keep the raw ``aflags``
# value as well so that later cleaning versions can make different choices.
DASCH_STANDARD_AFLAG_BITS = (7, 12, 13, 14, 16)
DASCH_STANDARD_AFLAG_MASK = sum(1 << bit for bit in DASCH_STANDARD_AFLAG_BITS)


def _raise_for_status(response: requests.Response) -> None:
    response.raise_for_status()
    if not response.content:
        raise RuntimeError("archive returned an empty response")


def _post_text_tolerating_broken_chunk_terminator(
    url: str,
    *,
    data: dict[str, str],
    timeout: float,
) -> str:
    """Read legacy CGI text even when the server omits the final HTTP chunk.

    The ASAS CGI intermittently closes an otherwise complete response with an
    outstanding chunk marker. Requests raises and discards ``response.content``
    in that case, so retain successfully received chunks and still require a
    non-empty payload.
    """
    response = requests.post(url, data=data, timeout=timeout, stream=True)
    response.raise_for_status()
    chunks: list[bytes] = []
    try:
        chunks.extend(response.iter_content(chunk_size=16_384))
    except ChunkedEncodingError:
        if not chunks:
            raise
    content = b"".join(chunk for chunk in chunks if chunk)
    if not content:
        raise RuntimeError("archive returned an empty response")
    return content.decode(response.encoding or "utf-8", errors="replace")


def _finite_numeric(series: pd.Series, *, maximum: float | None = None) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    good = np.isfinite(values)
    if maximum is not None:
        good &= values < maximum
    return values.where(good)


def _cone_separations(
    rows: pd.DataFrame,
    *,
    ra: float,
    dec: float,
    ra_col: str = "ra",
    dec_col: str = "dec",
) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=float)
    coords = SkyCoord(
        pd.to_numeric(rows[ra_col], errors="coerce").to_numpy(dtype=float) * u.deg,
        pd.to_numeric(rows[dec_col], errors="coerce").to_numpy(dtype=float) * u.deg,
    )
    target = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
    return pd.Series(coords.separation(target).arcsec, index=rows.index, dtype=float)


def _query_nasa_tap_csv(query: str, *, timeout: float = 90.0) -> pd.DataFrame:
    response = requests.get(
        NASA_TAP_SYNC_URL,
        params={"query": query, "format": "csv"},
        timeout=timeout,
    )
    _raise_for_status(response)
    try:
        return pd.read_csv(StringIO(response.text))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _nasa_cone_query(
    table: str,
    columns: tuple[str, ...],
    *,
    ra: float,
    dec: float,
    radius_arcsec: float,
) -> pd.DataFrame:
    radius_deg = float(radius_arcsec) / 3600.0
    dec_min = max(-90.0, float(dec) - radius_deg)
    dec_max = min(90.0, float(dec) + radius_deg)
    cos_dec = max(abs(np.cos(np.deg2rad(float(dec)))), 1e-3)
    ra_half_width = min(180.0, radius_deg / cos_dec)
    ra_min = (float(ra) - ra_half_width) % 360.0
    ra_max = (float(ra) + ra_half_width) % 360.0
    if ra_min <= ra_max:
        ra_clause = f"ra between {ra_min:.12f} and {ra_max:.12f}"
    else:
        ra_clause = (
            f"(ra >= {ra_min:.12f} or ra <= {ra_max:.12f})"
        )
    # The Exoplanet Archive time-series views currently fail spatial ADQL
    # predicates with an internal HTM20-column error. Query a small bounding
    # box and apply the exact spherical radius locally.
    query = (
        f"select {','.join(columns)} from {table} "
        f"where dec between {dec_min:.12f} and {dec_max:.12f} "
        f"and {ra_clause}"
    )
    rows = _query_nasa_tap_csv(query)
    if rows.empty:
        return rows
    rows.columns = [str(column).lower() for column in rows.columns]
    separation = _cone_separations(rows, ra=ra, dec=dec)
    return rows[separation <= float(radius_arcsec)].reset_index(drop=True)


def _read_ipac_table(content: bytes) -> pd.DataFrame:
    table = Table.read(BytesIO(content), format="ascii.ipac")
    frame = table.to_pandas()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    return frame


def _empty_mag_lc(extra_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    columns = ["hjd", "mag", "mag_err", "proc_type", "source_id", "sep_arcsec"]
    columns.extend(extra_columns)
    return pd.DataFrame(columns=list(dict.fromkeys(columns)))


def query_superwasp_lightcurve(
    ra: float,
    dec: float,
    *,
    radius_arcsec: float = SUPERWASP_MATCH_RADIUS_ARCSEC,
) -> pd.DataFrame:
    """Return the nearest SuperWASP source with raw and SysRem rows."""
    matches = _nasa_cone_query(
        "superwasptimeseries",
        ("sourceid", "ra", "dec", "tile", "npts"),
        ra=ra,
        dec=dec,
        radius_arcsec=radius_arcsec,
    )
    if matches.empty:
        return _empty_mag_lc(("tile", "photometry_level"))

    matches.columns = [str(column).lower() for column in matches.columns]
    matches["sep_arcsec"] = _cone_separations(matches, ra=ra, dec=dec)
    match = matches.sort_values("sep_arcsec").iloc[0]
    if float(match["sep_arcsec"]) > float(radius_arcsec):
        return _empty_mag_lc(("tile", "photometry_level"))

    source_id = str(match["sourceid"]).strip()
    tile = str(match["tile"]).strip()
    filename = f"{source_id.replace(' ', '_')}_lc.tbl"
    url = f"{SUPERWASP_DATA_ROOT}/{tile}/{filename}"
    response = requests.get(url, timeout=120.0)
    _raise_for_status(response)
    raw = _read_ipac_table(response.content)

    outputs: list[pd.DataFrame] = []
    for proc_type, mag_col, err_col, level in (
        ("raw", "mag2", "mag2_err", "raw"),
        ("sysrem", "tammag2", "tammag2_err", "minimally_detrended"),
    ):
        if "hjd" not in raw.columns or mag_col not in raw.columns:
            continue
        part = pd.DataFrame(
            {
                "hjd": _finite_numeric(raw["hjd"]),
                "mag": _finite_numeric(raw[mag_col], maximum=50.0),
                "mag_err": (
                    _finite_numeric(raw[err_col], maximum=10.0)
                    if err_col in raw.columns
                    else np.nan
                ),
            }
        ).dropna(subset=["hjd", "mag"])
        if part.empty:
            continue
        part["proc_type"] = proc_type
        part["photometry_level"] = level
        part["source_id"] = source_id
        part["tile"] = tile
        part["sep_arcsec"] = float(match["sep_arcsec"])
        outputs.append(part)

    if not outputs:
        return _empty_mag_lc(("tile", "photometry_level"))
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["proc_type", "hjd"]
    ).reset_index(drop=True)


def _download_cached_file(url: str, path: Path, *, timeout: float = 180.0) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout)
    _raise_for_status(response)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_bytes(response.content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


_KELT_WGET_RE = re.compile(
    r"""wget\s+-O\s+['"](?P<filename>[^'"]+)['"]\s+['"](?P<url>https?://[^'"]+)['"]""",
    re.IGNORECASE,
)


def _kelt_urls_for_matches(matches: pd.DataFrame, cache_dir: Path) -> list[tuple[pd.Series, str]]:
    archive = _download_cached_file(
        KELT_INDEX_URL,
        cache_dir / "KELT_wget.tar.gz",
    )
    wanted_fields = {str(value).strip().upper() for value in matches["kelt_field"]}
    wanted_ids = {str(value).strip() for value in matches["kelt_sourceid"]}
    found: list[tuple[pd.Series, str]] = []

    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            member_name = Path(member.name).name.upper()
            if wanted_fields and not any(field in member_name for field in wanted_fields):
                continue
            stream = bundle.extractfile(member)
            if stream is None:
                continue
            text = stream.read().decode("utf-8", errors="replace")
            for match in _KELT_WGET_RE.finditer(text):
                filename = match.group("filename")
                source_id = next(
                    (candidate for candidate in wanted_ids if filename.startswith(f"{candidate}_")),
                    None,
                )
                if source_id is None:
                    continue
                metadata = matches[matches["kelt_sourceid"].astype(str) == source_id]
                proc_match = re.search(r"_(raw|tfa)_lc\.tbl$", filename, re.IGNORECASE)
                proc_type = proc_match.group(1).lower() if proc_match else ""
                if "proc_type" in metadata.columns and proc_type:
                    same_proc = metadata[
                        metadata["proc_type"].astype(str).str.lower() == proc_type
                    ]
                    if not same_proc.empty:
                        metadata = same_proc
                orientation_match = re.search(
                    r"_(east|west)_(?:raw|tfa)_lc\.tbl$",
                    filename,
                    re.IGNORECASE,
                )
                orientation = orientation_match.group(1).lower() if orientation_match else ""
                if "orientation" in metadata.columns and orientation:
                    same_orientation = metadata[
                        metadata["orientation"].astype(str).str.lower() == orientation
                    ]
                    if not same_orientation.empty:
                        metadata = same_orientation
                row = metadata.iloc[0]
                url = re.sub(
                    r"^http://([^/:]+):80/",
                    r"https://\1/",
                    match.group("url"),
                    flags=re.IGNORECASE,
                )
                if url.lower().startswith("http://"):
                    url = "https://" + url[7:]
                found.append((row, url))

    unique: dict[str, tuple[pd.Series, str]] = {}
    for metadata, url in found:
        unique[url] = (metadata, url)
    return list(unique.values())


def query_kelt_lightcurve(
    ra: float,
    dec: float,
    *,
    cache_dir: Path,
    radius_arcsec: float = KELT_MATCH_RADIUS_ARCSEC,
) -> pd.DataFrame:
    """Return all raw/TFA files for the nearest KELT source position."""
    matches = _nasa_cone_query(
        "kelttimeseries",
        (
            "kelt_sourceid",
            "kelt_field",
            "kelt_orientation",
            "proc_type",
            "ra",
            "dec",
            "npts",
        ),
        ra=ra,
        dec=dec,
        radius_arcsec=radius_arcsec,
    )
    if matches.empty:
        return _empty_mag_lc(("kelt_field", "orientation", "photometry_level"))

    matches.columns = [str(column).lower() for column in matches.columns]
    matches = matches.rename(columns={"kelt_orientation": "orientation"})
    matches["sep_arcsec"] = _cone_separations(matches, ra=ra, dec=dec)
    minimum_sep = float(matches["sep_arcsec"].min())
    if minimum_sep > float(radius_arcsec):
        return _empty_mag_lc(("kelt_field", "orientation", "photometry_level"))
    # The same star can have several field/orientation/product entries with
    # identical coordinates. Do not blend neighboring KELT sources.
    matches = matches[matches["sep_arcsec"] <= minimum_sep + 1.0].copy()

    indexed_urls = _kelt_urls_for_matches(matches, cache_dir)
    if not indexed_urls:
        raise RuntimeError("KELT metadata matched, but no official download URL was indexed")

    outputs: list[pd.DataFrame] = []
    for metadata, url in indexed_urls:
        response = requests.get(url, timeout=120.0)
        _raise_for_status(response)
        raw = _read_ipac_table(response.content)
        if not {"time", "mag"}.issubset(raw.columns):
            continue
        part = pd.DataFrame(
            {
                "hjd": _finite_numeric(raw["time"]),
                "mag": _finite_numeric(raw["mag"], maximum=50.0),
                "mag_err": (
                    _finite_numeric(raw["mag_err"], maximum=10.0)
                    if "mag_err" in raw.columns
                    else np.nan
                ),
            }
        ).dropna(subset=["hjd", "mag"])
        if part.empty:
            continue
        proc_type = str(metadata.get("proc_type", "")).strip().lower()
        if not proc_type:
            found_proc = re.search(r"_(raw|tfa)_lc\.tbl", url, re.IGNORECASE)
            proc_type = found_proc.group(1).lower() if found_proc else "unknown"
        part["proc_type"] = proc_type
        part["photometry_level"] = "raw" if proc_type == "raw" else "minimally_detrended"
        part["source_id"] = str(metadata["kelt_sourceid"]).strip()
        part["kelt_field"] = str(metadata["kelt_field"]).strip()
        part["orientation"] = str(metadata.get("orientation", "")).strip().lower()
        part["sep_arcsec"] = float(metadata["sep_arcsec"])
        outputs.append(part)

    if not outputs:
        return _empty_mag_lc(("kelt_field", "orientation", "photometry_level"))
    return pd.concat(outputs, ignore_index=True).drop_duplicates(
        subset=["source_id", "proc_type", "orientation", "hjd"]
    ).sort_values(["source_id", "proc_type", "orientation", "hjd"]).reset_index(drop=True)


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.links: list[list[str]] = []
        self._row: list[str] | None = None
        self._row_links: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
            self._row_links = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
        elif tag == "a" and self._row_links is not None:
            href = dict(attrs).get("href")
            if href:
                self._row_links.append(str(href))

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self.links.append(self._row_links or [])
            self._row = None
            self._row_links = None


def _designation_coord(designation: str) -> SkyCoord | None:
    match = re.fullmatch(r"(\d{2})(\d{2})(\d{2}(?:\.\d)?)([+-])(\d{2})(\d{2}(?:\.\d)?)", designation)
    if match is None:
        return None
    hh, mm, ss, sign, dd, dm = match.groups()
    ra_deg = 15.0 * (float(hh) + float(mm) / 60.0 + float(ss) / 3600.0)
    dec_deg = float(dd) + float(dm) / 60.0
    if sign == "-":
        dec_deg *= -1.0
    return SkyCoord(ra_deg * u.deg, dec_deg * u.deg)


def _asas_designations(html: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"asas_variable/([^,/?\"']+),asas3", html)))


def _select_asas_designation(
    designations: list[str],
    *,
    ra: float,
    dec: float,
    radius_arcsec: float,
) -> tuple[str, float] | None:
    target = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
    candidates: list[tuple[float, str]] = []
    for designation in designations:
        coord = _designation_coord(designation)
        if coord is None:
            continue
        candidates.append((float(coord.separation(target).arcsec), designation))
    if not candidates:
        return None
    separation, designation = min(candidates)
    return (designation, separation) if separation <= float(radius_arcsec) else None


def _choose_asas_aperture(metadata: dict[str, str]) -> int:
    dispersions: list[tuple[float, int]] = []
    for aperture in range(5):
        try:
            value = float(
                str(metadata.get(f"cmer_{aperture}", "nan")).split()[0]
            )
        except (IndexError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            dispersions.append((value, aperture))
    if dispersions:
        return min(dispersions)[1]

    magnitudes = []
    for aperture in range(5):
        try:
            value = float(
                str(metadata.get(f"cmag_{aperture}", "nan")).split()[0]
            )
        except (IndexError, ValueError):
            continue
        if np.isfinite(value) and 0 < value < 29:
            magnitudes.append(value)
    magnitude = float(np.nanmedian(magnitudes)) if magnitudes else 12.0
    if magnitude < 9:
        return 4
    if magnitude < 10:
        return 3
    if magnitude < 11:
        return 2
    if magnitude < 12:
        return 1
    return 0


def _asas_metadata_number(
    metadata: dict[str, str],
    key: str,
    *,
    scale: float = 1.0,
) -> float:
    try:
        return float(str(metadata.get(key, "nan")).split()[0]) * float(scale)
    except (IndexError, TypeError, ValueError):
        return np.nan


def _asas_dataset_descriptor_parts(descriptor: str) -> tuple[str, str]:
    parts = str(descriptor).strip().split()
    if len(parts) >= 2 and parts[0].isdigit():
        return parts[0], parts[1]
    if parts:
        return "", parts[-1]
    return "", ""


def _asas_metadata_first_text(metadata: dict[str, str], key: str) -> str:
    parts = str(metadata.get(key, "")).strip().split()
    return parts[0] if parts else ""


def _asas_measurement_status(magnitude: float) -> tuple[str, bool]:
    if not np.isfinite(magnitude):
        return "nonfinite", False
    if np.isclose(magnitude, 29.999, rtol=0.0, atol=1e-6):
        return "below_detection_threshold", False
    if np.isclose(magnitude, 99.999, rtol=0.0, atol=1e-6):
        return "negative_aperture_flux", False
    if magnitude >= 29:
        return "invalid_magnitude", False
    return "detection", True


def parse_asas3_lightcurve(
    text: str,
    *,
    ra: float,
    dec: float,
    radius_arcsec: float = ASAS3_MATCH_RADIUS_ARCSEC,
) -> pd.DataFrame:
    """Parse the multi-dataset ASAS-3 response without discarding apertures."""
    target = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
    outputs: list[dict] = []
    metadata: dict[str, str] = {}
    response_metadata: dict[str, str] = {}
    dataset_number = ""
    dataset_descriptor = ""
    selected_aperture = 2
    dataset_sep = np.nan
    catalog_ra_deg = np.nan
    catalog_dec_deg = np.nan
    measured_ra_deg = np.nan
    measured_dec_deg = np.nan

    def _refresh_dataset_state() -> None:
        nonlocal selected_aperture, dataset_sep
        nonlocal catalog_ra_deg, catalog_dec_deg
        nonlocal measured_ra_deg, measured_dec_deg
        selected_aperture = _choose_asas_aperture(metadata)
        catalog_ra_deg = _asas_metadata_number(metadata, "cra", scale=15.0)
        catalog_dec_deg = _asas_metadata_number(metadata, "cdec")
        measured_ra_deg = _asas_metadata_number(metadata, "ra", scale=15.0)
        measured_dec_deg = _asas_metadata_number(metadata, "dec")
        if np.isfinite(catalog_ra_deg) and np.isfinite(catalog_dec_deg):
            dataset_coord = SkyCoord(
                catalog_ra_deg * u.deg,
                catalog_dec_deg * u.deg,
            )
            dataset_sep = float(dataset_coord.separation(target).arcsec)
        else:
            dataset_sep = np.nan

    aperture_order = tuple(range(5))
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#dataset="):
            metadata = {}
            dataset_value = line.split("=", 1)[1]
            dataset_parts = dataset_value.split(";", 1)
            dataset_number = dataset_parts[0].strip()
            dataset_descriptor = (
                dataset_parts[1].strip() if len(dataset_parts) > 1 else ""
            )
            continue
        if line.startswith("#") and "=" in line:
            key, value = line[1:].split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if dataset_number:
                metadata[key] = value
            else:
                response_metadata[key] = value
            continue
        if line.startswith("#"):
            continue

        _refresh_dataset_state()
        if np.isfinite(dataset_sep) and dataset_sep > float(radius_arcsec):
            continue
        fields = line.split()
        if len(fields) < 13:
            continue
        try:
            hjd = 2450000.0 + float(fields[0])
        except ValueError:
            continue
        dataset_entry, dataset_field = _asas_dataset_descriptor_parts(
            dataset_descriptor
        )
        grade = str(fields[11]).strip().upper()
        for offset, aperture in enumerate(aperture_order):
            try:
                magnitude = float(fields[1 + offset])
                frame_error = float(fields[6 + offset])
            except (IndexError, ValueError):
                continue
            measurement_status, magnitude_valid = _asas_measurement_status(
                magnitude
            )
            grade_pass = grade in {"A", "B"}
            quality_reasons: list[str] = []
            if not grade_pass:
                quality_reasons.append(f"grade_{grade or 'unknown'}")
            if not magnitude_valid:
                quality_reasons.append(measurement_status)
            row = {
                "hjd": hjd,
                "mag": magnitude if magnitude_valid else np.nan,
                "raw_mag": magnitude,
                # ASAS MER/ERR columns describe average frame quality, not the
                # uncertainty of this source measurement. Keep ``mag_err``
                # explicitly empty so viewers cannot accidentally use these
                # values as inverse-variance weights.
                "mag_err": np.nan,
                "frame_error": (
                    frame_error
                    if np.isfinite(frame_error) and 0 <= frame_error < 10
                    else np.nan
                ),
                "measurement_status": measurement_status,
                "magnitude_valid": magnitude_valid,
                "grade_pass": grade_pass,
                "quality_pass": bool(grade_pass and magnitude_valid),
                "quality_reasons": ";".join(quality_reasons),
                "aperture": aperture,
                "selected": aperture == selected_aperture,
                "grade": grade,
                "frame": fields[12],
                "dataset": dataset_number,
                "dataset_descriptor": dataset_descriptor,
                "dataset_entry": dataset_entry,
                "dataset_field": dataset_field,
                "dataset_header_json": json.dumps(
                    metadata, sort_keys=True, separators=(",", ":")
                ),
                "response_header_json": json.dumps(
                    response_metadata, sort_keys=True, separators=(",", ":")
                ),
                "response_ndata": _asas_metadata_number(
                    response_metadata, "ndata"
                ),
                "catalog_class": _asas_metadata_first_text(metadata, "class"),
                "source_id": _asas_metadata_first_text(metadata, "desig"),
                "band": "V",
                "sep_arcsec": dataset_sep,
                "catalog_ra_deg": catalog_ra_deg,
                "catalog_dec_deg": catalog_dec_deg,
                "measured_ra_deg": measured_ra_deg,
                "measured_dec_deg": measured_dec_deg,
                "cmag_aperture": _asas_metadata_number(
                    metadata, f"cmag_{aperture}"
                ),
                "cmer_aperture": _asas_metadata_number(
                    metadata, f"cmer_{aperture}"
                ),
                "nskip_aperture": _asas_metadata_number(
                    metadata, f"nskip_{aperture}"
                ),
            }
            for header_aperture in range(5):
                row[f"cmag_{header_aperture}"] = _asas_metadata_number(
                    metadata, f"cmag_{header_aperture}"
                )
                row[f"cmer_{header_aperture}"] = _asas_metadata_number(
                    metadata, f"cmer_{header_aperture}"
                )
                row[f"nskip_{header_aperture}"] = _asas_metadata_number(
                    metadata, f"nskip_{header_aperture}"
                )
            outputs.append(row)

    columns = [
        "hjd", "mag", "raw_mag", "mag_err", "frame_error",
        "measurement_status", "magnitude_valid", "grade_pass", "quality_pass",
        "quality_reasons", "aperture", "selected", "grade", "frame", "dataset",
        "dataset_descriptor", "dataset_entry", "dataset_field",
        "dataset_header_json", "response_header_json", "response_ndata",
        "catalog_class", "source_id", "band", "sep_arcsec", "catalog_ra_deg",
        "catalog_dec_deg", "measured_ra_deg", "measured_dec_deg",
        "cmag_aperture", "cmer_aperture", "nskip_aperture",
    ]
    for header_aperture in range(5):
        columns.extend(
            [
                f"cmag_{header_aperture}",
                f"cmer_{header_aperture}",
                f"nskip_{header_aperture}",
            ]
        )
    if not outputs:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(outputs, columns=columns).sort_values(
        ["dataset", "hjd", "aperture"]
    ).reset_index(drop=True)


def query_asas3_lightcurve(
    ra: float,
    dec: float,
    *,
    radius_arcsec: float = ASAS3_MATCH_RADIUS_ARCSEC,
) -> pd.DataFrame:
    response = requests.post(
        ASAS_SEARCH_URL,
        data={
            "source": "asas3",
            "coo": f"{float(ra) / 15.0:.8f} {float(dec):+.8f}",
            "equinox": "2000",
            "nmin": "4",
            "box": "15",
            "submit": "Search",
        },
        timeout=90.0,
    )
    _raise_for_status(response)
    selected = _select_asas_designation(
        _asas_designations(response.text),
        ra=ra,
        dec=dec,
        radius_arcsec=radius_arcsec,
    )
    if selected is None:
        return parse_asas3_lightcurve("", ra=ra, dec=dec, radius_arcsec=radius_arcsec)
    designation, _ = selected

    data_text = _post_text_tolerating_broken_chunk_terminator(
        f"{ASAS_DATA_URL}?{designation},asas3",
        data={"desig": designation, "GetData": "GetData"},
        timeout=120.0,
    )
    return parse_asas3_lightcurve(
        data_text,
        ra=ra,
        dec=dec,
        radius_arcsec=radius_arcsec,
    )


def _csv_records_frame(payload: object) -> pd.DataFrame:
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame()
    lines = [str(line) for line in payload if str(line).strip()]
    if not lines:
        return pd.DataFrame()
    return pd.read_csv(StringIO("\n".join(lines)))


def query_dasch_lightcurve(
    ra: float,
    dec: float,
    *,
    radius_arcsec: float = DASCH_MATCH_RADIUS_ARCSEC,
) -> pd.DataFrame:
    query_radius_arcsec = max(
        float(radius_arcsec),
        DASCH_SOURCE_CONTEXT_RADIUS_ARCSEC,
    )
    query_response = requests.post(
        f"{DASCH_API_ROOT}/querycat",
        json={
            "refcat": "apass",
            "ra_deg": float(ra),
            "dec_deg": float(dec),
            "radius_arcsec": query_radius_arcsec,
        },
        timeout=90.0,
    )
    _raise_for_status(query_response)
    matches = _csv_records_frame(query_response.json())
    if matches.empty:
        return pd.DataFrame(
            columns=[
                "hjd", "mag", "mag_err", "limiting_mag", "reject_flag",
                "series", "plate_number", "source_id", "refcat", "sep_arcsec",
                "catalog_sep_arcsec", "epoch_sep_arcsec", "aflags",
                "a2flags", "bflags", "b2flags", "standard_aflag_mask",
                "standard_aflag_reject", "quality_standard_pass",
            ]
        )
    matches.columns = [str(column).lower() for column in matches.columns]
    matches["sep_arcsec"] = _cone_separations(
        matches,
        ra=ra,
        dec=dec,
        ra_col="ra_deg",
        dec_col="dec_deg",
    )
    match = matches.sort_values("sep_arcsec").iloc[0]
    if float(match["sep_arcsec"]) > float(radius_arcsec):
        return pd.DataFrame()

    lc_response = requests.post(
        f"{DASCH_API_ROOT}/lightcurve",
        json={
            "refcat": "apass",
            "ref_number": int(match["ref_number"]),
            "gsc_bin_index": int(match["gsc_bin_index"]),
        },
        timeout=180.0,
    )
    _raise_for_status(lc_response)
    raw = _csv_records_frame(lc_response.json())
    if raw.empty:
        return pd.DataFrame()
    raw.columns = [str(column).lower() for column in raw.columns]

    mag = _finite_numeric(
        raw.get("magcal_magdep", pd.Series(index=raw.index)),
        maximum=50.0,
    )
    error_candidates = [
        _finite_numeric(raw.get(column, pd.Series(index=raw.index)), maximum=10.0)
        for column in (
            "magcal_local_rms",
            "magcal_local_error",
            "magcal_magdep_rms",
        )
    ]
    mag_error = error_candidates[0]
    for candidate in error_candidates[1:]:
        mag_error = mag_error.fillna(candidate)

    # Preserve the complete upstream schema. DASCH's raw lightcurve response
    # includes the quality flags, fitted positions, source-shape diagnostics,
    # and plate identifiers needed by the official reduction workflow.
    normalized = raw.copy()
    normalized["hjd"] = _finite_numeric(
        raw.get("date_jd", pd.Series(index=raw.index))
    )
    normalized["mag"] = mag
    normalized["mag_err"] = mag_error
    normalized["limiting_mag"] = _finite_numeric(
        raw.get("limiting_mag_local", pd.Series(index=raw.index)),
        maximum=50.0,
    )
    normalized = normalized.dropna(subset=["hjd"])

    match_ra = float(match["ra_deg"])
    match_dec = float(match["dec_deg"])
    normalized["source_id"] = str(match.get("ref_text", match["ref_number"]))
    normalized["refcat"] = "apass"
    normalized["sep_arcsec"] = float(match["sep_arcsec"])
    normalized["catalog_sep_arcsec"] = float(match["sep_arcsec"])
    normalized["catalog_ra_deg"] = match_ra
    normalized["catalog_dec_deg"] = match_dec
    normalized["query_radius_arcsec"] = query_radius_arcsec
    normalized["n_refcat_candidates"] = int(len(matches))
    context_columns = [
        column
        for column in (
            "ref_text", "ref_number", "gsc_bin_index", "ra_deg", "dec_deg",
            "stdmag", "color", "num_matches", "sep_arcsec",
        )
        if column in matches.columns
    ]
    normalized["refcat_candidates_json"] = (
        matches.sort_values("sep_arcsec")[context_columns].to_json(
            orient="records"
        )
    )

    alternatives = matches.drop(index=match.name).sort_values("sep_arcsec")
    if alternatives.empty:
        normalized["nearest_alternative_source_id"] = ""
        normalized["nearest_alternative_sep_arcsec"] = np.nan
        normalized["nearest_alternative_num_matches"] = np.nan
    else:
        alternative = alternatives.iloc[0]
        normalized["nearest_alternative_source_id"] = str(
            alternative.get("ref_text", alternative.get("ref_number", ""))
        )
        normalized["nearest_alternative_sep_arcsec"] = float(
            alternative["sep_arcsec"]
        )
        normalized["nearest_alternative_num_matches"] = pd.to_numeric(
            alternative.get("num_matches", np.nan),
            errors="coerce",
        )

    if {"ra_deg", "dec_deg"}.issubset(normalized.columns):
        normalized["epoch_sep_arcsec"] = _cone_separations(
            normalized,
            ra=match_ra,
            dec=match_dec,
            ra_col="ra_deg",
            dec_col="dec_deg",
        )
    else:
        normalized["epoch_sep_arcsec"] = np.nan

    aflags = pd.to_numeric(
        normalized.get("aflags", pd.Series(np.nan, index=normalized.index)),
        errors="coerce",
    )
    aflags_integer = aflags.fillna(0).astype("int64")
    normalized["standard_aflag_mask"] = (
        aflags_integer & DASCH_STANDARD_AFLAG_MASK
    )
    normalized["standard_aflag_reject"] = (
        normalized["mag"].notna()
        & normalized["standard_aflag_mask"].ne(0)
    )
    normalized["quality_standard_pass"] = (
        normalized["mag"].notna()
        & ~normalized["standard_aflag_reject"]
    )
    return normalized.sort_values("hjd").reset_index(drop=True)


def _nsvs_search_rows(html: str) -> pd.DataFrame:
    parser = _HTMLTableParser()
    parser.feed(html)
    records = []
    for cells, links in zip(parser.rows, parser.links):
        if len(cells) < 4:
            continue
        numeric_id = re.sub(r"\D", "", cells[0])
        if not numeric_id:
            for link in links:
                id_match = re.search(r"[?&]id=(\d+)", link)
                if id_match:
                    numeric_id = id_match.group(1)
                    break
        try:
            row_ra = float(cells[2])
            row_dec = float(cells[3])
        except (ValueError, IndexError):
            continue
        if numeric_id:
            records.append({"source_id": numeric_id, "ra": row_ra, "dec": row_dec})
    return pd.DataFrame(records)


def parse_nsvs_lightcurve(text: str, *, source_id: str, sep_arcsec: float) -> pd.DataFrame:
    if "file not found" in text.lower():
        raise RuntimeError("NSVS source matched, but the archive light-curve file is unavailable")

    parser = _HTMLTableParser()
    parser.feed(text)
    numeric_rows: list[list[str]] = []
    for row in parser.rows:
        if len(row) >= 3:
            numeric_rows.append(row)
    if not numeric_rows:
        numeric_rows = [
            re.split(r"[\s,]+", line.strip())
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith(("#", "<"))
        ]

    outputs = []
    for fields in numeric_rows:
        try:
            epoch = float(fields[0])
            magnitude = float(fields[1])
            error = float(fields[2])
        except (ValueError, IndexError):
            continue
        if not np.isfinite(epoch) or not np.isfinite(magnitude) or magnitude >= 50:
            continue
        hjd = epoch + 2400000.5 if epoch < 1_000_000 else epoch
        outputs.append(
            {
                "hjd": hjd,
                "mag": magnitude,
                "mag_err": error if np.isfinite(error) and 0 <= error < 10 else np.nan,
                "source_id": source_id,
                "sep_arcsec": float(sep_arcsec),
            }
        )
    if not outputs:
        raise RuntimeError("NSVS returned an unrecognized or empty light-curve response")
    return pd.DataFrame(outputs).drop_duplicates(subset=["hjd"]).sort_values(
        "hjd"
    ).reset_index(drop=True)


def query_nsvs_lightcurve(
    ra: float,
    dec: float,
    *,
    radius_arcsec: float = NSVS_MATCH_RADIUS_ARCSEC,
) -> pd.DataFrame:
    response = requests.post(
        NSVS_SEARCH_URL,
        data={
            "ra": f"{float(ra):.10f}",
            "dec": f"{float(dec):.10f}",
            # The KASI interactive cone service rejects radii above 0.005 deg.
            "sr": f"{min(float(radius_arcsec) / 3600.0, 0.005):.8f}",
        },
        timeout=90.0,
    )
    _raise_for_status(response)
    matches = _nsvs_search_rows(response.text)
    if matches.empty:
        return pd.DataFrame(columns=["hjd", "mag", "mag_err", "source_id", "sep_arcsec"])
    matches["sep_arcsec"] = _cone_separations(matches, ra=ra, dec=dec)
    match = matches.sort_values("sep_arcsec").iloc[0]
    if float(match["sep_arcsec"]) > float(radius_arcsec):
        return pd.DataFrame(columns=["hjd", "mag", "mag_err", "source_id", "sep_arcsec"])

    data_response = requests.get(
        NSVS_DATA_URL,
        params={"id": str(match["source_id"])},
        timeout=120.0,
    )
    _raise_for_status(data_response)
    return parse_nsvs_lightcurve(
        data_response.text,
        source_id=str(match["source_id"]),
        sep_arcsec=float(match["sep_arcsec"]),
    )
