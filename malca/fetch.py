"""Fetch ASAS-SN light curves via selectable SkyPatrol backends.

Supported backends:
  - ``skypatrol2``: SkyPatrol V2 API (HTTP endpoints used by pyasassn)
  - ``skypatrol1``: SkyPatrol V1 web API endpoints (photometry/variables JSON+CSV)

Public entry points:
  - download_lightcurve_by_id(asas_sn_id, backend=...)
  - download_lightcurve_by_gaia_id(gaia_id, backend=...)
  - cone_search(ra, dec, radius_arcsec, backend=...)

All downloaded files are saved in SkyPatrol web-CSV schema expected by
``malca.utils.read_skypatrol_csv``.
"""
from __future__ import annotations

from base64 import encodebytes
from pathlib import Path
import io
import json
import math
import os
import re
import time

import numpy as np
import pandas as pd
import requests

from malca.config import SKYPATROL_CACHE_DIR


SUPPORTED_FETCH_BACKENDS = ("skypatrol2", "skypatrol1")

_SKYPATROL2_BASE_URL = "http://asassn-lb01.ifa.hawaii.edu:9006"
_SKYPATROL2_DEFAULT_TIMEOUT = (4.0, 15.0)
_SKYPATROL2_BLOCK_TIMEOUT = (4.0, 15.0)
_SKYPATROL2_HTTP_ATTEMPTS = 2
_SKYPATROL2_BLOCK_ATTEMPTS = 1

_SKYPATROL1_BASE_URL = "https://asas-sn.osu.edu"
_SKYPATROL1_DEFAULT_TIMEOUT = (4.0, 20.0)
_SKYPATROL1_HTTP_ATTEMPTS = 2

_HTTP_BACKOFF_SECONDS = 0.35

_SKYPATROL2_DEFAULT_BLOCK_SERVERS = [
    "asassn-data01.ifa.hawaii.edu",
    "asassn-data02.ifa.hawaii.edu",
    "asassn-data03.ifa.hawaii.edu",
    "asassn-data04.ifa.hawaii.edu",
    "asassn-data05.ifa.hawaii.edu",
    "asassn-data06.ifa.hawaii.edu",
]

_SKYPATROL1_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)

_SKYPATROL_WEB_COLS = [
    "JD",
    "Flux",
    "Flux Error",
    "Mag",
    "Mag Error",
    "Limit",
    "FWHM",
    "Filter",
    "Quality",
    "Camera",
]


_service_meta: dict | None = None


# ---------------------------------------------------------------------------
# Default cache directory
# ---------------------------------------------------------------------------
_DEFAULT_CACHE = SKYPATROL_CACHE_DIR


def _normalize_backend_name(backend: str | None) -> str:
    """Return a supported backend name, defaulting to SkyPatrol2."""
    raw = backend if backend is not None else os.environ.get("MALCA_FETCH_BACKEND", "skypatrol2")
    value = str(raw).strip().lower()
    aliases = {
        "sp2": "skypatrol2",
        "skypatrol2": "skypatrol2",
        "v2": "skypatrol2",
        "sp1": "skypatrol1",
        "skypatrol1": "skypatrol1",
        "v1": "skypatrol1",
    }
    normalized = aliases.get(value, value)
    if normalized not in SUPPORTED_FETCH_BACKENDS:
        normalized = "skypatrol2"
    return normalized


def _ensure_cache(cache_dir: str | Path) -> Path:
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _to_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _extract_large_integer_like(value) -> int | None:
    """Extract first large integer from any scalar-ish input."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    m = re.search(r"(\d{8,})", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _safe_cache_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    token = token.strip("._")
    return token or "target"


def _json_safe_scalar(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _skypatrol_meta_path(lc_path: Path) -> Path:
    return lc_path.with_suffix(lc_path.suffix + ".meta.json")


def _read_skypatrol_metadata(lc_path: Path) -> dict:
    meta_path = _skypatrol_meta_path(lc_path)
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_skypatrol_metadata(lc_path: Path, metadata: dict) -> None:
    if not metadata:
        return
    payload = {str(k): _json_safe_scalar(v) for k, v in metadata.items()}
    try:
        with _skypatrol_meta_path(lc_path).open("w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
    except Exception:
        pass


def _is_valid_skypatrol_cache(lc_path: Path) -> bool:
    if not lc_path.exists():
        return False
    try:
        if lc_path.stat().st_size <= 0:
            return False
        df = pd.read_csv(lc_path, nrows=25)
    except Exception:
        return False
    if df.empty or "JD" not in df.columns:
        return False
    if not ({"Mag", "Flux"} & set(df.columns)):
        return False
    jd = pd.to_numeric(df["JD"], errors="coerce")
    return bool(jd.notna().any())


def _cached_skypatrol_result(lc_path: Path, refresh_cache: bool) -> tuple[Path, dict] | None:
    if refresh_cache:
        return None
    if not _is_valid_skypatrol_cache(lc_path):
        return None
    return lc_path, _read_skypatrol_metadata(lc_path)


def _infer_filter_from_camera(camera) -> str:
    """Infer ASAS-SN filter band from camera code.

    Historical ASAS-SN camera mapping:
      - ba..bh => V-band
      - bi..bt => g-band
    """
    cam = str(camera or "").strip()
    if not cam:
        return "g"

    lower = cam.lower()
    if len(lower) >= 2 and lower[1].isalpha():
        second = lower[1]
        if "a" <= second <= "h":
            return "V"
        if "i" <= second <= "t":
            return "g"

    return "V" if cam[0].isupper() else "g"


# ---------------------------------------------------------------------------
# CSV normalization: backend payload -> SkyPatrol web-CSV schema
# ---------------------------------------------------------------------------
_SP2_TO_WEB = {
    "jd": "JD",
    "flux": "Flux",
    "flux_err": "Flux Error",
    "mag": "Mag",
    "mag_err": "Mag Error",
    "limit": "Limit",
    "fwhm": "FWHM",
    "camera": "Camera",
    "quality": "Quality",
    "phot_filter": "Filter",
}


def _save_lc_as_skypatrol_csv(lc_data: pd.DataFrame, out_path: Path) -> Path:
    """Save a light curve in canonical SkyPatrol web-CSV format."""
    df = lc_data.rename(columns=_SP2_TO_WEB).copy()

    if "JD" not in df.columns and "hjd" in df.columns:
        df["JD"] = pd.to_numeric(df["hjd"], errors="coerce")

    if "Camera" not in df.columns and "camera" in df.columns:
        df["Camera"] = df["camera"]

    if "Filter" not in df.columns:
        if "Camera" in df.columns:
            df["Filter"] = df["Camera"].map(_infer_filter_from_camera)
        else:
            df["Filter"] = "g"

    if "Quality" not in df.columns:
        df["Quality"] = "G"

    for c in _SKYPATROL_WEB_COLS:
        if c not in df.columns:
            df[c] = np.nan

    df["JD"] = pd.to_numeric(df["JD"], errors="coerce")
    df = df[pd.notna(df["JD"])].copy()
    df = df.sort_values("JD").reset_index(drop=True)
    df = df[_SKYPATROL_WEB_COLS]
    df.to_csv(out_path, index=False)
    return out_path


def _normalize_sp1_lc_frame(lc_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize SkyPatrol1 photometry/variables frame to canonical columns."""
    df = lc_data.rename(
        columns={
            "hjd": "JD",
            "HJD": "JD",
            "jd": "JD",
            "mag": "Mag",
            "mag_err": "Mag Error",
            "flux": "Flux",
            "flux_err": "Flux Error",
            "camera": "Camera",
            "filter": "Filter",
        }
    ).copy()

    if "Filter" not in df.columns:
        if "Camera" in df.columns:
            df["Filter"] = df["Camera"].map(_infer_filter_from_camera)
        else:
            df["Filter"] = "g"

    if "Quality" not in df.columns:
        df["Quality"] = "G"
    if "Limit" not in df.columns:
        df["Limit"] = np.nan
    if "FWHM" not in df.columns:
        df["FWHM"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------
def _http_request(
    url: str,
    *,
    method: str,
    payload: dict | None = None,
    params: dict | None = None,
    timeout: tuple[float, float],
    attempts: int,
    context: str,
) -> requests.Response:
    """Perform HTTP request with retries and explicit timeout."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if method == "get":
                response = requests.get(url, params=params, timeout=timeout)
            elif method == "post":
                response = requests.post(url, params=params, json=payload, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < attempts:
                time.sleep(_HTTP_BACKOFF_SECONDS * attempt)

    raise RuntimeError(f"SkyPatrol {context} failed after {attempts} attempts: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# SkyPatrol2 (V2 API) backend helpers
# ---------------------------------------------------------------------------
def _deserialize_arrow_df(content: bytes, *, context: str) -> pd.DataFrame:
    """Decode parquet-bytes payload from SkyPatrol2 API."""
    try:
        return pd.read_parquet(io.BytesIO(content))
    except Exception as exc:
        raise RuntimeError(f"SkyPatrol2 returned invalid payload for {context}: {exc}") from exc


def _load_service_meta() -> dict:
    """Load and cache SkyPatrol2 schema + block-server metadata."""
    global _service_meta
    if _service_meta is not None:
        return _service_meta

    schema_resp = _http_request(
        f"{_SKYPATROL2_BASE_URL}/get_schema",
        method="get",
        timeout=_SKYPATROL2_DEFAULT_TIMEOUT,
        attempts=_SKYPATROL2_HTTP_ATTEMPTS,
        context="schema lookup",
    )
    block_resp = _http_request(
        f"{_SKYPATROL2_BASE_URL}/get_block_servers",
        method="get",
        timeout=_SKYPATROL2_DEFAULT_TIMEOUT,
        attempts=_SKYPATROL2_HTTP_ATTEMPTS,
        context="block-server lookup",
    )

    schema = schema_resp.json()
    block_servers = block_resp.json()

    if not isinstance(schema, dict):
        schema = {}

    if isinstance(block_servers, list):
        block_servers = [str(s).strip() for s in block_servers if str(s).strip()]
    else:
        block_servers = []

    if not block_servers:
        block_servers = list(_SKYPATROL2_DEFAULT_BLOCK_SERVERS)

    _service_meta = {"schema": schema, "block_servers": block_servers}
    return _service_meta


def _normalize_target_ids(target_ids: list) -> list:
    out: list = []
    for raw in target_ids:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        try:
            out.append(int(text))
        except ValueError:
            out.append(text)
    return out


def _sp2_default_query_cols(catalog: str, id_col: str) -> list[str]:
    if catalog in {"asteroids", "comets"}:
        cols = ["mpc_entry"]
    else:
        cols = ["asas_sn_id", "ra_deg", "dec_deg"]
        if catalog == "m_giants":
            cols.append("gaia_id")
        elif catalog == "master_list":
            cols.append("catalog_sources")
        elif catalog not in {"stellar_main", "master_list"}:
            cols.append("name")

    if id_col not in cols:
        cols.append(id_col)
    return cols


def _sp2_catalog_columns(catalog: str) -> list[str]:
    schema = _load_service_meta().get("schema", {})
    cat_data = schema.get(catalog, {}) if isinstance(schema, dict) else {}

    if isinstance(cat_data, list):
        cols = [
            str(row.get("col_names"))
            for row in cat_data
            if isinstance(row, dict) and row.get("col_names")
        ]
        if cols:
            return cols

    if isinstance(cat_data, dict):
        cols = cat_data.get("col_names")
        if isinstance(cols, list) and cols:
            return [str(c) for c in cols if str(c)]

    return _sp2_default_query_cols(catalog, "asas_sn_id")


def _sp2_lookup_targets(
    target_ids: list,
    *,
    id_col: str = "asas_sn_id",
    catalog: str = "master_list",
    cols: list[str] | None = None,
    download: bool = False,
) -> pd.DataFrame:
    ids = _normalize_target_ids(target_ids)
    if not ids:
        return pd.DataFrame()

    query_cols = list(cols) if cols else _sp2_default_query_cols(catalog, id_col)
    if id_col not in query_cols:
        query_cols.append(id_col)

    response = _http_request(
        f"{_SKYPATROL2_BASE_URL}/lookup_targets/catalog_list",
        method="post",
        payload={
            "tar_ids": ids,
            "catalog": catalog,
            "id_col": id_col,
            "cols": query_cols,
            "format": "arrow",
            "download": bool(download),
        },
        timeout=_SKYPATROL2_DEFAULT_TIMEOUT,
        attempts=_SKYPATROL2_HTTP_ATTEMPTS,
        context=f"target lookup ({catalog}.{id_col})",
    )
    return _deserialize_arrow_df(response.content, context=f"target lookup ({catalog}.{id_col})")


def _sp2_lookup_cone(
    ra: float,
    dec: float,
    *,
    radius_deg: float,
    catalog: str = "stellar_main",
    cols: list[str] | None = None,
    download: bool = False,
) -> pd.DataFrame:
    query_cols = list(cols) if cols else _sp2_default_query_cols(catalog, "asas_sn_id")
    response = _http_request(
        f"{_SKYPATROL2_BASE_URL}/lookup_cone/radius{radius_deg}_ra{ra}_dec{dec}",
        method="post",
        payload={
            "catalog": catalog,
            "cols": query_cols,
            "format": "arrow",
            "download": bool(download),
        },
        timeout=_SKYPATROL2_DEFAULT_TIMEOUT,
        attempts=_SKYPATROL2_HTTP_ATTEMPTS,
        context=f"cone search ({catalog})",
    )
    return _deserialize_arrow_df(response.content, context=f"cone search ({catalog})")


def _sp2_query_hash_for_list(target_ids: list, *, catalog: str, id_col: str, cols: list[str]) -> str:
    query_id = (
        f"listlen-{len(target_ids)}_listfirst-{target_ids[0]}_listend-{target_ids[-1]}"
        f"|catalog-{catalog}|id_col-{id_col}|cols-" + "/".join(cols)
    )
    return encodebytes(query_id.encode("utf-8")).decode()


def _sp2_download_block(query_hash: str, *, block_idx: int, catalog: str) -> pd.DataFrame:
    block_servers = list(_load_service_meta().get("block_servers") or _SKYPATROL2_DEFAULT_BLOCK_SERVERS)
    if not block_servers:
        block_servers = list(_SKYPATROL2_DEFAULT_BLOCK_SERVERS)

    n_servers = len(block_servers)
    start = block_idx % n_servers
    ordered_servers = [block_servers[(start + i) % n_servers] for i in range(n_servers)]

    last_exc: Exception | None = None
    for attempt in range(1, _SKYPATROL2_BLOCK_ATTEMPTS + 1):
        for server in ordered_servers:
            try:
                response = _http_request(
                    f"http://{server}:9006/get_block/"
                    f"query_hash-{query_hash}-block_idx-{block_idx}-catalog-{catalog}",
                    method="get",
                    timeout=_SKYPATROL2_BLOCK_TIMEOUT,
                    attempts=1,
                    context=f"block download ({server}, block {block_idx})",
                )
                return _deserialize_arrow_df(
                    response.content,
                    context=f"block download ({server}, block {block_idx})",
                )
            except Exception as exc:
                last_exc = exc

        if attempt < _SKYPATROL2_BLOCK_ATTEMPTS:
            time.sleep(_HTTP_BACKOFF_SECONDS * attempt)

    raise RuntimeError(f"SkyPatrol2 block download failed for block {block_idx}: {last_exc}") from last_exc


def _sp2_query_catalog_info(
    target_ids: list,
    id_col: str = "asas_sn_id",
    catalog: str = "stellar_main",
) -> dict:
    try:
        all_cols = _sp2_catalog_columns(catalog)
        result = _sp2_lookup_targets(
            target_ids,
            id_col=id_col,
            catalog=catalog,
            cols=all_cols,
            download=False,
        )
    except Exception:
        result = _sp2_lookup_targets(
            target_ids,
            id_col=id_col,
            catalog=catalog,
            cols=_sp2_default_query_cols(catalog, id_col),
            download=False,
        )

    if result is None or result.empty:
        return {}

    row = result.iloc[0].to_dict()
    return {
        k: v
        for k, v in row.items()
        if v is not None and not (isinstance(v, float) and pd.isna(v))
    }


def _sp2_download_lc(
    target_ids: list,
    id_col: str = "asas_sn_id",
    catalog: str = "master_list",
) -> pd.DataFrame | None:
    ids = _normalize_target_ids(target_ids)
    if not ids:
        return None

    query_cols = _sp2_default_query_cols(catalog, id_col)
    index_df = _sp2_lookup_targets(
        ids,
        id_col=id_col,
        catalog=catalog,
        cols=query_cols,
        download=True,
    )
    if index_df is None or index_df.empty:
        return None

    query_hash = _sp2_query_hash_for_list(ids, catalog=catalog, id_col=id_col, cols=query_cols)
    n_chunks = max(1, int(math.ceil(len(index_df) / 1000.0)))

    chunks: list[pd.DataFrame] = []
    for block_idx in range(n_chunks):
        block = _sp2_download_block(query_hash, block_idx=block_idx, catalog=catalog)
        if block is not None and not block.empty:
            chunks.append(block)

    if not chunks:
        return None
    if len(chunks) == 1:
        return chunks[0]
    return pd.concat(chunks, ignore_index=True)


def _sp2_cone_search_catalog(
    ra: float,
    dec: float,
    radius_arcsec: float = 3.0,
    catalog: str = "stellar_main",
) -> dict:
    try:
        result = _sp2_cone_search(ra, dec, radius_arcsec=radius_arcsec, catalog=catalog)
    except Exception:
        return {}

    if result is None or result.empty:
        return {}

    row = result.iloc[0].to_dict()
    return {
        k: v
        for k, v in row.items()
        if v is not None and not (isinstance(v, float) and pd.isna(v))
    }


def _sp2_download_lightcurve_by_id(
    asas_sn_id: str,
    cache_dir: str | Path,
    *,
    refresh_cache: bool = False,
) -> tuple[Path, dict]:
    cache = _ensure_cache(cache_dir)
    out = cache / f"{asas_sn_id}.csv"
    cached = _cached_skypatrol_result(out, refresh_cache)
    if cached is not None:
        return cached

    catalog_info = _sp2_query_catalog_info([int(asas_sn_id)], id_col="asas_sn_id", catalog="stellar_main")

    if not catalog_info:
        ml_info = _sp2_query_catalog_info([int(asas_sn_id)], id_col="asas_sn_id", catalog="master_list")
        if ml_info:
            catalog_info = dict(ml_info)
            ra = ml_info.get("ra_deg")
            dec = ml_info.get("dec_deg")
            if ra is not None and dec is not None:
                enriched = _sp2_cone_search_catalog(ra, dec, radius_arcsec=3.0)
                if enriched:
                    enriched.update({
                        k: v
                        for k, v in catalog_info.items()
                        if k in ("asas_sn_id", "ra_deg", "dec_deg")
                    })
                    catalog_info = enriched

    lc = _sp2_download_lc([int(asas_sn_id)], id_col="asas_sn_id", catalog="master_list")
    if lc is None or lc.empty:
        raise RuntimeError(f"No light curve returned for ASAS-SN ID {asas_sn_id}")

    _save_lc_as_skypatrol_csv(lc, out)
    _write_skypatrol_metadata(out, catalog_info)
    return out, catalog_info


def _sp2_download_lightcurve_by_gaia_id(
    gaia_id: str,
    cache_dir: str | Path,
    *,
    refresh_cache: bool = False,
) -> tuple[Path, dict]:
    cache = _ensure_cache(cache_dir)
    out = cache / f"gaia_{gaia_id}.csv"
    cached = _cached_skypatrol_result(out, refresh_cache)
    if cached is not None:
        return cached

    catalog_info = _sp2_query_catalog_info([int(gaia_id)], id_col="gaia_id", catalog="stellar_main")
    lc = _sp2_download_lc([int(gaia_id)], id_col="gaia_id", catalog="stellar_main")
    if lc is None or lc.empty:
        raise RuntimeError(f"No light curve returned for Gaia ID {gaia_id}")

    _save_lc_as_skypatrol_csv(lc, out)
    _write_skypatrol_metadata(out, catalog_info)
    return out, catalog_info


def _sp2_cone_search(
    ra: float,
    dec: float,
    radius_arcsec: float = 5.0,
    catalog: str = "stellar_main",
) -> pd.DataFrame:
    radius_deg = radius_arcsec / 3600.0
    try:
        cols = _sp2_catalog_columns(catalog)
        return _sp2_lookup_cone(
            ra,
            dec,
            radius_deg=radius_deg,
            catalog=catalog,
            cols=cols,
            download=False,
        )
    except Exception:
        return _sp2_lookup_cone(
            ra,
            dec,
            radius_deg=radius_deg,
            catalog=catalog,
            cols=_sp2_default_query_cols(catalog, "asas_sn_id"),
            download=False,
        )


# ---------------------------------------------------------------------------
# SkyPatrol1 backend helpers (non-pyasassn)
# ---------------------------------------------------------------------------
def _sp1_is_uuid(text: str) -> bool:
    return _SKYPATROL1_UUID_RE.fullmatch(str(text).strip()) is not None


def _sp1_extract_uuid(link: str) -> str | None:
    if not link:
        return None
    m = _SKYPATROL1_UUID_RE.search(str(link))
    return m.group(0) if m else None


def _sp1_nearest_row_by_coords(catalog_df: pd.DataFrame, ra_deg: float, dec_deg: float) -> tuple[dict | None, float | None]:
    """Return nearest row and angular separation (arcsec) for RA/Dec."""
    if catalog_df is None or catalog_df.empty:
        return None, None

    if "ra_deg" not in catalog_df.columns or "dec_deg" not in catalog_df.columns:
        return None, None

    ra_vals = pd.to_numeric(catalog_df["ra_deg"], errors="coerce").to_numpy(dtype=float)
    dec_vals = pd.to_numeric(catalog_df["dec_deg"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(ra_vals) & np.isfinite(dec_vals)
    if not np.any(valid):
        return None, None

    ra0 = float(ra_deg) % 360.0
    dec0 = float(dec_deg)
    ra = np.mod(ra_vals[valid], 360.0)
    dec = dec_vals[valid]

    ra0_rad = np.deg2rad(ra0)
    dec0_rad = np.deg2rad(dec0)
    dra = np.deg2rad(((ra - ra0 + 180.0) % 360.0) - 180.0)
    dec_rad = np.deg2rad(dec)

    sin_ddec = np.sin((dec_rad - dec0_rad) / 2.0)
    sin_dra = np.sin(dra / 2.0)
    a = sin_ddec * sin_ddec + np.cos(dec0_rad) * np.cos(dec_rad) * sin_dra * sin_dra
    a = np.clip(a, 0.0, 1.0)
    sep_rad = 2.0 * np.arcsin(np.sqrt(a))
    sep_arcsec = np.rad2deg(sep_rad) * 3600.0

    nearest_local = int(np.argmin(sep_arcsec))
    valid_indices = np.flatnonzero(valid)
    nearest_idx = int(valid_indices[nearest_local])
    return catalog_df.iloc[nearest_idx].to_dict(), float(sep_arcsec[nearest_local])


def _sp1_cone_search(
    ra: float,
    dec: float,
    radius_arcsec: float = 5.0,
    catalog: str = "stellar_main",
) -> pd.DataFrame:
    """Cone-search SkyPatrol1 photometry database.

    Returns rows with compatibility columns:
      - ``asas_sn_id``: SkyPatrol1 source UUID (string)
      - ``source_id``: SkyPatrol1 source UUID (string)
      - ``ra_deg``, ``dec_deg``
    """
    del catalog  # SkyPatrol1 photometry endpoint is catalog-agnostic.

    radius_arcmin = max(0.01, min(10.0, float(radius_arcsec) / 60.0))
    response = _http_request(
        f"{_SKYPATROL1_BASE_URL}/photometry.json",
        method="get",
        params={
            "ra": f"{float(ra):.8f}",
            "dec": f"{float(dec):.8f}",
            "radius": f"{radius_arcmin:.6f}",
            "sort_order": "asc",
        },
        timeout=_SKYPATROL1_DEFAULT_TIMEOUT,
        attempts=_SKYPATROL1_HTTP_ATTEMPTS,
        context="SkyPatrol1 cone search",
    )

    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(f"SkyPatrol1 cone search returned invalid JSON: {exc}") from exc

    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not results:
        return pd.DataFrame()

    rows: list[dict] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        link = str(row.get("link") or "")
        source_uuid = _sp1_extract_uuid(link)
        ra_row = _to_float(row.get("raj2000"))
        dec_row = _to_float(row.get("dej2000"))

        record = {
            "asas_sn_id": source_uuid,
            "source_id": source_uuid,
            "ra_deg": ra_row,
            "dec_deg": dec_row,
            "mean_vmag": _to_float(row.get("mean_vmag")),
            "epochs": row.get("epochs"),
            "sp1_source_catalog": row.get("source"),
            "sp1_link": link,
        }
        if ra_row is not None and dec_row is not None:
            nearest, sep = _sp1_nearest_row_by_coords(
                pd.DataFrame([{"ra_deg": ra_row, "dec_deg": dec_row}]),
                float(ra),
                float(dec),
            )
            if nearest is not None and sep is not None:
                record["sp1_sep_arcsec"] = sep
        rows.append(record)

    df = pd.DataFrame(rows)
    if "sp1_sep_arcsec" in df.columns:
        df = df.sort_values("sp1_sep_arcsec").reset_index(drop=True)
    return df


def _sp1_download_photometry_csv(source_uuid: str) -> pd.DataFrame:
    response = _http_request(
        f"{_SKYPATROL1_BASE_URL}/photometry/{source_uuid}.csv",
        method="get",
        timeout=_SKYPATROL1_DEFAULT_TIMEOUT,
        attempts=_SKYPATROL1_HTTP_ATTEMPTS,
        context=f"SkyPatrol1 photometry CSV ({source_uuid})",
    )
    try:
        df = pd.read_csv(io.StringIO(response.text))
    except Exception as exc:
        raise RuntimeError(f"SkyPatrol1 photometry CSV parse failed: {exc}") from exc
    if df.empty:
        raise RuntimeError(f"SkyPatrol1 photometry CSV is empty for source {source_uuid}")
    return df


def _sp1_download_variable_csv(variable_uuid: str) -> pd.DataFrame:
    response = _http_request(
        f"{_SKYPATROL1_BASE_URL}/variables/{variable_uuid}.csv",
        method="get",
        timeout=_SKYPATROL1_DEFAULT_TIMEOUT,
        attempts=_SKYPATROL1_HTTP_ATTEMPTS,
        context=f"SkyPatrol1 variable CSV ({variable_uuid})",
    )
    try:
        df = pd.read_csv(io.StringIO(response.text))
    except Exception as exc:
        raise RuntimeError(f"SkyPatrol1 variable CSV parse failed: {exc}") from exc
    if df.empty:
        raise RuntimeError(f"SkyPatrol1 variable CSV is empty for source {variable_uuid}")
    return df


def _sp1_lookup_variable_uuid(name: str) -> str | None:
    """Resolve an ASAS-SN variable name to a SkyPatrol1 variable UUID."""
    response = _http_request(
        f"{_SKYPATROL1_BASE_URL}/variables/lookup",
        method="get",
        params={"name": name},
        timeout=_SKYPATROL1_DEFAULT_TIMEOUT,
        attempts=_SKYPATROL1_HTTP_ATTEMPTS,
        context=f"SkyPatrol1 variable lookup ({name})",
    )

    resolved = _sp1_extract_uuid(response.url)
    if resolved:
        return resolved

    # Fallback: parse first variable link from response HTML.
    m = re.search(r'href="/variables/([0-9a-fA-F-]{36})"', response.text)
    if m:
        return m.group(1)
    return None


def _sp1_extract_coords_from_photometry_html(source_uuid: str) -> tuple[float | None, float | None]:
    """Parse RA/Dec from SkyPatrol1 photometry detail HTML heading."""
    response = _http_request(
        f"{_SKYPATROL1_BASE_URL}/photometry/{source_uuid}",
        method="get",
        timeout=_SKYPATROL1_DEFAULT_TIMEOUT,
        attempts=_SKYPATROL1_HTTP_ATTEMPTS,
        context=f"SkyPatrol1 photometry detail ({source_uuid})",
    )
    match = re.search(r"\(([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?)\)", response.text)
    if not match:
        return None, None
    return _to_float(match.group(1)), _to_float(match.group(2))


def _sp1_fetch_variable_metadata_by_name(name: str) -> dict:
    """Fetch variable catalog metadata row by ASASSN variable name."""
    response = _http_request(
        f"{_SKYPATROL1_BASE_URL}/variables.csv",
        method="get",
        params={
            "action": "index",
            "controller": "variables",
            "name": name,
        },
        timeout=_SKYPATROL1_DEFAULT_TIMEOUT,
        attempts=_SKYPATROL1_HTTP_ATTEMPTS,
        context=f"SkyPatrol1 variables CSV lookup ({name})",
    )

    try:
        df = pd.read_csv(io.StringIO(response.text))
    except Exception:
        return {}
    if df.empty:
        return {}

    row = df.iloc[0].to_dict()
    out: dict = {}
    for key, value in row.items():
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        out[str(key)] = value

    if "raj2000" in out and "ra_deg" not in out:
        out["ra_deg"] = out.get("raj2000")
    if "dej2000" in out and "dec_deg" not in out:
        out["dec_deg"] = out.get("dej2000")

    gaia_id = _extract_large_integer_like(out.get("edr3_source_id"))
    if gaia_id is not None:
        out["gaia_id"] = gaia_id

    if "asassn_name" in out and "asas_sn_id" not in out:
        out["asas_sn_id"] = out.get("asassn_name")

    return out


def _sp1_resolve_source_from_coords(ra: float, dec: float, radius_arcsec: float = 5.0) -> tuple[dict, float]:
    """Resolve nearest SkyPatrol1 photometry source around coordinates."""
    search_radii = [max(radius_arcsec, 5.0), 15.0, 30.0, 60.0]
    tried: list[float] = []
    for rad in search_radii:
        if rad in tried:
            continue
        tried.append(rad)
        candidates = _sp1_cone_search(ra, dec, radius_arcsec=rad)
        if candidates is None or candidates.empty:
            continue
        nearest_row, sep_arcsec = _sp1_nearest_row_by_coords(candidates, ra, dec)
        if nearest_row and sep_arcsec is not None:
            return nearest_row, sep_arcsec

    tried_txt = ", ".join(f"{r:.0f}\"" for r in tried)
    raise RuntimeError(f"No SkyPatrol1 source found near ({ra:.6f}, {dec:.6f}) within radii [{tried_txt}]")


def _sp1_download_lightcurve_by_id(
    asas_sn_id: str,
    cache_dir: str | Path,
    *,
    refresh_cache: bool = False,
) -> tuple[Path, dict]:
    """SkyPatrol1 implementation for ASAS-SN ID fetch.

    Supports:
      - UUID photometry source ids directly
      - Numeric SkyPatrol2 ASAS-SN ids via SkyPatrol2->coords resolution
      - ASASSN-V variable names via SkyPatrol1 variable lookup
    """
    cache = _ensure_cache(cache_dir)
    query = str(asas_sn_id).strip()
    if not query:
        raise RuntimeError("ASAS-SN query is empty")

    # Case 1: direct SkyPatrol1 photometry UUID
    if _sp1_is_uuid(query):
        out = cache / f"{query}.csv"
        cached = _cached_skypatrol_result(out, refresh_cache)
        if cached is not None:
            return cached
        lc = _sp1_download_photometry_csv(query)
        _save_lc_as_skypatrol_csv(_normalize_sp1_lc_frame(lc), out)
        ra_deg, dec_deg = _sp1_extract_coords_from_photometry_html(query)
        catalog_info: dict = {
            "source_id": query,
            "asas_sn_id": query,
        }
        if ra_deg is not None and dec_deg is not None:
            catalog_info["ra_deg"] = ra_deg
            catalog_info["dec_deg"] = dec_deg
        _write_skypatrol_metadata(out, catalog_info)
        return out, catalog_info

    # Case 2: numeric SkyPatrol2 ASAS-SN id -> resolve coords -> nearest SkyPatrol1 source
    if query.isdigit():
        out = cache / f"{query}.csv"
        cached = _cached_skypatrol_result(out, refresh_cache)
        if cached is not None:
            return cached

        catalog_info = _sp2_query_catalog_info([int(query)], id_col="asas_sn_id", catalog="stellar_main")
        if not catalog_info:
            catalog_info = _sp2_query_catalog_info([int(query)], id_col="asas_sn_id", catalog="master_list")
        if not catalog_info:
            raise RuntimeError(f"SkyPatrol1 could not resolve ASAS-SN ID {query} to coordinates")

        ra = _to_float(catalog_info.get("ra_deg"))
        dec = _to_float(catalog_info.get("dec_deg"))
        if ra is None or dec is None:
            raise RuntimeError(f"ASAS-SN ID {query} has no resolvable RA/Dec")

        nearest_row, sep_arcsec = _sp1_resolve_source_from_coords(ra, dec, radius_arcsec=5.0)
        source_uuid = str(nearest_row.get("source_id") or nearest_row.get("asas_sn_id") or "").strip()
        if not _sp1_is_uuid(source_uuid):
            raise RuntimeError(f"Could not resolve SkyPatrol1 source UUID near ASAS-SN ID {query}")

        lc = _sp1_download_photometry_csv(source_uuid)
        _save_lc_as_skypatrol_csv(_normalize_sp1_lc_frame(lc), out)

        merged = dict(catalog_info)
        merged.update({k: v for k, v in nearest_row.items() if v is not None})
        merged["source_id"] = source_uuid
        merged["sp1_source_id"] = source_uuid
        merged["sp1_sep_arcsec"] = sep_arcsec
        merged["asas_sn_id"] = query
        _write_skypatrol_metadata(out, merged)
        return out, merged

    # Case 3: ASASSN variable-style name (ASASSN-V J...) via variable lookup
    out = cache / f"{_safe_cache_token(query)}.csv"
    cached = _cached_skypatrol_result(out, refresh_cache)
    if cached is not None:
        return cached

    variable_uuid = _sp1_lookup_variable_uuid(query)
    if variable_uuid:
        lc = _sp1_download_variable_csv(variable_uuid)
        _save_lc_as_skypatrol_csv(_normalize_sp1_lc_frame(lc), out)

        metadata = _sp1_fetch_variable_metadata_by_name(query)
        metadata.setdefault("sp1_variable_uuid", variable_uuid)
        metadata.setdefault("asas_sn_id", metadata.get("asassn_name", query))
        _write_skypatrol_metadata(out, metadata)
        return out, metadata

    raise RuntimeError(
        "SkyPatrol1 could not resolve this ASAS-SN query. "
        "Use a numeric ASAS-SN ID, SkyPatrol1 source UUID, ASASSN-V name, or coordinates."
    )


def _sp1_download_lightcurve_by_gaia_id(
    gaia_id: str,
    cache_dir: str | Path,
    *,
    refresh_cache: bool = False,
) -> tuple[Path, dict]:
    """SkyPatrol1 implementation for Gaia DR3 id fetch.

    Uses SkyPatrol2 metadata lookup only to resolve RA/Dec, then fetches
    the light curve from SkyPatrol1 photometry endpoints.
    """
    cache = _ensure_cache(cache_dir)
    query = str(gaia_id).strip()
    if not query:
        raise RuntimeError("Gaia query is empty")
    if not query.isdigit():
        raise RuntimeError(f"Gaia ID must be numeric, got: {gaia_id}")

    out = cache / f"gaia_{query}.csv"
    cached = _cached_skypatrol_result(out, refresh_cache)
    if cached is not None:
        return cached

    catalog_info = _sp2_query_catalog_info([int(query)], id_col="gaia_id", catalog="stellar_main")
    if not catalog_info:
        raise RuntimeError(f"SkyPatrol1 could not resolve Gaia ID {query} to coordinates")

    ra = _to_float(catalog_info.get("ra_deg"))
    dec = _to_float(catalog_info.get("dec_deg"))
    if ra is None or dec is None:
        raise RuntimeError(f"Gaia ID {query} has no resolvable RA/Dec")

    nearest_row, sep_arcsec = _sp1_resolve_source_from_coords(ra, dec, radius_arcsec=5.0)
    source_uuid = str(nearest_row.get("source_id") or nearest_row.get("asas_sn_id") or "").strip()
    if not _sp1_is_uuid(source_uuid):
        raise RuntimeError(f"Could not resolve SkyPatrol1 source UUID near Gaia ID {query}")

    lc = _sp1_download_photometry_csv(source_uuid)
    _save_lc_as_skypatrol_csv(_normalize_sp1_lc_frame(lc), out)

    merged = dict(catalog_info)
    merged.update({k: v for k, v in nearest_row.items() if v is not None})
    merged["gaia_id"] = int(query)
    merged["source_id"] = source_uuid
    merged["sp1_source_id"] = source_uuid
    merged["sp1_sep_arcsec"] = sep_arcsec
    _write_skypatrol_metadata(out, merged)
    return out, merged


# ---------------------------------------------------------------------------
# Public API (backend-dispatched)
# ---------------------------------------------------------------------------
def download_lightcurve_by_id(
    asas_sn_id: str,
    cache_dir: str | Path = _DEFAULT_CACHE,
    *,
    backend: str | None = None,
    refresh_cache: bool = False,
) -> tuple[Path, dict]:
    """Download a light curve by ASAS-SN id using selected backend."""
    backend_name = _normalize_backend_name(backend)
    if backend_name == "skypatrol1":
        return _sp1_download_lightcurve_by_id(asas_sn_id, cache_dir, refresh_cache=refresh_cache)
    return _sp2_download_lightcurve_by_id(asas_sn_id, cache_dir, refresh_cache=refresh_cache)


def download_lightcurve_by_gaia_id(
    gaia_id: str,
    cache_dir: str | Path = _DEFAULT_CACHE,
    *,
    backend: str | None = None,
    refresh_cache: bool = False,
) -> tuple[Path, dict]:
    """Download a light curve by Gaia DR3 source id using selected backend."""
    backend_name = _normalize_backend_name(backend)
    if backend_name == "skypatrol1":
        return _sp1_download_lightcurve_by_gaia_id(gaia_id, cache_dir, refresh_cache=refresh_cache)
    return _sp2_download_lightcurve_by_gaia_id(gaia_id, cache_dir, refresh_cache=refresh_cache)


def cone_search(
    ra: float,
    dec: float,
    radius_arcsec: float = 5.0,
    catalog: str = "stellar_main",
    *,
    backend: str | None = None,
) -> pd.DataFrame:
    """Cone-search catalog rows using selected backend (no LC download)."""
    backend_name = _normalize_backend_name(backend)
    if backend_name == "skypatrol1":
        return _sp1_cone_search(ra, dec, radius_arcsec=radius_arcsec, catalog=catalog)
    return _sp2_cone_search(ra, dec, radius_arcsec=radius_arcsec, catalog=catalog)
