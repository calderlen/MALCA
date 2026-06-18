from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from malca.enrich.neighbor import _ensure_candidate_id
from malca.enrich.spectra_provenance import merge_external_spectral_provenance


OSC_SEARCH_URL = "https://api.astronomyapi.com/0.9/json/search/ra/{ra}/dec/{dec}/radius/{radius_arcsec}"
TNS_OBJECT_URL = "https://www.wis-tns.org/api/get/object"


def _http_json(url: str, *, headers: dict[str, str] | None = None, timeout: float = 20.0) -> object | None:
    req = Request(url, headers=headers or {"User-Agent": "malca-transient-spectra/1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None


def _osc_rows_for_coordinate(ra_deg: float, dec_deg: float, *, radius_arcsec: float) -> list[dict[str, object]]:
    url = OSC_SEARCH_URL.format(ra=ra_deg, dec=dec_deg, radius_arcsec=radius_arcsec)
    payload = _http_json(url)
    if not isinstance(payload, list):
        return []

    rows: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("event") or "").strip()
        if not name:
            continue
        redshift = item.get("redshift") or item.get("z")
        spec_count = item.get("num_spectra") or item.get("spectra") or item.get("claimedtype")
        rows.append(
            {
                "name": name,
                "type": str(item.get("claimedtype") or item.get("type") or ""),
                "redshift": pd.to_numeric(redshift, errors="coerce"),
                "n_spectra": int(pd.to_numeric(spec_count, errors="coerce")) if pd.notna(pd.to_numeric(spec_count, errors="coerce")) else np.nan,
                "link": f"https://sne.space/{quote(name)}",
            }
        )
    return rows


def _tns_spectrum_metadata(name: str, *, api_key: str | None) -> dict[str, object]:
    if not name or not api_key:
        return {}
    url = f"{TNS_OBJECT_URL}?objname={quote(name)}&include_spectra=1&api_key={quote(api_key)}"
    payload = _http_json(url)
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data", payload)
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return {}
    spectra = data.get("spectra") or data.get("spectra_list") or []
    return {
        "tns_spectrum_count": len(spectra) if isinstance(spectra, list) else 0,
        "tns_spectrum_instrument": (
            str(spectra[0].get("instrument") or spectra[0].get("telescope") or "")
            if isinstance(spectra, list) and spectra and isinstance(spectra[0], dict)
            else ""
        ),
    }


def _load_catalog_csv(path: Path, *, source_name: str) -> pd.DataFrame:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return pd.DataFrame()
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(file_path)
    elif suffix in {".tsv", ".tab"}:
        frame = pd.read_csv(file_path, sep="\t")
    else:
        frame = pd.read_csv(file_path)

    # Normalize minimal schema.
    ra_col = next((c for c in frame.columns if str(c).lower() in {"ra", "ra_deg", "raj2000"}), None)
    dec_col = next((c for c in frame.columns if str(c).lower() in {"dec", "dec_deg", "dej2000"}), None)
    if ra_col is None or dec_col is None:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["ra_deg"] = pd.to_numeric(frame[ra_col], errors="coerce")
    out["dec_deg"] = pd.to_numeric(frame[dec_col], errors="coerce")
    out["survey"] = source_name
    out["catalog"] = f"local:{source_name}"
    name_col = next((c for c in frame.columns if str(c).lower() in {"name", "object", "tns_name", "sn_name"}), None)
    out["provenance_name"] = frame[name_col].astype(str) if name_col else ""
    z_col = next((c for c in frame.columns if str(c).lower() in {"redshift", "z", "z_spec"}), None)
    out["spectrum_redshift"] = pd.to_numeric(frame[z_col], errors="coerce") if z_col else np.nan
    type_col = next((c for c in frame.columns if str(c).lower() in {"type", "class", "spectral_type"}), None)
    out["spectrum_spectral_type"] = frame[type_col].astype(str) if type_col else ""
    out["link"] = ""
    return out.dropna(subset=["ra_deg", "dec_deg"])


def _crossmatch_local_catalog(
    coords: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    radius_arcsec: float,
) -> pd.DataFrame:
    if coords.empty or catalog.empty:
        return pd.DataFrame()

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    src = SkyCoord(ra=coords["ra_deg"].values, dec=coords["dec_deg"].values, unit="deg")
    cat = SkyCoord(ra=catalog["ra_deg"].values, dec=catalog["dec_deg"].values, unit="deg")
    idx, sep2d, _ = src.match_to_catalog_sky(cat)
    max_sep = radius_arcsec * u.arcsec

    rows: list[dict[str, object]] = []
    for i, cid in enumerate(coords["candidate_id"].astype(str)):
        if sep2d[i] > max_sep:
            continue
        hit = catalog.iloc[int(idx[i])]
        rows.append(
            {
                "candidate_id": cid,
                "survey": hit["survey"],
                "catalog": hit["catalog"],
                "sep_arcsec": float(sep2d[i].arcsec),
                "link": hit.get("link", ""),
                "spectrum_redshift": hit.get("spectrum_redshift", np.nan),
                "spectrum_spectral_type": hit.get("spectrum_spectral_type", ""),
                "provenance_name": hit.get("provenance_name", ""),
            }
        )
    return pd.DataFrame(rows)


def run_transient_spectra_enrichment(
    df: pd.DataFrame,
    *,
    radius_arcsec: float = 5.0,
    tns_api_key: str | None = None,
    osc_enabled: bool = True,
    pessto_catalog: Path | None = None,
    ztf_bts_catalog: Path | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Augment spectra_long-style rows with transient follow-up spectroscopy metadata."""
    frame = _ensure_candidate_id(df)
    if not {"candidate_id", "ra_deg", "dec_deg"}.issubset(frame.columns):
        return pd.DataFrame()

    coords = frame[["candidate_id", "ra_deg", "dec_deg"]].dropna().drop_duplicates("candidate_id")
    coords["candidate_id"] = coords["candidate_id"].astype(str)
    rows: list[pd.DataFrame] = []

    if osc_enabled:
        osc_rows: list[dict[str, object]] = []
        row_iter = coords.itertuples(index=False)
        if show_progress:
            from tqdm.auto import tqdm

            row_iter = tqdm(row_iter, total=len(coords), desc="transient:osc")

        for row in row_iter:
            for hit in _osc_rows_for_coordinate(float(row.ra_deg), float(row.dec_deg), radius_arcsec=radius_arcsec):
                osc_rows.append(
                    {
                        "candidate_id": str(row.candidate_id),
                        "survey": "osc",
                        "catalog": "api:osc",
                        "sep_arcsec": np.nan,
                        "link": hit.get("link"),
                        "spectrum_redshift": hit.get("redshift"),
                        "spectrum_spectral_type": hit.get("type", ""),
                        "provenance_name": hit.get("name", ""),
                        "transient_n_spectra": hit.get("n_spectra"),
                    }
                )
        if osc_rows:
            rows.append(pd.DataFrame(osc_rows))

    if "tns_name" in frame.columns:
        tns_rows: list[dict[str, object]] = []
        for _, row in frame.iterrows():
            name = str(row.get("tns_name") or "").strip()
            if not name:
                continue
            meta = _tns_spectrum_metadata(name, api_key=tns_api_key)
            tns_rows.append(
                {
                    "candidate_id": str(row["candidate_id"]),
                    "survey": "tns_spectra",
                    "catalog": "api:tns",
                    "sep_arcsec": np.nan,
                    "link": f"https://www.wis-tns.org/object/{quote(name)}",
                    "spectrum_redshift": pd.to_numeric(row.get("tns_redshift"), errors="coerce"),
                    "spectrum_spectral_type": str(row.get("tns_type") or ""),
                    "provenance_name": name,
                    "transient_n_spectra": meta.get("tns_spectrum_count"),
                    "transient_instrument": meta.get("tns_spectrum_instrument", ""),
                }
            )
        if tns_rows:
            rows.append(pd.DataFrame(tns_rows))

    for source_name, catalog_path in (("pessto", pessto_catalog), ("ztf_bts", ztf_bts_catalog)):
        if catalog_path is None:
            continue
        local = _load_catalog_csv(Path(catalog_path), source_name=source_name)
        if not local.empty:
            matched = _crossmatch_local_catalog(coords, local, radius_arcsec=radius_arcsec)
            if not matched.empty:
                rows.append(matched)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
