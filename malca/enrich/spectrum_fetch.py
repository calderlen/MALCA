from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class FetchBackend(str, Enum):
    SDSS = "sdss"
    DIRECT_FITS = "direct_fits"
    GAIA = "gaia"
    ESO = "eso"
    OAC = "oac"
    TNS = "tns"
    MAST = "mast"
    LAMOST = "lamost"
    DESI = "desi"
    APOGEE = "apogee"
    LINK_ONLY = "link_only"


class FetchStatus(str, Enum):
    OK = "ok"
    LINK_ONLY = "link_only"
    AUTH_REQUIRED = "auth_required"
    NOT_FOUND = "not_found"
    ERROR = "error"


@dataclass
class SpectrumData:
    wavelength: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray | None = None


@dataclass
class SpectrumFetchResult:
    status: FetchStatus
    data: SpectrumData | None = None
    link: str | None = None
    message: str = ""


# Maps survey_key → (backend, kwargs passed to the backend fetcher).
SURVEY_BACKEND_MAP: dict[str, tuple[FetchBackend, dict[str, Any]]] = {
    # Group 1 — direct flux API
    "sdss_dr16_spec": (FetchBackend.SDSS, {}),
    "sdss_boss": (FetchBackend.SDSS, {}),
    "sdss_eboss": (FetchBackend.SDSS, {}),
    "sdss_legacy": (FetchBackend.SDSS, {}),
    "sdss_segue": (FetchBackend.SDSS, {}),
    "sdss_spiders": (FetchBackend.SDSS, {}),
    "sdss_tdss": (FetchBackend.SDSS, {}),
    "sdss2_sn": (FetchBackend.SDSS, {}),
    "sdss_v": (FetchBackend.SDSS, {}),
    "desi_dr1": (FetchBackend.DESI, {}),
    "galah_dr3": (FetchBackend.LINK_ONLY, {}),
    "galah_dr4": (FetchBackend.LINK_ONLY, {}),
    "lamost_dr7": (FetchBackend.LAMOST, {}),
    "apogee_dr16": (FetchBackend.APOGEE, {}),
    "apogee_dr17": (FetchBackend.APOGEE, {}),
    "rave_dr6": (FetchBackend.DIRECT_FITS, {"url_template": "https://www.rave-survey.org/files/fits/{DATE}/RAVE_{ObsID}.fits"}),
    "cks": (FetchBackend.LINK_ONLY, {}),
    # Group 2 — astroquery / simple service
    "tns_spectra": (FetchBackend.TNS, {}),
    "gaia_rvs": (FetchBackend.GAIA, {"retrieval_type": "RVS"}),
    "gaia_xp": (FetchBackend.GAIA, {"retrieval_type": "XP_SAMPLED"}),
    "gaia_eso": (FetchBackend.ESO, {"collection": "Gaia-ESO"}),
    "osc": (FetchBackend.OAC, {}),
    "pessto": (FetchBackend.ESO, {"collection": "PESSTO"}),
    "vvds": (FetchBackend.ESO, {"collection": "VVDS"}),
    "zcosmos": (FetchBackend.ESO, {"collection": "zCOSMOS"}),
    "vandels": (FetchBackend.ESO, {"collection": "VANDELS"}),
    "vipers": (FetchBackend.ESO, {"collection": "VIPERS"}),
    "3d_hst": (FetchBackend.MAST, {}),
    # Group 3 — link_only
    "sixdf_gs": (FetchBackend.LINK_ONLY, {}),
    "2dfgrs": (FetchBackend.LINK_ONLY, {}),
    "ozdes": (FetchBackend.LINK_ONLY, {}),
    "deep2": (FetchBackend.LINK_ONLY, {}),
    "wigglez": (FetchBackend.LINK_ONLY, {}),
    "fmos_cosmos": (FetchBackend.LINK_ONLY, {}),
    "s5": (FetchBackend.LINK_ONLY, {}),
    "efeds_agn": (FetchBackend.LINK_ONLY, {}),
    "gama": (FetchBackend.LINK_ONLY, {}),
    "ztf_bts": (FetchBackend.LINK_ONLY, {}),
    "manga_dr17": (FetchBackend.LINK_ONLY, {}),
    # Group 4 — metadata only
    "simbad": (FetchBackend.LINK_ONLY, {}),
    "ned": (FetchBackend.LINK_ONLY, {}),
    "milliquas": (FetchBackend.LINK_ONLY, {}),
    "tns": (FetchBackend.LINK_ONLY, {}),
    "class": (FetchBackend.LINK_ONLY, {}),
    "primus": (FetchBackend.LINK_ONLY, {}),
    "2dflens": (FetchBackend.LINK_ONLY, {}),
    "desi_clagn": (FetchBackend.LINK_ONLY, {}),
}


def _cache_key(candidate_id: str, survey: str) -> str:
    raw = f"{candidate_id}:{survey}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_path(cache_dir: Path, candidate_id: str, survey: str) -> Path:
    return cache_dir / f"{candidate_id}_{survey}.npz"


def save_spectrum_cache(path: Path, data: SpectrumData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kw: dict[str, np.ndarray] = {"wavelength": data.wavelength, "flux": data.flux}
    if data.flux_err is not None:
        kw["flux_err"] = data.flux_err
    np.savez_compressed(path, **kw)


def load_spectrum_cache(path: Path) -> SpectrumData | None:
    if not path.exists():
        return None
    try:
        d = np.load(path)
        return SpectrumData(
            wavelength=d["wavelength"],
            flux=d["flux"],
            flux_err=d.get("flux_err"),
        )
    except Exception:
        return None


# ---------- Backend implementations ----------


def _fetch_sdss(row: pd.Series, **kwargs: Any) -> SpectrumFetchResult:
    spec_obj_id = row.get("SpecObjID") or row.get("SpObjID") or row.get("specobjid")
    plate = row.get("plate") or row.get("Plate")
    mjd = row.get("mjd") or row.get("MJD")
    fiberid = row.get("fiberID") or row.get("FiberID") or row.get("fiberid")

    if not spec_obj_id and not (plate and mjd and fiberid):
        return SpectrumFetchResult(FetchStatus.LINK_ONLY, message="No SDSS spectrum identifiers in row")

    try:
        from astroquery.sdss import SDSS
        if plate and mjd and fiberid:
            sp = SDSS.get_spectra(plate=int(plate), mjd=int(mjd), fiberID=int(fiberid))
        else:
            from astropy.table import Table
            t = Table([{"SpecObjID": int(spec_obj_id)}])
            sp = SDSS.get_spectra(matches=t)

        if not sp:
            return SpectrumFetchResult(FetchStatus.NOT_FOUND, message="SDSS query returned no spectra")

        hdu = sp[0]
        flux = np.array(hdu[1].data["flux"], dtype=np.float64)
        loglam = np.array(hdu[1].data["loglam"], dtype=np.float64)
        wavelength = 10.0 ** loglam
        ivar = hdu[1].data.get("ivar")
        flux_err = None
        if ivar is not None:
            ivar_arr = np.array(ivar, dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                flux_err = np.where(ivar_arr > 0, 1.0 / np.sqrt(ivar_arr), 0.0)

        return SpectrumFetchResult(FetchStatus.OK, data=SpectrumData(wavelength, flux, flux_err))
    except ImportError:
        return SpectrumFetchResult(FetchStatus.ERROR, message="astroquery not installed")
    except Exception as e:
        return SpectrumFetchResult(FetchStatus.ERROR, message=str(e))


def _fetch_gaia(row: pd.Series, *, retrieval_type: str = "RVS", **kwargs: Any) -> SpectrumFetchResult:
    source_id = row.get("source_id") or row.get("SOURCE_ID") or row.get("Source")
    if not source_id:
        return SpectrumFetchResult(FetchStatus.LINK_ONLY, message="No Gaia source_id")
    try:
        from astroquery.gaia import Gaia
        dl = Gaia.load_data(ids=[int(source_id)], retrieval_type=retrieval_type, data_release="Gaia DR3")
        if not dl:
            return SpectrumFetchResult(FetchStatus.NOT_FOUND, message=f"No {retrieval_type} data for {source_id}")
        key = next(iter(dl))
        table = dl[key][0].to_table()
        wavelength = np.array(table["wavelength"], dtype=np.float64)
        flux = np.array(table["flux"], dtype=np.float64)
        flux_err = np.array(table["flux_error"], dtype=np.float64) if "flux_error" in table.colnames else None
        return SpectrumFetchResult(FetchStatus.OK, data=SpectrumData(wavelength, flux, flux_err))
    except ImportError:
        return SpectrumFetchResult(FetchStatus.ERROR, message="astroquery not installed")
    except Exception as e:
        return SpectrumFetchResult(FetchStatus.ERROR, message=str(e))


def _fetch_eso(row: pd.Series, *, collection: str = "", config: Any = None, **kwargs: Any) -> SpectrumFetchResult:
    object_name = row.get("object_name") or row.get("provenance_name") or ""
    if not object_name:
        return SpectrumFetchResult(FetchStatus.LINK_ONLY, message="No object name for ESO query")
    try:
        from astroquery.eso import Eso
        eso = Eso()
        if config and config.eso_username and config.eso_password:
            eso.login(username=config.eso_username, store_password=False)
        table = eso.query_criteria(column_filters={"target": object_name}, collection=collection or None)
        if table is None or len(table) == 0:
            return SpectrumFetchResult(FetchStatus.NOT_FOUND, message=f"No ESO {collection} data for {object_name}")
        dp_id_col = "DP.ID" if "DP.ID" in table.colnames else table.colnames[0]
        files = eso.retrieve_data([str(table[dp_id_col][0])])
        if not files:
            return SpectrumFetchResult(FetchStatus.NOT_FOUND, message="ESO retrieve_data returned empty")
        return _parse_fits_spectrum(files[0])
    except ImportError:
        return SpectrumFetchResult(FetchStatus.ERROR, message="astroquery not installed")
    except Exception as e:
        if "login" in str(e).lower() or "auth" in str(e).lower():
            return SpectrumFetchResult(FetchStatus.AUTH_REQUIRED, message=str(e))
        return SpectrumFetchResult(FetchStatus.ERROR, message=str(e))


def _fetch_oac(row: pd.Series, **kwargs: Any) -> SpectrumFetchResult:
    name = row.get("provenance_name") or ""
    if not name:
        return SpectrumFetchResult(FetchStatus.LINK_ONLY, message="No OAC object name")
    try:
        from astroquery.oac import OAC
        result = OAC.get_single_spectrum(event=str(name))
        if result is None:
            return SpectrumFetchResult(FetchStatus.NOT_FOUND)
        wavelength = np.array(result["wavelength"], dtype=np.float64)
        flux = np.array(result["flux"], dtype=np.float64)
        flux_err = np.array(result["e_flux"], dtype=np.float64) if "e_flux" in result.colnames else None
        return SpectrumFetchResult(FetchStatus.OK, data=SpectrumData(wavelength, flux, flux_err))
    except ImportError:
        return SpectrumFetchResult(FetchStatus.ERROR, message="astroquery not installed")
    except Exception as e:
        return SpectrumFetchResult(FetchStatus.ERROR, message=str(e))


def _fetch_tns(row: pd.Series, *, config: Any = None, **kwargs: Any) -> SpectrumFetchResult:
    name = row.get("provenance_name") or row.get("tns_name") or ""
    api_key = config.tns_api_key if config else None
    if not name or not api_key:
        return SpectrumFetchResult(FetchStatus.LINK_ONLY, message="No TNS name or API key")
    try:
        import json
        from urllib.parse import quote
        from urllib.request import Request, urlopen

        url = f"https://www.wis-tns.org/api/get/object?objname={quote(str(name))}&include_spectra=1&api_key={quote(api_key)}"
        req = Request(url, headers={"User-Agent": "malca-spectrum-fetch/1.0"})
        with urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())

        data = payload.get("data", payload)
        if isinstance(data, list) and data:
            data = data[0]
        if not isinstance(data, dict):
            return SpectrumFetchResult(FetchStatus.NOT_FOUND)

        spectra = data.get("spectra", [])
        if not spectra:
            return SpectrumFetchResult(FetchStatus.NOT_FOUND, message="TNS object has no spectra")

        spec_url = spectra[0].get("ascii", {}).get("url") or spectra[0].get("asciifile")
        if not spec_url:
            return SpectrumFetchResult(FetchStatus.LINK_ONLY, message="TNS spectrum has no downloadable file URL")

        req2 = Request(spec_url, headers={"User-Agent": "malca-spectrum-fetch/1.0"})
        with urlopen(req2, timeout=30) as resp2:
            lines = resp2.read().decode(errors="replace").strip().splitlines()

        wl_list, fl_list = [], []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    wl_list.append(float(parts[0]))
                    fl_list.append(float(parts[1]))
                except ValueError:
                    continue
        if not wl_list:
            return SpectrumFetchResult(FetchStatus.NOT_FOUND, message="Could not parse TNS spectrum file")
        return SpectrumFetchResult(
            FetchStatus.OK,
            data=SpectrumData(np.array(wl_list), np.array(fl_list)),
        )
    except Exception as e:
        return SpectrumFetchResult(FetchStatus.ERROR, message=str(e))


def _fetch_lamost(row: pd.Series, **kwargs: Any) -> SpectrumFetchResult:
    obs_id = row.get("ObsID") or row.get("obsid") or row.get("obsID")
    if not obs_id:
        return SpectrumFetchResult(FetchStatus.LINK_ONLY, message="No LAMOST ObsID")
    try:
        from urllib.request import Request, urlopen
        url = f"https://www.lamost.org/dr10/v1.0/spectrum/fits/{obs_id}"
        req = Request(url, headers={"User-Agent": "malca-spectrum-fetch/1.0"})
        with urlopen(req, timeout=30) as resp:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".fits") as tmp:
                tmp.write(resp.read())
                tmp.flush()
                return _parse_fits_spectrum(tmp.name)
    except Exception as e:
        return SpectrumFetchResult(FetchStatus.ERROR, message=str(e))


def _fetch_direct_fits(row: pd.Series, *, url_template: str = "", **kwargs: Any) -> SpectrumFetchResult:
    if not url_template:
        return SpectrumFetchResult(FetchStatus.LINK_ONLY, message="No URL template")
    try:
        row_dict = row.to_dict()
        if "sobject_id" not in row_dict and "GALAH" in row_dict and pd.notna(row_dict.get("GALAH")):
            row_dict["sobject_id"] = int(row_dict["GALAH"])
            
        if "RAVEID" not in row_dict and "ID" in row_dict and pd.notna(row_dict.get("ID")):
            row_dict["RAVEID"] = str(row_dict["ID"]).strip()
        if "ObsID" in row_dict and pd.notna(row_dict.get("ObsID")):
            obsid = str(row_dict["ObsID"])
            if "_" in obsid:
                row_dict["DATE"] = obsid.split("_")[0]
                
        url = url_template.format_map({k: str(v) for k, v in row_dict.items() if pd.notna(v)})
    except (KeyError, IndexError):
        return SpectrumFetchResult(FetchStatus.LINK_ONLY, message="Could not format URL from row fields")
    try:
        from urllib.request import Request, urlopen
        req = Request(url, headers={"User-Agent": "malca-spectrum-fetch/1.0"})
        with urlopen(req, timeout=30) as resp:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".fits") as tmp:
                tmp.write(resp.read())
                tmp.flush()
                return _parse_fits_spectrum(tmp.name)
    except Exception as e:
        return SpectrumFetchResult(FetchStatus.ERROR, message=str(e))


def _fetch_mast(row: pd.Series, **kwargs: Any) -> SpectrumFetchResult:
    return SpectrumFetchResult(FetchStatus.LINK_ONLY, link="https://archive.stsci.edu/prepds/3d-hst/", message="MAST 3D-HST: use archive link")


def _fetch_desi(row: pd.Series, **kwargs: Any) -> SpectrumFetchResult:
    target_id = row.get("TARGETID") or row.get("targetid") or row.get("TARGET_ID")
    if not target_id:
        return SpectrumFetchResult(FetchStatus.LINK_ONLY, message="No DESI TARGETID")
    return SpectrumFetchResult(
        FetchStatus.LINK_ONLY,
        link=f"https://www.desi.lbl.gov/documents/etc/DESI-DR1-target-{target_id}/",
        message="DESI DR1 portal download not yet automated",
    )

_APOGEE_LOOKUP = None

def _fetch_apogee(row: pd.Series, **kwargs: Any) -> SpectrumFetchResult:
    global _APOGEE_LOOKUP
    if _APOGEE_LOOKUP is None:
        lookup_path = Path(__file__).parent.parent / "data" / "apogee_lookup.parquet"
        if not lookup_path.exists():
            return SpectrumFetchResult(FetchStatus.LINK_ONLY, message="apogee_lookup.parquet not found")
        try:
            _APOGEE_LOOKUP = pd.read_parquet(lookup_path).set_index("APOGEE_ID")
        except Exception as e:
            return SpectrumFetchResult(FetchStatus.ERROR, message=f"Failed to load lookup: {e}")
            
    apogee_id = row.get("ID") or row.get("id") or row.get("APOGEE_ID")
    if not apogee_id:
        return SpectrumFetchResult(FetchStatus.LINK_ONLY, message="No APOGEE ID")
    apogee_id = str(apogee_id).strip()
    
    if apogee_id not in _APOGEE_LOOKUP.index:
        return SpectrumFetchResult(FetchStatus.LINK_ONLY, message=f"APOGEE_ID {apogee_id} not in lookup")
        
    meta = _APOGEE_LOOKUP.loc[apogee_id]
    if isinstance(meta, pd.DataFrame):
        meta = meta.iloc[0]
        
    telescope = str(meta["TELESCOPE"])
    field = str(meta["FIELD"])
    
    url = f"https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/{telescope}/{field}/apStar-dr17-{apogee_id}.fits"
    try:
        from urllib.request import Request, urlopen
        req = Request(url, headers={"User-Agent": "malca-spectrum-fetch/1.0"})
        with urlopen(req, timeout=30) as resp:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".fits") as tmp:
                tmp.write(resp.read())
                tmp.flush()
                return _parse_fits_spectrum(tmp.name)
    except Exception as e:
        return SpectrumFetchResult(FetchStatus.ERROR, message=str(e))



def _parse_fits_spectrum(filepath: str) -> SpectrumFetchResult:
    """Best-effort extraction of wavelength + flux from a FITS file."""
    try:
        from astropy.io import fits
        with fits.open(filepath) as hdul:
            for hdu in hdul:
                if hdu.data is None:
                    continue
                if hasattr(hdu.data, "dtype") and hdu.data.dtype.names:
                    names = [n.lower() for n in hdu.data.dtype.names]
                    wl_col = next((n for n in hdu.data.dtype.names if n.lower() in {"wavelength", "wave", "lam", "lambda", "loglam"}), None)
                    fl_col = next((n for n in hdu.data.dtype.names if n.lower() in {"flux", "spec", "counts", "data"}), None)
                    if wl_col and fl_col:
                        wl = np.array(hdu.data[wl_col], dtype=np.float64)
                        if "log" in wl_col.lower():
                            wl = 10.0 ** wl
                        fl = np.array(hdu.data[fl_col], dtype=np.float64)
                        err_col = next((n for n in hdu.data.dtype.names if n.lower() in {"flux_err", "e_flux", "ivar", "sigma", "err", "error"}), None)
                        fe = None
                        if err_col:
                            fe = np.array(hdu.data[err_col], dtype=np.float64)
                            if "ivar" in err_col.lower():
                                with np.errstate(divide="ignore", invalid="ignore"):
                                    fe = np.where(fe > 0, 1.0 / np.sqrt(fe), 0.0)
                        return SpectrumFetchResult(FetchStatus.OK, data=SpectrumData(wl, fl, fe))

            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim in (1, 2):
                    header = hdu.header
                    crval = header.get("CRVAL1")
                    cdelt = header.get("CDELT1") or header.get("CD1_1")
                    if crval is not None and cdelt is not None:
                        flux = np.array(hdu.data, dtype=np.float64)
                        if flux.ndim == 2:
                            flux = flux[0]  # APOGEE apStar: row 0 is combined flux
                            
                        n = len(flux)
                        crpix = header.get("CRPIX1", 1.0)
                        wl = crval + (np.arange(n) - (crpix - 1)) * cdelt
                        
                        ctype1 = str(header.get("CTYPE1", "")).lower()
                        if header.get("DC-FLAG", 0) == 1 or "log" in ctype1 or "10" in ctype1:
                            wl = 10.0 ** wl
                            
                        err = None
                        if hdu.data.ndim == 2 and hdu.data.shape[0] > 1:
                            err = np.array(hdu.data[1], dtype=np.float64)
                            
                        return SpectrumFetchResult(
                            FetchStatus.OK,
                            data=SpectrumData(wl, flux, err),
                        )

        return SpectrumFetchResult(FetchStatus.NOT_FOUND, message="No recognized spectrum layout in FITS")
    except ImportError:
        return SpectrumFetchResult(FetchStatus.ERROR, message="astropy not installed")
    except Exception as e:
        return SpectrumFetchResult(FetchStatus.ERROR, message=str(e))


_BACKEND_DISPATCH: dict[FetchBackend, Any] = {
    FetchBackend.SDSS: _fetch_sdss,
    FetchBackend.GAIA: _fetch_gaia,
    FetchBackend.ESO: _fetch_eso,
    FetchBackend.OAC: _fetch_oac,
    FetchBackend.TNS: _fetch_tns,
    FetchBackend.LAMOST: _fetch_lamost,
    FetchBackend.DIRECT_FITS: _fetch_direct_fits,
    FetchBackend.MAST: _fetch_mast,
    FetchBackend.DESI: _fetch_desi,
    FetchBackend.APOGEE: _fetch_apogee,
}


def fetch_spectrum(
    row: pd.Series,
    *,
    survey_key: str | None = None,
    cache_dir: Path | None = None,
    config: Any = None,
) -> SpectrumFetchResult:
    """Fetch a spectrum for one row from spectra_long, using cache when available."""
    survey = survey_key or str(row.get("survey", ""))
    candidate_id = str(row.get("candidate_id", ""))

    if cache_dir:
        cached = load_spectrum_cache(_cache_path(cache_dir, candidate_id, survey))
        if cached is not None:
            return SpectrumFetchResult(FetchStatus.OK, data=cached)

    backend_spec = SURVEY_BACKEND_MAP.get(survey)
    if backend_spec is None:
        return SpectrumFetchResult(FetchStatus.LINK_ONLY, message=f"No backend for survey {survey}")

    backend, backend_kwargs = backend_spec
    if backend == FetchBackend.LINK_ONLY:
        link = str(row.get("link", "")) or None
        return SpectrumFetchResult(FetchStatus.LINK_ONLY, link=link)

    fetcher = _BACKEND_DISPATCH.get(backend)
    if fetcher is None:
        return SpectrumFetchResult(FetchStatus.ERROR, message=f"No fetcher for backend {backend}")

    result = fetcher(row, config=config, **backend_kwargs)

    if result.status == FetchStatus.OK and result.data and cache_dir:
        save_spectrum_cache(_cache_path(cache_dir, candidate_id, survey), result.data)

    return result


def prefetch_spectra(
    spectra_long: pd.DataFrame,
    *,
    cache_dir: Path,
    config: Any = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Batch-download spectra for all rows with a fetchable backend. Returns index DataFrame."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, object]] = []
    rows = list(spectra_long.iterrows())
    if show_progress:
        from tqdm.auto import tqdm
        rows = tqdm(rows, desc="Prefetch spectra", total=len(spectra_long))

    for _, row in rows:
        survey = str(row.get("survey", ""))
        candidate_id = str(row.get("candidate_id", ""))
        backend_spec = SURVEY_BACKEND_MAP.get(survey)
        if not backend_spec or backend_spec[0] == FetchBackend.LINK_ONLY:
            continue

        cache_file = _cache_path(cache_dir, candidate_id, survey)
        if cache_file.exists():
            index_rows.append({"candidate_id": candidate_id, "survey": survey, "status": "cached", "path": str(cache_file)})
            continue

        result = fetch_spectrum(row, survey_key=survey, cache_dir=cache_dir, config=config)
        index_rows.append({
            "candidate_id": candidate_id,
            "survey": survey,
            "status": result.status.value,
            "path": str(cache_file) if result.status == FetchStatus.OK else "",
            "message": result.message,
        })

    index_df = pd.DataFrame(index_rows) if index_rows else pd.DataFrame(columns=["candidate_id", "survey", "status", "path", "message"])
    index_path = cache_dir / "spectra_download_index.parquet"
    index_df.to_parquet(index_path, index=False, compression="zstd")
    return index_df
