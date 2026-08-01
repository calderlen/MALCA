"""Adapters from MALCA light-curve products to joint-fit datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from malca.config import MJD_TO_JD
from malca.enrichment.atlas_forced_photometry import atlas_science_view
from malca.external_lc_manifest import lookup_external_lc_paths_from_manifest
from malca.io.lightcurve_io import load_lightcurve_df
from malca.review.lightcurve_sources import normalize_external_lc_dataframe

from .pspl import magnitude_to_relative_flux


@dataclass
class PhotometryDataset:
    dataset_id: str
    survey: str
    band: str
    instrument: str
    time_jd: np.ndarray
    flux: np.ndarray
    flux_error: np.ndarray
    flux_kind: str
    provenance_path: str
    reference_mag: float = np.nan

    def __post_init__(self) -> None:
        self.time_jd = np.asarray(self.time_jd, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        self.flux_error = np.asarray(self.flux_error, dtype=float)
        if not (self.time_jd.shape == self.flux.shape == self.flux_error.shape):
            raise ValueError(f"Mismatched arrays for dataset {self.dataset_id}")
        if self.flux_kind not in {"direct", "difference"}:
            raise ValueError(f"Unknown flux kind {self.flux_kind!r}")

    @property
    def n_points(self) -> int:
        return int(len(self.time_jd))


def _numeric(frame: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _text(frame: pd.DataFrame, names: Iterable[str], default: str) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name].fillna(default).astype(str).str.strip()
    return pd.Series(default, index=frame.index, dtype=str)


def _full_jd(values: pd.Series) -> np.ndarray:
    out = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = out[np.isfinite(out)]
    if finite.size and float(np.nanmedian(finite)) < 1_000_000.0:
        out = out + MJD_TO_JD
    return out


def _direct_dataset(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    survey: str,
    band: str,
    instrument: str,
    time_names: tuple[str, ...],
    mag_names: tuple[str, ...],
    error_names: tuple[str, ...],
    provenance_path: Path,
    min_points: int,
) -> PhotometryDataset | None:
    time = _full_jd(_numeric(frame, time_names))
    mag = _numeric(frame, mag_names).to_numpy(dtype=float)
    error = _numeric(frame, error_names).to_numpy(dtype=float)
    valid = np.isfinite(time) & np.isfinite(mag) & np.isfinite(error) & (error > 0.0)
    if int(valid.sum()) < int(min_points):
        return None
    flux, flux_error, reference_mag = magnitude_to_relative_flux(mag[valid], error[valid])
    order = np.argsort(time[valid])
    return PhotometryDataset(
        dataset_id=dataset_id,
        survey=survey,
        band=band,
        instrument=instrument,
        time_jd=time[valid][order],
        flux=flux[order],
        flux_error=flux_error[order],
        flux_kind="direct",
        provenance_path=str(provenance_path),
        reference_mag=reference_mag,
    )


def load_asassn_datasets(path: Path | str, *, min_points: int = 5) -> list[PhotometryDataset]:
    """Load quality-filtered ASAS-SN data, split by band and camera."""
    path = Path(path)
    frame = load_lightcurve_df(path, apply_quality=True)
    if frame.empty:
        return []
    band = _text(frame, ("band",), "unknown").str.lower().replace({"0": "g", "1": "v"})
    camera = _text(frame, ("camera_name", "camera"), "combined")
    frame = frame.assign(_ml_band=band, _ml_camera=camera)
    datasets: list[PhotometryDataset] = []
    for (band_name, camera_name), group in frame.groupby(["_ml_band", "_ml_camera"], sort=True):
        dataset = _direct_dataset(
            group,
            dataset_id=f"asassn:{band_name}:{camera_name}",
            survey="asassn",
            band=str(band_name),
            instrument=str(camera_name),
            time_names=("jd",),
            mag_names=("mag",),
            error_names=("mag_err",),
            provenance_path=path,
            min_points=min_points,
        )
        if dataset is not None:
            datasets.append(dataset)
    return datasets


def load_atlas_datasets(path: Path | str, *, min_points: int = 5) -> list[PhotometryDataset]:
    path = Path(path)
    frame = atlas_science_view(pd.read_parquet(path))
    if frame.empty:
        return []
    frame = frame.assign(_ml_band=_text(frame, ("filter", "F"), "unknown").str.lower())
    datasets: list[PhotometryDataset] = []
    for band_name, group in frame.groupby("_ml_band", sort=True):
        if band_name not in {"c", "o"}:
            continue
        dataset = _direct_dataset(
            group,
            dataset_id=f"atlas:{band_name}:forced",
            survey="atlas",
            band=str(band_name),
            instrument="forced",
            time_names=("mjd", "MJD", "jd", "JD"),
            mag_names=("mag",),
            error_names=("mag_err",),
            provenance_path=path,
            min_points=min_points,
        )
        if dataset is not None:
            datasets.append(dataset)
    return datasets


def _ztf_band(values: pd.Series) -> pd.Series:
    mapping = {
        "1": "zg", "1.0": "zg", "g": "zg", "ztf_g": "zg", "zg": "zg",
        "2": "zr", "2.0": "zr", "r": "zr", "ztf_r": "zr", "zr": "zr",
        "3": "zi", "3.0": "zi", "i": "zi", "ztf_i": "zi", "zi": "zi",
    }
    return values.fillna("unknown").astype(str).str.strip().str.lower().map(lambda x: mapping.get(x, x))


def load_ztf_datasets(path: Path | str, *, min_points: int = 5) -> list[PhotometryDataset]:
    path = Path(path)
    frame = normalize_external_lc_dataframe("ztf", pd.read_parquet(path))
    frame = frame.assign(_ml_band=_ztf_band(_text(frame, ("band", "filtercode", "filter"), "unknown")))
    datasets: list[PhotometryDataset] = []
    for band_name, group in frame.groupby("_ml_band", sort=True):
        if band_name not in {"zg", "zr", "zi"}:
            continue
        dataset = _direct_dataset(
            group,
            dataset_id=f"ztf:{band_name}:catalog",
            survey="ztf",
            band=str(band_name),
            instrument="catalog",
            time_names=("mjd", "hjd", "jd"),
            mag_names=("mag",),
            error_names=("mag_err", "magerr"),
            provenance_path=path,
            min_points=min_points,
        )
        if dataset is not None:
            datasets.append(dataset)
    return datasets


def load_ztf_forced_datasets(path: Path | str, *, min_points: int = 5) -> list[PhotometryDataset]:
    """Load ZTF forced difference flux without converting it to magnitude."""
    path = Path(path)
    frame = pd.read_parquet(path)
    if "procstatus" in frame.columns:
        status = pd.to_numeric(frame["procstatus"], errors="coerce").fillna(255)
        frame = frame.loc[status.eq(0)].copy()
    frame = frame.assign(_ml_band=_ztf_band(_text(frame, ("filter", "filtercode", "band"), "unknown")))
    datasets: list[PhotometryDataset] = []
    for band_name, group in frame.groupby("_ml_band", sort=True):
        if band_name not in {"zg", "zr", "zi"}:
            continue
        time = _full_jd(_numeric(group, ("jd", "mjd", "hjd")))
        flux = _numeric(group, ("forcediffimflux", "diff_flux", "flux")).to_numpy(dtype=float)
        error = _numeric(group, ("forcediffimfluxunc", "diff_flux_err", "flux_err")).to_numpy(dtype=float)
        valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(error) & (error > 0.0)
        if int(valid.sum()) < int(min_points):
            continue
        order = np.argsort(time[valid])
        datasets.append(
            PhotometryDataset(
                dataset_id=f"ztf_forced:{band_name}:forced",
                survey="ztf_forced",
                band=str(band_name),
                instrument="forced",
                time_jd=time[valid][order],
                flux=flux[valid][order],
                flux_error=error[valid][order],
                flux_kind="difference",
                provenance_path=str(path),
            )
        )
    return datasets


def load_candidate_datasets(
    candidate_id: str,
    *,
    asassn_path: Path | str | None,
    external_lc_dir: Path | str | None,
    surveys: Iterable[str] = ("asassn", "atlas", "ztf", "ztf_forced"),
    candidate_aliases: Iterable[str] = (),
    min_points: int = 5,
) -> list[PhotometryDataset]:
    """Load already-cached survey products for one candidate; never fetch remotely."""
    requested = {str(value).strip().lower() for value in surveys}
    datasets: list[PhotometryDataset] = []
    if "asassn" in requested and asassn_path is not None and Path(asassn_path).exists():
        datasets.extend(load_asassn_datasets(asassn_path, min_points=min_points))

    external_requested = requested & {"atlas", "ztf", "ztf_forced"}
    if external_lc_dir is None or not external_requested:
        return datasets
    identities = [str(candidate_id), *(str(value) for value in candidate_aliases if str(value).strip())]
    paths = lookup_external_lc_paths_from_manifest(
        external_lc_dir,
        tuple(sorted(external_requested)),
        tuple(identities),
    )
    loaders = {
        "atlas": load_atlas_datasets,
        "ztf": load_ztf_datasets,
        "ztf_forced": load_ztf_forced_datasets,
    }
    for survey in ("atlas", "ztf", "ztf_forced"):
        if survey not in external_requested:
            continue
        survey_paths = paths.get(survey, {})
        selected_path = next((survey_paths.get(identity) for identity in identities if survey_paths.get(identity)), None)
        if selected_path:
            datasets.extend(loaders[survey](selected_path, min_points=min_points))
    return datasets
