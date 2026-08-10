"""Adapters from MALCA light-curve products to joint-fit datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from malca.config import MJD_TO_JD
from malca.core.baseline import per_camera_gp_baseline_masked
from malca.core.utils import clean_lc, filter_bad_cameras, filter_residual_bad_cameras
from malca.enrichment.atlas_forced_photometry import (
    ATLAS_NOISE_MODEL_VERSION,
    apply_atlas_noise_model,
    atlas_huber_weights,
    estimate_atlas_noise_model,
    preprocess_atlas_frame,
)
from malca.external_lc_manifest import lookup_external_lc_paths_from_manifest
from malca.io.lightcurve_io import load_lightcurve_df, to_asassn_algorithm_frame
from malca.review.lightcurve_sources import normalize_external_lc_dataframe

from .pspl import magnitude_to_relative_flux, pspl_magnification, solve_linear_flux_parameters


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
    metadata: dict[str, object] = field(default_factory=dict)
    point_metadata: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time_jd = np.asarray(self.time_jd, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        self.flux_error = np.asarray(self.flux_error, dtype=float)
        if not (self.time_jd.shape == self.flux.shape == self.flux_error.shape):
            raise ValueError(f"Mismatched arrays for dataset {self.dataset_id}")
        if self.flux_kind not in {"direct", "difference"}:
            raise ValueError(f"Unknown flux kind {self.flux_kind!r}")
        aligned_metadata: dict[str, np.ndarray] = {}
        for name, values in self.point_metadata.items():
            array = np.asarray(values)
            if array.ndim != 1 or len(array) != len(self.time_jd):
                raise ValueError(
                    f"Point metadata {name!r} is not aligned for dataset {self.dataset_id}"
                )
            aligned_metadata[str(name)] = array
        self.point_metadata = aligned_metadata

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
    """Load pipeline-cleaned ASAS-SN data as one normalized dataset per band."""
    path = Path(path)
    frame = load_lightcurve_df(path, apply_quality=True)
    if frame.empty:
        return []

    frame = clean_lc(to_asassn_algorithm_frame(frame)).reset_index(drop=True)
    if frame.empty:
        return []

    frame, _ = filter_bad_cameras(
        frame,
        lc_path=str(path),
        filter_scatter=False,
        filter_offset=False,
        filter_catastrophic=True,
    )
    if frame.empty:
        return []

    baseline = per_camera_gp_baseline_masked(frame)
    filtered, residual_bad_cameras = filter_residual_bad_cameras(frame, baseline)
    if residual_bad_cameras:
        frame = filtered.reset_index(drop=True)
        if frame.empty:
            return []
        baseline = per_camera_gp_baseline_masked(frame)

    frame = frame.reset_index(drop=True)
    baseline = baseline.reset_index(drop=True)
    if len(baseline) != len(frame):
        raise ValueError("ASAS-SN baseline changed the number of light-curve rows")
    required = {"baseline", "resid", "sigma_eff", "is_masked"}
    missing = sorted(required - set(baseline.columns))
    if missing:
        raise RuntimeError(f"ASAS-SN baseline is missing required columns: {missing}")

    prepared = baseline.copy()
    prepared["_ml_band"] = (
        pd.to_numeric(prepared["v_g_band"], errors="coerce")
        .map({0.0: "g", 1.0: "v"})
    )
    # The masked GP is fitted separately for each camera. Convert its aligned
    # residual magnitudes to relative flux, so the detrended baseline is unity.
    prepared["_ml_flux"], prepared["_ml_flux_error"], _ = magnitude_to_relative_flux(
        pd.to_numeric(prepared["resid"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(prepared["sigma_eff"], errors="coerce").to_numpy(dtype=float),
        reference_mag=0.0,
    )

    datasets: list[PhotometryDataset] = []
    for band_name, group in prepared.groupby("_ml_band", sort=True):
        time = _full_jd(pd.to_numeric(group["JD"], errors="coerce"))
        flux = pd.to_numeric(group["_ml_flux"], errors="coerce").to_numpy(dtype=float)
        flux_error = pd.to_numeric(group["_ml_flux_error"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_error) & (flux_error > 0.0)
        if int(valid.sum()) < int(min_points):
            continue
        order = np.argsort(time[valid])
        datasets.append(
            PhotometryDataset(
                dataset_id=f"asassn:{band_name}:combined",
                survey="asassn",
                band=str(band_name),
                instrument="combined_cameras",
                time_jd=time[valid][order],
                flux=flux[valid][order],
                flux_error=flux_error[valid][order],
                flux_kind="direct",
                provenance_path=str(path),
                # The converted magnitude is mag - per-camera GP baseline.
                reference_mag=0.0,
            )
        )
    return datasets


def load_atlas_datasets(path: Path | str, *, min_points: int = 5) -> list[PhotometryDataset]:
    """Load reduced-image ATLAS forced photometry in its native microJy scale.

    This applies the shared epoch-quality rules but deliberately leaves the
    empirical noise calibration to the scientific caller, which must define
    which epochs represent the quiescent source.
    """
    path = Path(path)
    flagged = preprocess_atlas_frame(pd.read_parquet(path))
    if flagged.empty:
        return []
    datasets: list[PhotometryDataset] = []
    for band_name, all_band_rows in flagged.groupby("atlas_filter", sort=True):
        if band_name not in {"c", "o"}:
            continue
        counts_by_site: list[dict[str, object]] = []
        for (site_code, site_name), site_rows in all_band_rows.groupby(
            ["atlas_obs_site_code", "atlas_obs_site"], sort=True, dropna=False
        ):
            n_raw = int(len(site_rows))
            n_good = int(site_rows["atlas_good"].fillna(False).sum())
            counts_by_site.append(
                {
                    "atlas_obs_site_code": str(site_code),
                    "atlas_obs_site": str(site_name),
                    "n_raw": n_raw,
                    "n_good": n_good,
                    "n_rejected_quality": n_raw - n_good,
                }
            )

        group = all_band_rows.loc[all_band_rows["atlas_good"].fillna(False)].copy()
        time = _full_jd(pd.to_numeric(group["atlas_mjd"], errors="coerce"))
        flux_ujy = pd.to_numeric(group["atlas_flux_ujy"], errors="coerce").to_numpy(dtype=float)
        error_ujy = pd.to_numeric(
            group["atlas_flux_error_formal_ujy"], errors="coerce"
        ).to_numpy(dtype=float)
        valid = (
            np.isfinite(time)
            & np.isfinite(flux_ujy)
            & np.isfinite(error_ujy)
            & (flux_ujy > 0.0)
            & (error_ujy > 0.0)
        )
        if int(valid.sum()) < int(min_points):
            continue
        reference_flux_ujy = float(np.nanmedian(flux_ujy[valid]))
        if not np.isfinite(reference_flux_ujy) or reference_flux_ujy <= 0.0:
            continue
        valid_indices = np.flatnonzero(valid)
        order = np.argsort(time[valid])
        selected = valid_indices[order]
        point_metadata = {
            "atlas_row_id": group["atlas_row_id"].to_numpy(dtype=np.int64)[selected],
            "atlas_mjd": group["atlas_mjd"].to_numpy(dtype=float)[selected],
            "atlas_filter": group["atlas_filter"].astype(str).to_numpy()[selected],
            "atlas_chi_n": group["atlas_chi_n"].to_numpy(dtype=float)[selected],
            "atlas_obs": group["atlas_obs"].astype(str).to_numpy()[selected],
            "atlas_obs_site_code": group["atlas_obs_site_code"].astype(str).to_numpy()[selected],
            "atlas_obs_site": group["atlas_obs_site"].astype(str).to_numpy()[selected],
            "atlas_camera": group["atlas_camera"].astype(str).to_numpy()[selected],
            "atlas_flux_ujy": flux_ujy[selected],
            "atlas_flux_error_formal_ujy": error_ujy[selected],
            "atlas_original_ujy": flux_ujy[selected],
            "atlas_original_dujy": error_ujy[selected],
            "atlas_faq_good": group["atlas_faq_good"].to_numpy(dtype=bool)[selected],
            "atlas_reduced_good": group["atlas_reduced_good"].to_numpy(dtype=bool)[selected],
            "atlas_filter_good": group["atlas_filter_good"].to_numpy(dtype=bool)[selected],
            "atlas_flux_good": group["atlas_flux_good"].to_numpy(dtype=bool)[selected],
            "atlas_snr_good": group["atlas_snr_good"].to_numpy(dtype=bool)[selected],
            "atlas_good": group["atlas_good"].to_numpy(dtype=bool)[selected],
            "atlas_reject_reason": group["atlas_reject_reason"].astype(str).to_numpy()[selected],
        }
        datasets.append(
            PhotometryDataset(
                dataset_id=f"atlas:{band_name}:forced",
                survey="atlas",
                band=str(band_name),
                instrument="forced",
                time_jd=time[selected],
                flux=flux_ujy[selected] / reference_flux_ujy,
                flux_error=error_ujy[selected] / reference_flux_ujy,
                flux_kind="direct",
                provenance_path=str(path),
                reference_mag=float(23.9 - 2.5 * np.log10(reference_flux_ujy)),
                metadata={
                    "atlas_reference_flux_ujy": reference_flux_ujy,
                    "atlas_n_raw": int(len(all_band_rows)),
                    "atlas_n_good": int(valid.sum()),
                    "atlas_n_rejected_quality": int(len(all_band_rows) - valid.sum()),
                    "atlas_counts_by_site": counts_by_site,
                    "atlas_noise_status": "not_calibrated",
                },
                point_metadata=point_metadata,
            )
        )
    return datasets


def calibrate_atlas_datasets(
    datasets: Iterable[PhotometryDataset],
    *,
    t0_jd: float,
    u0: float,
    tE_days: float,
    event_mask_tau: float = 3.0,
    min_calibration_points: int = 30,
    min_time_span_days: float = 30.0,
    huber_tuning: float = 5.0,
    min_points: int = 5,
) -> list[PhotometryDataset]:
    """Apply shared ATLAS floors and fixed robust weights for one event.

    The ASAS-SN geometry only defines the masked event window and a provisional
    PSPL residual model. It is not imposed on the subsequent joint fit. ATLAS
    passbands without enough quiescent coverage are omitted instead of being
    assigned an arbitrary floor.
    """
    if not (
        np.isfinite(t0_jd)
        and np.isfinite(u0)
        and float(u0) > 0.0
        and np.isfinite(tE_days)
        and float(tE_days) > 0.0
    ):
        raise ValueError("A finite positive ASAS-SN PSPL geometry is required for ATLAS calibration")

    calibrated: list[PhotometryDataset] = []
    for dataset in datasets:
        if dataset.survey != "atlas":
            calibrated.append(dataset)
            continue

        required_metadata = {
            "atlas_mjd",
            "atlas_filter",
            "atlas_obs_site_code",
            "atlas_obs_site",
            "atlas_chi_n",
            "atlas_flux_ujy",
            "atlas_flux_error_formal_ujy",
        }
        missing = sorted(required_metadata - set(dataset.point_metadata))
        if missing:
            raise ValueError(
                f"ATLAS dataset {dataset.dataset_id} is missing point metadata: {missing}"
            )

        frame = pd.DataFrame(dataset.point_metadata)
        frame["atlas_good"] = True
        calibration_mask = (
            np.abs((dataset.time_jd - float(t0_jd)) / float(tE_days))
            >= float(event_mask_tau)
        )
        noise_model = estimate_atlas_noise_model(
            frame,
            calibration_mask=calibration_mask,
            min_points=int(min_calibration_points),
            min_time_span_days=float(min_time_span_days),
        )
        frame = apply_atlas_noise_model(frame, noise_model)
        usable = (
            frame["atlas_noise_model_usable"].fillna(False).to_numpy(dtype=bool)
            & np.isfinite(frame["atlas_flux_error_eff_ujy"].to_numpy(dtype=float))
        )
        if int(np.sum(usable)) < int(min_points):
            continue

        reference_flux_ujy = float(dataset.metadata.get("atlas_reference_flux_ujy", np.nan))
        if not np.isfinite(reference_flux_ujy) or reference_flux_ujy <= 0.0:
            raise ValueError(f"ATLAS dataset {dataset.dataset_id} has no valid flux reference")

        magnification = pspl_magnification(
            dataset.time_jd,
            t0=float(t0_jd),
            u0=float(u0),
            tE=float(tE_days),
        )
        effective_ujy = frame["atlas_flux_error_eff_ujy"].to_numpy(dtype=float)
        provisional = solve_linear_flux_parameters(
            magnification[usable],
            dataset.flux[usable],
            effective_ujy[usable] / reference_flux_ujy,
            flux_kind="direct",
        )
        robust_weights = np.zeros(dataset.n_points, dtype=float)
        if provisional.success:
            provisional_model = (
                provisional.source_flux * magnification + provisional.blend_flux
            )
            provisional_residual_ujy = (
                dataset.flux - provisional_model
            ) * reference_flux_ujy
            robust_weights[usable] = atlas_huber_weights(
                provisional_residual_ujy[usable],
                effective_ujy[usable],
                tuning=float(huber_tuning),
            )
            robust_reference = "asassn_geometry_profiled_flux"
        else:
            locations = frame["atlas_noise_location_ujy"].to_numpy(dtype=float)
            provisional_residual_ujy = (
                frame["atlas_flux_ujy"].to_numpy(dtype=float) - locations
            )
            robust_weights[usable] = atlas_huber_weights(
                provisional_residual_ujy[usable],
                effective_ujy[usable],
                tuning=float(huber_tuning),
            )
            robust_reference = "quiescent_group_location"

        fit_error_ujy = np.full(dataset.n_points, np.nan, dtype=float)
        positive_weight = usable & np.isfinite(robust_weights) & (robust_weights > 0.0)
        fit_error_ujy[positive_weight] = (
            effective_ujy[positive_weight] / np.sqrt(robust_weights[positive_weight])
        )
        retain = positive_weight & np.isfinite(fit_error_ujy) & (fit_error_ujy > 0.0)
        if int(np.sum(retain)) < int(min_points):
            continue

        site_codes = frame["atlas_obs_site_code"].fillna("").astype(str).to_numpy()
        site_names = frame["atlas_obs_site"].fillna("unknown").astype(str).to_numpy()
        calibration_counts_by_site: list[dict[str, object]] = []
        for site_code in sorted(set(site_codes)):
            site_mask = site_codes == site_code
            names = site_names[site_mask]
            calibration_counts_by_site.append(
                {
                    "atlas_obs_site_code": site_code,
                    "atlas_obs_site": str(names[0]) if names.size else "unknown",
                    "n_loaded_good": int(np.sum(site_mask)),
                    "n_calibration": int(np.sum(site_mask & calibration_mask)),
                    "n_noise_usable": int(np.sum(site_mask & usable)),
                    "n_retained_fit": int(np.sum(site_mask & retain)),
                    "n_downweighted": int(
                        np.sum(site_mask & (robust_weights > 0.0) & (robust_weights < 1.0))
                    ),
                    "n_excluded_noise_or_robust": int(np.sum(site_mask & ~retain)),
                }
            )

        calibrated_metadata = dict(dataset.metadata)
        calibrated_metadata.update(
            {
                "atlas_noise_model_version": ATLAS_NOISE_MODEL_VERSION,
                "atlas_noise_status": "ok",
                "atlas_noise_model_records": noise_model.to_dict(orient="records"),
                "atlas_event_mask_tau": float(event_mask_tau),
                "atlas_n_calibration": int(np.sum(calibration_mask)),
                "atlas_n_noise_usable": int(np.sum(usable)),
                "atlas_n_noise_unusable": int(dataset.n_points - np.sum(usable)),
                "atlas_huber_tuning": float(huber_tuning),
                "atlas_robust_reference": robust_reference,
                "atlas_n_downweighted": int(np.sum((robust_weights > 0.0) & (robust_weights < 1.0))),
                "atlas_n_excluded_robust": int(np.sum(usable & ~positive_weight)),
                "atlas_calibration_counts_by_site": calibration_counts_by_site,
            }
        )
        point_metadata = {
            name: np.asarray(values)[retain]
            for name, values in dataset.point_metadata.items()
        }
        point_metadata.update(
            {
                "atlas_calibration_mask": calibration_mask[retain],
                "atlas_noise_model_usable": usable[retain],
                "atlas_noise_status": frame["atlas_noise_status"].astype(str).to_numpy()[retain],
                "atlas_noise_scope": frame["atlas_noise_scope"].astype(str).to_numpy()[retain],
                "atlas_noise_group": frame["atlas_noise_group"].astype(str).to_numpy()[retain],
                "atlas_noise_location_ujy": frame["atlas_noise_location_ujy"].to_numpy(dtype=float)[retain],
                "atlas_noise_floor_ujy": frame["atlas_noise_floor_ujy"].to_numpy(dtype=float)[retain],
                "atlas_flux_error_eff_ujy": effective_ujy[retain],
                "atlas_robust_weight": robust_weights[retain],
                "atlas_fit_error_ujy": fit_error_ujy[retain],
            }
        )
        calibrated.append(
            PhotometryDataset(
                dataset_id=dataset.dataset_id,
                survey=dataset.survey,
                band=dataset.band,
                instrument=dataset.instrument,
                time_jd=dataset.time_jd[retain],
                flux=dataset.flux[retain],
                flux_error=fit_error_ujy[retain] / reference_flux_ujy,
                flux_kind=dataset.flux_kind,
                provenance_path=dataset.provenance_path,
                reference_mag=dataset.reference_mag,
                metadata=calibrated_metadata,
                point_metadata=point_metadata,
            )
        )
    return calibrated


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
