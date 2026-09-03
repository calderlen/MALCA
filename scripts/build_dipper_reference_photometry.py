#!/usr/bin/env python3
"""Build a plot-ready infrared catalog for named dipper reference sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroquery.gaia import Gaia
from astroquery.ipac.irsa import Irsa

from malca.enrichment.astrometry import angular_separation_arcsec, propagate_linear_icrs
from malca.enrichment.characterize import get_dust_extinction
from malca.plotting.extinction import add_dereddened_ir_magnitudes, dereddened_color
from malca.plotting.irac import irac_vega_magnitude, irac_vega_magnitude_error
from malca.review.sed import (
    SED_FETCH_STATUS_ATTR,
    query_irsa_allwise_photometry,
    query_irsa_spitzer_photometry,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ROOT / "input/dipper_color_reference_sources/reference_sources.json"
DEFAULT_OUTPUT_DIR = ROOT / "output/dipper_color_reference_sources"
MATCH_RADIUS_ARCSEC = 3.0


def _targets(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    frame = pd.DataFrame(payload["sources"])
    frame["candidate_id"] = frame["source_key"]
    frame["ra"] = pd.to_numeric(frame.pop("fallback_ra_deg"), errors="coerce")
    frame["dec"] = pd.to_numeric(frame.pop("fallback_dec_deg"), errors="coerce")
    frame["ref_epoch"] = pd.to_numeric(
        frame.pop("fallback_ref_epoch_jyear"), errors="coerce"
    )
    frame["coordinate_source"] = "reference_registry"
    return frame


def _query_gaia(frame: pd.DataFrame) -> pd.DataFrame:
    ids = [str(value) for value in frame["gaia_dr3_source_id"].dropna()]
    if not ids:
        return frame
    query = f"""
        SELECT
            g.source_id, g.ra, g.dec, g.ref_epoch,
            g.parallax, g.parallax_error,
            g.pmra, g.pmra_error, g.pmdec, g.pmdec_error,
            g.ruwe, g.phot_g_mean_mag,
            ap.distance_gspphot, ap.distance_gspphot_lower, ap.distance_gspphot_upper
        FROM gaiadr3.gaia_source AS g
        LEFT OUTER JOIN gaiadr3.astrophysical_parameters AS ap
            ON g.source_id = ap.source_id
        WHERE g.source_id IN ({','.join(ids)})
    """
    result = Gaia.launch_job(query).get_results().to_pandas()
    result.columns = [str(column).lower() for column in result.columns]
    result["gaia_dr3_source_id"] = result.pop("source_id").astype("int64").astype(str)
    numeric = [column for column in result.columns if column != "gaia_dr3_source_id"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")

    out = frame.merge(result, on="gaia_dr3_source_id", how="left", suffixes=("", "_gaia"))
    matched = out["ra_gaia"].notna() & out["dec_gaia"].notna()
    for column in ("ra", "dec", "ref_epoch"):
        out.loc[matched, column] = out.loc[matched, f"{column}_gaia"]
    out.loc[matched, "coordinate_source"] = "Gaia DR3"
    out["gaia_query_status"] = np.where(matched, "matched", "not_matched")
    return out.drop(columns=["ra_gaia", "dec_gaia", "ref_epoch_gaia"])


def _query_position(row: pd.Series, epoch_jyear: float) -> tuple[float, float]:
    ra, dec, _ = propagate_linear_icrs(
        float(row["ra"]),
        float(row["dec"]),
        Time(float(epoch_jyear), format="jyear").mjd,
        pmra_mas_per_year=pd.to_numeric(pd.Series([row.get("pmra")]), errors="coerce").iloc[0],
        pmdec_mas_per_year=pd.to_numeric(pd.Series([row.get("pmdec")]), errors="coerce").iloc[0],
        reference_epoch_jyear=float(row.get("ref_epoch", 2016.0)),
    )
    return float(ra), float(dec)


def _clean_2mass_id(value: object) -> str:
    text = str(value or "").strip().upper().replace("2MASS", "").replace("_", "")
    return text[1:] if text.startswith("J") else text


def _char(value: object, index: int) -> str:
    text = str(value or "").strip()
    return text[index] if index < len(text) else ""


def _query_2mass(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    columns = ",".join(
        [
            "designation", "ra", "dec",
            "j_m", "j_cmsig", "j_snr",
            "h_m", "h_cmsig", "h_snr",
            "k_m", "k_cmsig", "k_snr",
            "ph_qual", "rd_flg", "bl_flg", "cc_flg",
            "prox", "gal_contam", "mp_flg", "use_src",
        ]
    )
    for _, target in frame.iterrows():
        record: dict[str, object] = {"candidate_id": target["candidate_id"]}
        try:
            target_ra, target_dec = _query_position(target, 2000.0)
            table = Irsa.query_region(
                SkyCoord(target_ra * u.deg, target_dec * u.deg),
                catalog="fp_psc",
                radius=MATCH_RADIUS_ARCSEC * u.arcsec,
                columns=columns,
            )
            matches = table.to_pandas() if hasattr(table, "to_pandas") else pd.DataFrame(table)
            if matches.empty:
                record["tmass_query_status"] = "no_match"
                rows.append(record)
                continue
            separations = angular_separation_arcsec(
                target_ra,
                target_dec,
                pd.to_numeric(matches["ra"], errors="coerce").to_numpy(float),
                pd.to_numeric(matches["dec"], errors="coerce").to_numpy(float),
            )
            matches = matches.assign(_sep_arcsec=separations)
            expected = _clean_2mass_id(target.get("expected_2mass_id"))
            expected_mask = matches["designation"].map(_clean_2mass_id).eq(expected) if expected else None
            if expected and expected_mask is not None and expected_mask.any():
                selected = matches.loc[expected_mask].sort_values("_sep_arcsec").iloc[0]
                identity_status = "expected_id"
            else:
                selected = matches.sort_values("_sep_arcsec").iloc[0]
                identity_status = "nearest" if not expected else "expected_id_missing"

            ph_qual = str(selected.get("ph_qual", "") or "")
            cc_flg = str(selected.get("cc_flg", "") or "")
            record.update(
                {
                    "tmass_query_status": "matched",
                    "tmass_designation": f"J{_clean_2mass_id(selected.get('designation'))}",
                    "tmass_sep_arcsec": float(selected["_sep_arcsec"]),
                    "tmass_match_count": int(len(matches)),
                    "tmass_identity_status": identity_status,
                    "tmass_ph_qual": ph_qual,
                    "tmass_rd_flg": str(selected.get("rd_flg", "") or ""),
                    "tmass_bl_flg": str(selected.get("bl_flg", "") or ""),
                    "tmass_cc_flg": cc_flg,
                    "tmass_use_src": selected.get("use_src"),
                    "tmass_gal_contam": selected.get("gal_contam"),
                    "tmass_prox_arcsec": selected.get("prox"),
                }
            )
            for index, (band, prefix) in enumerate(
                (("j", "tmass_j"), ("h", "tmass_h"), ("k", "tmass_k"))
            ):
                magnitude = pd.to_numeric(pd.Series([selected.get(f"{band}_m")]), errors="coerce").iloc[0]
                error = pd.to_numeric(pd.Series([selected.get(f"{band}_cmsig")]), errors="coerce").iloc[0]
                snr = pd.to_numeric(pd.Series([selected.get(f"{band}_snr")]), errors="coerce").iloc[0]
                quality = _char(ph_qual, index).upper()
                artifact = _char(cc_flg, index)
                record[prefix] = magnitude
                record[f"{prefix}_err"] = error
                record[f"{prefix}_snr"] = snr
                record[f"{prefix}_quality"] = quality
                record[f"{prefix}_plot_ok"] = bool(
                    np.isfinite(magnitude)
                    and quality in {"A", "B", "C"}
                    and artifact in {"", "0"}
                    and identity_status != "expected_id_missing"
                )
        except Exception as exc:
            record["tmass_query_status"] = "query_error"
            record["tmass_query_error"] = str(exc)
        rows.append(record)
    return frame.merge(pd.DataFrame(rows), on="candidate_id", how="left")


def _pivot_sed_catalog(
    targets: pd.DataFrame,
    measurements: pd.DataFrame,
    *,
    prefix: str,
    bands: tuple[str, ...],
) -> pd.DataFrame:
    statuses = measurements.attrs.get(SED_FETCH_STATUS_ATTR, {})
    rows: list[dict[str, object]] = []
    for candidate_id in targets["candidate_id"]:
        record: dict[str, object] = {
            "candidate_id": candidate_id,
            f"{prefix}_query_status": statuses.get(candidate_id, "missing_status"),
        }
        for band in bands:
            name = band.lower()
            record[name] = np.nan
            record[f"{name}_err"] = np.nan
            record[f"{name}_plot_ok"] = False
        source_rows = measurements[measurements["candidate_id"].astype(str) == str(candidate_id)]
        for band in bands:
            selected = source_rows[source_rows["band"].astype(str).str.upper() == band]
            if selected.empty:
                continue
            item = selected.iloc[0]
            name = band.lower()
            if prefix == "wise":
                record[name] = item.get("mag")
                record[f"{name}_err"] = item.get("mag_err")
            else:
                flux = pd.to_numeric(pd.Series([item.get("flux_nu_jy")]), errors="coerce").iloc[0]
                flux_err = pd.to_numeric(pd.Series([item.get("flux_nu_jy_err")]), errors="coerce").iloc[0]
                record[f"{name}_flux_jy"] = flux
                record[f"{name}_flux_err_jy"] = flux_err
                record[name] = irac_vega_magnitude(np.array([flux]), band)[0]
                record[f"{name}_err"] = irac_vega_magnitude_error(
                    np.array([flux]), np.array([flux_err])
                )[0]
            flags = str(item.get("quality_flags", "") or "")
            record[f"{name}_quality_flags"] = flags
            record[f"{name}_plot_ok"] = bool(
                np.isfinite(pd.to_numeric(pd.Series([record.get(name)]), errors="coerce").iloc[0])
                and not bool(item.get("is_upper_limit", False))
                and "bad_quality" not in flags
                and "ambiguous_counterpart" not in flags
            )
            record[f"{prefix}_designation"] = item.get("source_object_id")
            record[f"{prefix}_sep_arcsec"] = item.get("sep_arcsec")
        rows.append(record)
    return targets.merge(pd.DataFrame(rows), on="candidate_id", how="left")


def _add_extinction(frame: pd.DataFrame) -> pd.DataFrame:
    dust_input = frame.copy()
    parallax = pd.to_numeric(dust_input.get("parallax"), errors="coerce")
    parallax_error = pd.to_numeric(dust_input.get("parallax_error"), errors="coerce")
    good_parallax = np.isfinite(parallax) & np.isfinite(parallax_error) & (parallax_error > 0)
    good_parallax &= (parallax / parallax_error) >= 5.0
    dust_input["gaia_parallax"] = parallax.where(good_parallax)
    try:
        return get_dust_extinction(dust_input)
    except Exception as exc:
        out = frame.copy()
        out["A_v_3d"] = np.nan
        out["ebv_3d"] = np.nan
        out["dust_status"] = "query_error"
        out["dust_query_error"] = str(exc)
        return out


def _hypot(frame: pd.DataFrame, first: str, second: str) -> pd.Series:
    return np.hypot(
        pd.to_numeric(frame.get(first), errors="coerce"),
        pd.to_numeric(frame.get(second), errors="coerce"),
    )


def _add_colors(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_dereddened_ir_magnitudes(frame)
    out["extinction_fallback_zero"] = ~np.isfinite(pd.to_numeric(out["A_v_3d"], errors="coerce"))
    color_specs = (
        ("tmass_h", "tmass_k", "H_K"),
        ("tmass_j", "tmass_h", "j_h"),
        ("tmass_j", "tmass_k", "j_k"),
        ("tmass_k", "w2", "ks_w2"),
        ("tmass_k", "w3", "ks_w3"),
        ("tmass_k", "w4", "ks_w4"),
        ("w1", "w2", "w1_w2"),
        ("w1", "w3", "w1_w3"),
        ("w1", "w4", "w1_w4"),
        ("w2", "w3", "w2_w3"),
        ("w2", "w4", "w2_w4"),
        ("w3", "w4", "w3_w4"),
    )
    for left, right, name in color_specs:
        out[name] = pd.to_numeric(out.get(left), errors="coerce") - pd.to_numeric(
            out.get(right), errors="coerce"
        )
        out[f"{name}_err"] = _hypot(out, f"{left}_err", f"{right}_err")
        out[f"{name}_0"] = dereddened_color(out, left, right)
        left_ok = (
            pd.to_numeric(out[f"{left}_plot_ok"], errors="coerce").fillna(0).astype(bool)
            if f"{left}_plot_ok" in out
            else pd.Series(False, index=out.index)
        )
        right_ok = (
            pd.to_numeric(out[f"{right}_plot_ok"], errors="coerce").fillna(0).astype(bool)
            if f"{right}_plot_ok" in out
            else pd.Series(False, index=out.index)
        )
        out[f"{name}_plot_ok"] = left_ok & right_ok

    out["w1_0"] = pd.to_numeric(out.get("w1_0"), errors="coerce")
    out["irac_36_45"] = out["irac1"] - out["irac2"]
    out["irac_36_45_err"] = _hypot(out, "irac1_err", "irac2_err")
    out["irac_58_80"] = out["irac3"] - out["irac4"]
    out["irac_58_80_err"] = _hypot(out, "irac3_err", "irac4_err")
    out["irac_color_plot_ok"] = (
        out[[f"irac{number}_plot_ok" for number in range(1, 5)]]
        .fillna(False)
        .all(axis=1)
    )
    return out


def build_catalog(target_path: Path, output_dir: Path) -> pd.DataFrame:
    targets = _query_gaia(_targets(target_path))
    targets = _query_2mass(targets)

    sed_input = targets[["candidate_id", "ra", "dec", "pmra", "pmdec", "ref_epoch"]].copy()
    wise_long = query_irsa_allwise_photometry(sed_input, progress_callback=print)
    spitzer_long = query_irsa_spitzer_photometry(sed_input, progress_callback=print)
    targets = _pivot_sed_catalog(
        targets, wise_long, prefix="wise", bands=("W1", "W2", "W3", "W4")
    )
    targets = _pivot_sed_catalog(
        targets, spitzer_long, prefix="spitzer", bands=("IRAC1", "IRAC2", "IRAC3", "IRAC4")
    )
    catalog = _add_colors(_add_extinction(targets))

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dipper_reference_photometry.csv"
    parquet_path = output_dir / "dipper_reference_photometry.parquet"
    sed_path = output_dir / "dipper_reference_sed_photometry.parquet"
    catalog.to_csv(csv_path, index=False)
    catalog.to_parquet(parquet_path, index=False)
    pd.concat([wise_long, spitzer_long], ignore_index=True).to_parquet(sed_path, index=False)
    print(f"Saved {csv_path} ({len(catalog)} sources)")
    print(f"Saved {parquet_path}")
    print(f"Saved {sed_path}")
    return catalog


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_catalog(args.targets, args.output_dir)
