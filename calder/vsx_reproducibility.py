from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord


DEFAULT_VSX_CROSSMATCH = Path(
    "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/"
    "asassn_x_vsx_matches_20250920_1415.csv"
)
DEFAULT_EXISTING = Path(
    "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/bj_objects.csv"
)
DEFAULT_ASASSN_CATALOG = Path(
    "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/"
    "asassn_index_masked_concat_cleaned_20250920_1351.csv"
)
DEFAULT_VSX_CLEAN = Path(
    "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/vsx_cleaned_20250920_1351.csv"
)
DEFAULT_VSX_RAW = Path(
    "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/vsx_raw_20250921_0408.csv"
)

DEFAULT_OUT_MATCH_VSX = Path(
    "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/bj_objects_matched.csv"
)
DEFAULT_OUT_UNMATCH_VSX = Path(
    "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/bj_objects_unmatched.csv"
)
DEFAULT_OUT_MATCH_ASASSN = Path(
    "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/bj_objects_matched_asassn.csv"
)
DEFAULT_OUT_MATCH_VSXCLEAN = Path(
    "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/bj_objects_x_vsxclean.csv"
)
DEFAULT_OUT_MATCH_VSXRAW = Path(
    "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/bj_objects_x_vsxraw.csv"
)

DEFAULT_TOL_ARCSEC = 2.0


def validate_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{label} CSV missing columns: {sorted(missing)}")


def coords(df: pd.DataFrame, ra_col: str, dec_col: str) -> SkyCoord:
    return SkyCoord(
        df[ra_col].to_numpy(dtype=float) * u.deg,
        df[dec_col].to_numpy(dtype=float) * u.deg,
        frame="icrs",
    )


def match_catalog(
    existing: pd.DataFrame,
    coords_existing: SkyCoord,
    target: pd.DataFrame,
    coords_target: SkyCoord,
    tag: str,
    tol_arcsec: float,
    cols_to_keep: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    idx, sep2d, _ = coords_existing.match_to_catalog_sky(coords_target)
    mask = sep2d < (tol_arcsec * u.arcsec)

    updated = existing.copy()
    updated[f"matched_{tag}"] = mask
    updated[f"match_sep_arcsec_{tag}"] = sep2d.arcsec
    updated[f"{tag}_index"] = idx

    matched = updated[updated[f"matched_{tag}"]].copy()
    if cols_to_keep:
        subset = (
            target[list(cols_to_keep)]
            .reset_index()
            .rename(columns={"index": f"{tag}_index"})
        )
        matched = matched.merge(
            subset,
            on=f"{tag}_index",
            how="left",
            suffixes=("", f"_{tag}"),
        )
        front = [
            col
            for col in [
                "category",
                "name",
                "ra_deg",
                "dec_deg",
                "method",
                f"matched_{tag}",
                f"match_sep_arcsec_{tag}",
            ]
            if col in matched.columns
        ]
        matched = matched[
            front
            + [
                col
                for col in matched.columns
                if col not in front + [f"{tag}_index"]
            ]
        ]

    return updated, matched, mask


def run_matches(
    existing_csv: Path | str = DEFAULT_EXISTING,
    vsx_crossmatch_csv: Path | str = DEFAULT_VSX_CROSSMATCH,
    asassn_csv: Path | str = DEFAULT_ASASSN_CATALOG,
    vsx_clean_csv: Path | str = DEFAULT_VSX_CLEAN,
    vsx_raw_csv: Path | str = DEFAULT_VSX_RAW,
    *,
    tol_arcsec: float = DEFAULT_TOL_ARCSEC,
    output_paths: Mapping[str, Path | str] | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    existing = pd.read_csv(existing_csv)
    vsx_cm = pd.read_csv(vsx_crossmatch_csv)
    asassn_df = pd.read_csv(asassn_csv)
    vsx_clean = pd.read_csv(vsx_clean_csv)
    vsx_raw = pd.read_csv(vsx_raw_csv)

    validate_columns(existing, {"ra_deg", "dec_deg"}, "existing")
    validate_columns(vsx_cm, {"ra_deg", "dec_deg"}, "vsx")
    validate_columns(asassn_df, {"ra_deg", "dec_deg"}, "asassn")
    validate_columns(vsx_clean, {"ra", "dec"}, "vsx_clean")
    validate_columns(vsx_raw, {"ra", "dec"}, "vsx_raw")

    coords_existing = coords(existing, "ra_deg", "dec_deg")
    coords_vsx_cm = coords(vsx_cm, "ra_deg", "dec_deg")
    coords_asassn = coords(asassn_df, "ra_deg", "dec_deg")
    coords_vsx_clean = coords(vsx_clean, "ra", "dec")
    coords_vsx_raw = coords(vsx_raw, "ra", "dec")

    outputs = {
        "match_vsx": DEFAULT_OUT_MATCH_VSX,
        "unmatch_vsx": DEFAULT_OUT_UNMATCH_VSX,
        "match_asassn": DEFAULT_OUT_MATCH_ASASSN,
        "match_vsxclean": DEFAULT_OUT_MATCH_VSXCLEAN,
        "match_vsxraw": DEFAULT_OUT_MATCH_VSXRAW,
    }
    if output_paths:
        for key, value in output_paths.items():
            outputs[key] = Path(value)

    vsx_cols = [
        "targ_idx",
        "vsx_idx",
        "sep_arcsec",
        "asas_sn_id",
        "ra_deg",
        "dec_deg",
        "gaia_id",
        "name",
        "var_flag",
        "class",
        "gaia_mag",
        "gaia_b_mag",
        "gaia_r_mag",
        "gaia_eff_temp",
        "gaia_g_extinc",
        "gaia_var",
        "pstarrs_g_mag",
        "pstarrs_r_mag",
        "pstarrs_i_mag",
    ]
    vsx_cols = [c for c in vsx_cols if c in vsx_cm.columns]

    asassn_cols = [
        "targ_idx",
        "asas_sn_id",
        "ra_deg",
        "dec_deg",
        "gaia_id",
        "name",
        "var_flag",
        "class",
        "gaia_mag",
        "gaia_b_mag",
        "gaia_r_mag",
        "gaia_eff_temp",
        "gaia_g_extinc",
        "gaia_var",
        "pstarrs_g_mag",
        "pstarrs_r_mag",
        "pstarrs_i_mag",
    ]
    asassn_cols = [c for c in asassn_cols if c in asassn_df.columns]

    vsxclean_cols = [
        "id_vsx",
        "name",
        "var_flag",
        "ra",
        "dec",
        "class",
        "l_max",
        "mag_max",
        "u_max",
        "mag_band_max",
        "f_min",
        "l_min",
        "mag_min",
        "u_min",
        "mag_band_min",
        "epoch",
        "u_epoch",
        "l_period",
        "period",
        "u_period",
        "spectral_type",
    ]
    vsxclean_cols = [c for c in vsxclean_cols if c in vsx_clean.columns]

    vsxraw_cols = [c for c in vsxclean_cols if c in vsx_raw.columns]

    existing, matched_vsx, mask_vsx = match_catalog(
        existing,
        coords_existing,
        vsx_cm,
        coords_vsx_cm,
        "vsx",
        tol_arcsec,
        cols_to_keep=vsx_cols,
    )
    unmatched_vsx = existing[~existing["matched_vsx"]].copy()

    existing, matched_asassn, mask_asassn = match_catalog(
        existing,
        coords_existing,
        asassn_df,
        coords_asassn,
        "asassn",
        tol_arcsec,
        cols_to_keep=asassn_cols,
    )

    existing, matched_vsxclean, mask_vsxclean = match_catalog(
        existing,
        coords_existing,
        vsx_clean,
        coords_vsx_clean,
        "vsxclean",
        tol_arcsec,
        cols_to_keep=vsxclean_cols,
    )

    existing, matched_vsxraw, mask_vsxraw = match_catalog(
        existing,
        coords_existing,
        vsx_raw,
        coords_vsx_raw,
        "vsxraw",
        tol_arcsec,
        cols_to_keep=vsxraw_cols,
    )

    outputs["match_vsx"].parent.mkdir(parents=True, exist_ok=True)
    matched_vsx.to_csv(outputs["match_vsx"], index=False)
    unmatched_vsx.to_csv(outputs["unmatch_vsx"], index=False)
    matched_asassn.to_csv(outputs["match_asassn"], index=False)
    matched_vsxclean.to_csv(outputs["match_vsxclean"], index=False)
    matched_vsxraw.to_csv(outputs["match_vsxraw"], index=False)

    if verbose:
        total = len(existing)
        print(f"VSX: matched {int(mask_vsx.sum())} / {total} within {tol_arcsec}\"")
        if "category" in existing.columns:
            print(existing.groupby("category")["matched_vsx"].agg(["sum", "count"]))

        print(f"ASASSN: matched {int(mask_asassn.sum())} / {total} within {tol_arcsec}\"")
        if "category" in existing.columns:
            print(existing.groupby("category")["matched_asassn"].agg(["sum", "count"]))

        print(
            f"VSX-cleaned: matched {int(mask_vsxclean.sum())} / {total} "
            f"within {tol_arcsec}\""
        )
        if "category" in existing.columns:
            print(
                existing.groupby("category")["matched_vsxclean"].agg(["sum", "count"])
            )

        print(
            f"VSX-raw: matched {int(mask_vsxraw.sum())} / {total} within {tol_arcsec}\""
        )
        if "category" in existing.columns:
            print(
                existing.groupby("category")["matched_vsxraw"].agg(["sum", "count"])
            )

        print(
            "Wrote:",
            outputs["match_vsx"], "(VSX matches),",
            outputs["unmatch_vsx"], "(VSX non-matches),",
            outputs["match_asassn"], "(ASASSN matches),",
            outputs["match_vsxclean"], "(VSX-cleaned matches),",
            outputs["match_vsxraw"], "(VSX-raw matches)",
        )

    return {key: Path(value) for key, value in outputs.items()}


def main() -> dict[str, Path]:
    return run_matches()


if __name__ == "__main__":
    main()
