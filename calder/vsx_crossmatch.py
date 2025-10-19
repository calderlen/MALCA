from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

DEFAULT_ASASSN_PATH = Path(
    "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/"
    "asassn_index_masked_concat_cleaned_20250919_154524.csv"
)
DEFAULT_VSX_PATH = Path(
    "/data/poohbah/1/assassin/lenhart/code/calder/calder/output/"
    "vsx_cleaned_20250919_154524.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/poohbah/1/assassin/lenhart/code/calder/calder/output"
)


def load_asassn_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in ["ra_deg", "dec_deg", "pm_ra", "pm_dec"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    pm_ok = np.isfinite(df["pm_ra"].to_numpy()) & np.isfinite(df["pm_dec"].to_numpy())
    if not pm_ok.all():
        bad = (~pm_ok).sum()
        sample = df.loc[~pm_ok, ["asas_sn_id", "gaia_id", "pm_ra", "pm_dec"]].head(10)
        raise ValueError(
            f"{bad} row(s) missing/invalid proper motion.\nSample:\n{sample}"
        )
    return df


def load_vsx_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in ["ra", "dec"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def propagate_asassn_coords(
    df: pd.DataFrame, epoch_from=2016.0, epoch_to=2000.0
) -> SkyCoord:
    coord = SkyCoord(
        ra=df["ra_deg"].to_numpy(dtype=float) * u.deg,
        dec=df["dec_deg"].to_numpy(dtype=float) * u.deg,
        pm_ra_cosdec=df["pm_ra"].to_numpy(dtype=float) * u.mas / u.yr,
        pm_dec=df["pm_dec"].to_numpy(dtype=float) * u.mas / u.yr,
        obstime=Time(epoch_from, format="jyear"),
    )
    return coord.apply_space_motion(new_obstime=Time(epoch_to, format="jyear"))


def vsx_coords(df: pd.DataFrame) -> SkyCoord:
    return SkyCoord(
        ra=df["ra"].to_numpy(dtype=float) * u.deg,
        dec=df["dec"].to_numpy(dtype=float) * u.deg,
    )


def crossmatch_asassn_vsx(
    asassn_csv: Path = DEFAULT_ASASSN_PATH,
    vsx_csv: Path = DEFAULT_VSX_PATH,
    match_radius: u.Quantity = 3 * u.arcsec,
) -> pd.DataFrame:
    
    """
    Return a df of ASAS-SN and VSX matches within match_radius
    """

    df_asassn = load_asassn_catalog(Path(asassn_csv))
    df_vsx = load_vsx_catalog(Path(vsx_csv))

    coords_asassn = propagate_asassn_coords(df_asassn)
    coords_vsx = vsx_coords(df_vsx)

    idx_vsx, sep2d, _ = coords_asassn.match_to_catalog_sky(coords_vsx)
    mask = sep2d < match_radius

    df_pairs = pd.DataFrame(
        {
            "targ_idx": np.where(mask)[0],
            "vsx_idx": idx_vsx[mask],
            "sep_arcsec": sep2d[mask].to(u.arcsec).value,
        }
    )

    return (
        df_pairs.merge(
            df_asassn, left_on="targ_idx", right_index=True, how="left"
        ).merge(
            df_vsx,
            left_on="vsx_idx",
            right_index=True,
            how="left",
            suffixes=("_targ", "_vsx"),
        )
    )


def write_crossmatch(
    matches: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    stamp: str | None = None,
) -> Path:
    """
    Write matches to a timestamped CSV and return the path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M")
    path = output_dir / f"asassn_x_vsx_matches_{stamp}.csv"
    matches.to_csv(path, index=False)
    return path


def main(
    asassn_csv: Path = DEFAULT_ASASSN_PATH,
    vsx_csv: Path = DEFAULT_VSX_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    match_radius: u.Quantity = 3 * u.arcsec,
) -> Path:
    matches = crossmatch_asassn_vsx(asassn_csv, vsx_csv, match_radius=match_radius)
    return write_crossmatch(matches, output_dir=output_dir)


if __name__ == "__main__":
    OUTPUT_PATH = main()
    print(f"Wrote crossmatch catalog to {OUTPUT_PATH}")
