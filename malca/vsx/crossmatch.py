from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import numpy as np
import pandas as pd





DEFAULT_ASASSN_PATH = Path(__file__).parent.parent.parent / "input" / "vsx" / "asassn_catalog.parquet"
DEFAULT_VSX_PATH = Path(__file__).parent.parent.parent / "input" / "vsx" / "vsx_cleaned.parquet"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "input" / "vsx"


def load_asassn_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
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
    df = pd.read_parquet(path)
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
    asassn_table: Path = DEFAULT_ASASSN_PATH,
    vsx_table: Path = DEFAULT_VSX_PATH,
    match_radius: u.Quantity = 3 * u.arcsec,
) -> pd.DataFrame:
    
    """
    Return a df of ASAS-SN and VSX matches within match_radius
    """

    df_asassn = load_asassn_catalog(Path(asassn_table))
    df_vsx = load_vsx_catalog(Path(vsx_table))

    coords_asassn = propagate_asassn_coords(df_asassn)
    coords_vsx = vsx_coords(df_vsx)

    idx_vsx, sep2d, _ = coords_asassn.match_to_catalog_sky(coords_vsx)
    mask = sep2d < match_radius

    df_pairs = pd.DataFrame(
        {
            "targ_idx": np.where(mask)[0],
            "vsx_idx": idx_vsx[mask],
            "vsx_sep_arcsec": sep2d[mask].to(u.arcsec).value,
        }
    )

    out = (
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
    if "class" in out.columns and "vsx_class" not in out.columns:
        out = out.rename(columns={"class": "vsx_class"})
    return out


def write_crossmatch(
    matches: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    stamp: str | None = None,
) -> Path:
    """
    Write matches to a timestamped Parquet and return the path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M")
    path = output_dir / f"asassn_x_vsx_matches_{stamp}.parquet"
    matches.to_parquet(path, index=False)
    return path


def main(
    asassn_table: Path = DEFAULT_ASASSN_PATH,
    vsx_table: Path = DEFAULT_VSX_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    match_radius: u.Quantity = 3 * u.arcsec,
) -> Path:
    matches = crossmatch_asassn_vsx(asassn_table, vsx_table, match_radius=match_radius)
    return write_crossmatch(matches, output_dir=output_dir)


def cli():
    """CLI entry point for ``malca vsx-crossmatch``."""


    parser = argparse.ArgumentParser(
        description="Crossmatch ASAS-SN catalog with VSX catalog."
    )
    parser.add_argument("--asassn-table", type=Path, default=DEFAULT_ASASSN_PATH,
                        help=f"Cleaned ASAS-SN index Parquet (default: {DEFAULT_ASASSN_PATH})")
    parser.add_argument("--vsx-table", type=Path, default=DEFAULT_VSX_PATH,
                        help=f"Cleaned VSX catalog Parquet (default: {DEFAULT_VSX_PATH})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--radius", type=float, default=3.0,
                        help="Match radius in arcseconds (default: 3.0)")
    parser.add_argument("--stamp", type=str, default=None,
                        help="Timestamp suffix for output filename (default: current time)")
    args = parser.parse_args()

    matches = crossmatch_asassn_vsx(args.asassn_table, args.vsx_table,
                                     match_radius=args.radius * u.arcsec)
    path = write_crossmatch(matches, output_dir=args.output_dir, stamp=args.stamp)
    print(f"Wrote crossmatch catalog to {path}")


if __name__ == "__main__":
    cli()
