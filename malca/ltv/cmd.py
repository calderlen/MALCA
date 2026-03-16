"""
CMD (color-magnitude diagram) utilities for LTV candidates.

Scaffolding for:
- Extinction-corrected BP-RP and M_G
- MIST isochrone loading (vendored grid)
- Group assignment (rules supplied later)
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from astropy.table import Table
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import pyvo

from malca.config.config_ltv import (
    CMD_R_V,
    CMD_A_G_PER_AV,
    CMD_E_BP_RP_PER_AV,
    LTV_GAIA_CHUNK_SIZE,
    LTV_WORKERS,
)
from malca.config.config_paths import GAIA_AIP_TAP_URL, MIST_GRID_PATH








DEFAULT_MIST_PATH = MIST_GRID_PATH


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_mist_grid(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load a vendored MIST isochrone grid for CMD overlays.

    Expected location: input/mist/ (checked in)
    """
    grid_path = Path(path) if path is not None else DEFAULT_MIST_PATH
    if not grid_path.exists():
        raise FileNotFoundError(
            f"MIST grid not found: {grid_path} (vendor the grid under input/mist/)"
        )
    if grid_path.suffix == ".parquet":
        return pd.read_parquet(grid_path)
    return pd.read_csv(grid_path)


def fetch_bailer_jones_distances(
    df: pd.DataFrame,
    *,
    source_id_col: str | None = None,
    chunk_size: int = LTV_GAIA_CHUNK_SIZE,
    n_workers: int = LTV_WORKERS,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Fetch Bailer-Jones et al. (2023) distances from the Gaia TAP service.

    Queries ``external.gaiaedr3_distance`` (available on the Gaia AIP TAP)
    for every source with a known Gaia DR3 source_id and retrieves:

      - ``bj_r_med_photogeo``: median photogeometric distance (pc)
      - ``bj_r_med_geo``: median geometric distance (pc)

    Uses TAP_UPLOAD for batch efficiency — one async job per chunk.
    Distances are matched by source_id; sources without a BJ entry get NaN.
    """




    if df.empty:
        return df

    df = df.copy()
    df["bj_r_med_photogeo"] = np.nan
    df["bj_r_med_geo"] = np.nan

    src_col = source_id_col or _first_existing_column(
        df, ["source_id", "gaia_source_id", "gaia_dr3_source_id"]
    )
    if src_col is None:
        if verbose:
            print("Warning: no source_id column found; skipping Bailer-Jones query")
        return df

    valid_mask = df[src_col].notna()
    source_ids = df.loc[valid_mask, src_col].astype(np.int64)
    if len(source_ids) == 0:
        return df

    tap = pyvo.dal.TAPService(GAIA_AIP_TAP_URL)

    adql = """
        SELECT t.source_id, bj.r_med_photogeo, bj.r_med_geo
        FROM TAP_UPLOAD.t AS t
        JOIN external.gaiaedr3_distance AS bj
          ON t.source_id = bj.source_id
    """

    chunks = [
        source_ids.iloc[i : i + chunk_size]
        for i in range(0, len(source_ids), chunk_size)
    ]

    def _query_chunk(chunk: pd.Series) -> pd.DataFrame:
        upload = Table({"source_id": chunk.values.astype(np.int64)})
        job = tap.run_sync(adql, uploads={"t": upload})
        return job.to_table().to_pandas()

    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_query_chunk, ch): ch for ch in chunks}
        it = tqdm(as_completed(futures), total=len(futures), desc="BJ distances", disable=not verbose)
        for fut in it:
            try:
                results.append(fut.result())
            except Exception as exc:
                if verbose:
                    tqdm.write(f"[fetch_bailer_jones_distances] chunk failed: {exc}")

    if not results:
        return df

    combined = pd.concat(results, ignore_index=True)
    if combined.empty:
        return df

    combined = combined.set_index("source_id")
    n_matched = 0
    for df_idx, src_id in source_ids.items():
        if src_id in combined.index:
            row = combined.loc[src_id]
            df.loc[df_idx, "bj_r_med_photogeo"] = row["r_med_photogeo"]
            df.loc[df_idx, "bj_r_med_geo"] = row["r_med_geo"]
            n_matched += 1

    if verbose:
        print(
            f"[fetch_bailer_jones_distances] {n_matched}/{len(source_ids)} sources matched"
        )

    return df


def compute_cmd_features(
    df: pd.DataFrame,
    *,
    g_col: str | None = None,
    bp_col: str | None = None,
    rp_col: str | None = None,
    distance_pc_col: str | None = None,
    parallax_mas_col: str | None = None,
    av_col: str = "A_v_3d",
    r_v: float = CMD_R_V,
    a_g_per_av: float = CMD_A_G_PER_AV,
    e_bp_rp_per_av: float = CMD_E_BP_RP_PER_AV,
) -> pd.DataFrame:
    """
    Compute CMD quantities (BP-RP, M_G), with optional extinction correction.

    Uses A_v_3d if present to compute:
      A_G = a_g_per_av * A_V
      E(BP-RP) = e_bp_rp_per_av * A_V

    Adds columns:
      - bp_rp, bp_rp0
      - M_G, M_G0
      - distance_pc (if derived from parallax)
    """
    if df.empty:
        return df

    df = df.copy()

    g_col = g_col or _first_existing_column(df, ["gaia_phot_g_mean_mag", "phot_g_mean_mag", "G", "g_mag"])
    bp_col = bp_col or _first_existing_column(df, ["gaia_bp_mag", "phot_bp_mean_mag", "BP", "bp_mag"])
    rp_col = rp_col or _first_existing_column(df, ["gaia_rp_mag", "phot_rp_mean_mag", "RP", "rp_mag"])

    if g_col is None or bp_col is None or rp_col is None:
        return df

    # Distance in parsec — prefer Bailer-Jones photogeometric, then geometric,
    # then Gaia GSP-Phot, then a pre-existing distance_pc column.
    dist_col = distance_pc_col or _first_existing_column(
        df,
        [
            "bj_r_med_photogeo",
            "bj_r_med_geo",
            "distance_gspphot",
            "distance_pc",
            "gaia_distance_pc",
        ],
    )
    parallax_col = parallax_mas_col or _first_existing_column(
        df, ["gaia_parallax", "parallax"]
    )

    if dist_col is not None:
        dist_pc = df[dist_col].astype(float)
        dist_pc = dist_pc.where(dist_pc > 0, np.nan)
    elif parallax_col is not None:
        plx = df[parallax_col].astype(float)
        dist_pc = pd.Series(np.where(plx > 0, 1000.0 / plx, np.nan), index=df.index)
        df["distance_pc"] = dist_pc
    else:
        return df

    # Observed colors/magnitudes
    bp = df[bp_col].astype(float)
    rp = df[rp_col].astype(float)
    g = df[g_col].astype(float)

    df["bp_rp"] = bp - rp
    df["M_G"] = g - 5.0 * np.log10(dist_pc) + 5.0

    # Extinction correction if available
    if av_col in df.columns:
        av = df[av_col].astype(float)
        a_g = a_g_per_av * av
        e_bp_rp = e_bp_rp_per_av * av

        df["bp_rp0"] = df["bp_rp"] - e_bp_rp
        df["M_G0"] = df["M_G"] - a_g
        df["A_G"] = a_g
        df["E_bp_rp"] = e_bp_rp
        df["R_V"] = r_v

    return df


def assign_cmd_groups(
    df: pd.DataFrame,
    *,
    boundaries: dict | None = None,
    cmd_color_col: str = "bp_rp0",
    cmd_mag_col: str = "M_G0",
) -> pd.DataFrame:
    """
    Assign CMD groups based on provided boundary rules.

    If boundaries is None, adds a placeholder cmd_group column with None.
    """
    if df.empty:
        return df

    df = df.copy()

    if boundaries is None:
        df["cmd_group"] = None
        df["cmd_group_source"] = "unassigned"
        return df

    if cmd_color_col not in df.columns or cmd_mag_col not in df.columns:
        return df

    # Placeholder for future rule-based assignment
    df["cmd_group"] = None
    df["cmd_group_source"] = "ruleset_pending"
    return df
