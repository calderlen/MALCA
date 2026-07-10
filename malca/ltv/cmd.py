"""CMD (color-magnitude diagram) utilities for LTV candidates."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

from astropy.table import Table
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import pyvo

from malca.config import (
    CMD_R_V,
    CMD_A_G_PER_AV,
    CMD_E_BP_RP_PER_AV,
    LTV_GAIA_CHUNK_SIZE,
    LTV_WORKERS,
)
from malca.config import GAIA_AIP_TAP_URL, GAIA_ESA_TAP_URL, MIST_GRID_PATH
from malca.core.derived_stats import append_derived_features


DEFAULT_MIST_PATH = MIST_GRID_PATH
DEFAULT_CMD_COLOR_ERR_FLOOR = 0.03
DEFAULT_CMD_MAG_ERR_FLOOR = 0.03
DEFAULT_CMD_NEAREST_COLOR_SCALE = 0.08
DEFAULT_CMD_NEAREST_MAG_SCALE = 0.20


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


def _require_first_existing_column(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    col = _first_existing_column(df, candidates)
    if col is None:
        raise ValueError(f"MIST grid is missing required {label} column; tried {list(candidates)}")
    return col


def _age_column_key(age_myr: float) -> str:
    text = f"{float(age_myr):g}".replace(".", "p")
    return f"{text}myr"


def normalize_mist_cmd_grid(grid: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical Gaia CMD grid from a MIST photometry table.

    The canonical table uses intrinsic Gaia color and absolute Gaia G magnitude,
    which should be compared to dereddened observed CMD coordinates.
    """
    if grid.empty:
        return pd.DataFrame(
            columns=[
                "mist_age_myr",
                "mist_log_age_yr",
                "mist_initial_mass",
                "mist_star_mass",
                "mist_gaia_g",
                "mist_gaia_bp",
                "mist_gaia_rp",
                "mist_gaia_bp_rp",
            ]
        )

    g_col = _require_first_existing_column(grid, ["gaia_g", "Gaia_G_EDR3", "Gaia_G_DR2Rev"], "Gaia G")
    bp_col = _require_first_existing_column(grid, ["gaia_bp", "Gaia_BP_EDR3", "Gaia_BP_DR2Rev"], "Gaia BP")
    rp_col = _require_first_existing_column(grid, ["gaia_rp", "Gaia_RP_EDR3", "Gaia_RP_DR2Rev"], "Gaia RP")
    initial_mass_col = _require_first_existing_column(grid, ["initial_mass", "mist_initial_mass"], "initial mass")
    star_mass_col = _first_existing_column(grid, ["star_mass", "mist_star_mass"])
    log_age_col = _first_existing_column(grid, ["log10_isochrone_age_yr", "mist_log_age_yr", "log_age"])
    age_myr_col = _first_existing_column(grid, ["age_myr", "mist_age_myr"])

    out = pd.DataFrame(index=grid.index)
    if age_myr_col is not None:
        out["mist_age_myr"] = pd.to_numeric(grid[age_myr_col], errors="coerce")
    elif log_age_col is not None:
        log_age = pd.to_numeric(grid[log_age_col], errors="coerce")
        out["mist_age_myr"] = np.power(10.0, log_age) / 1.0e6
    else:
        raise ValueError("MIST grid is missing age_myr or log10_isochrone_age_yr")

    if log_age_col is not None:
        out["mist_log_age_yr"] = pd.to_numeric(grid[log_age_col], errors="coerce")
    else:
        out["mist_log_age_yr"] = np.log10(out["mist_age_myr"] * 1.0e6)

    out["mist_initial_mass"] = pd.to_numeric(grid[initial_mass_col], errors="coerce")
    if star_mass_col is not None:
        out["mist_star_mass"] = pd.to_numeric(grid[star_mass_col], errors="coerce")
    else:
        out["mist_star_mass"] = out["mist_initial_mass"]
    out["mist_gaia_g"] = pd.to_numeric(grid[g_col], errors="coerce")
    out["mist_gaia_bp"] = pd.to_numeric(grid[bp_col], errors="coerce")
    out["mist_gaia_rp"] = pd.to_numeric(grid[rp_col], errors="coerce")

    color_col = _first_existing_column(grid, ["gaia_bp_rp", "mist_gaia_bp_rp"])
    if color_col is not None:
        out["mist_gaia_bp_rp"] = pd.to_numeric(grid[color_col], errors="coerce")
    else:
        out["mist_gaia_bp_rp"] = out["mist_gaia_bp"] - out["mist_gaia_rp"]

    optional_cols = (
        "mist_version",
        "mesa_revision",
        "photometric_system",
        "feh",
        "alpha_fe",
        "v_vcrit",
        "eep",
        "phase",
    )
    for optional in optional_cols:
        if optional in grid.columns:
            out[optional] = grid[optional]

    required = [
        "mist_age_myr",
        "mist_initial_mass",
        "mist_star_mass",
        "mist_gaia_g",
        "mist_gaia_bp",
        "mist_gaia_rp",
        "mist_gaia_bp_rp",
    ]
    ok = np.ones(len(out), dtype=bool)
    for col in required:
        ok &= np.isfinite(pd.to_numeric(out[col], errors="coerce"))
    out = out.loc[ok].copy()
    return out.sort_values(["mist_age_myr", "mist_initial_mass"]).reset_index(drop=True)


def _resolve_grid_ages(grid: pd.DataFrame, ages_myr: Sequence[float] | None = None) -> list[float]:
    available = np.array(sorted(pd.to_numeric(grid["mist_age_myr"], errors="coerce").dropna().unique()), dtype=float)
    if available.size == 0:
        return []
    if ages_myr is None:
        return [float(age) for age in available]

    resolved: list[float] = []
    for requested in ages_myr:
        req = float(requested)
        delta = np.abs(available - req)
        idx = int(np.nanargmin(delta))
        match = float(available[idx])
        if np.isclose(match, req, rtol=5e-3, atol=1e-6) and match not in resolved:
            resolved.append(match)
    return resolved


def _nearest_isochrone_point(
    iso: pd.DataFrame,
    *,
    color: float,
    mag: float,
    color_err: float | None = None,
    mag_err: float | None = None,
    min_color_scale: float = DEFAULT_CMD_NEAREST_COLOR_SCALE,
    min_mag_scale: float = DEFAULT_CMD_NEAREST_MAG_SCALE,
) -> pd.Series | None:
    if iso.empty or not (np.isfinite(color) and np.isfinite(mag)):
        return None

    x = pd.to_numeric(iso["mist_gaia_bp_rp"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(iso["mist_gaia_g"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return None

    xerr = float(color_err) if color_err is not None and np.isfinite(color_err) and color_err > 0 else min_color_scale
    yerr = float(mag_err) if mag_err is not None and np.isfinite(mag_err) and mag_err > 0 else min_mag_scale
    xscale = max(xerr, min_color_scale)
    yscale = max(yerr, min_mag_scale)
    residual = np.sqrt(((x - color) / xscale) ** 2 + ((y - mag) / yscale) ** 2)
    residual = np.where(ok, residual, np.nan)
    if not np.isfinite(residual).any():
        return None
    idx = int(np.nanargmin(residual))
    point = iso.iloc[idx].copy()
    point["cmd_model_residual"] = float(residual[idx])
    point["cmd_model_delta_color"] = float(color - x[idx])
    point["cmd_model_delta_mag"] = float(mag - y[idx])
    return point


def estimate_cmd_masses(
    df: pd.DataFrame,
    mist_grid: pd.DataFrame,
    *,
    color_col: str = "cmd_color",
    mag_col: str = "cmd_mag",
    color_err_col: str = "cmd_color_err",
    mag_err_col: str = "cmd_mag_err",
    ages_myr: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Append nearest-isochrone mass estimates for each CMD point."""
    out = df.copy()
    grid = normalize_mist_cmd_grid(mist_grid)
    ages = _resolve_grid_ages(grid, ages_myr)
    if grid.empty or not ages:
        out["cmd_mass_source"] = "no_mist_grid"
        out["cmd_mass_best"] = np.nan
        out["cmd_mass_best_age_myr"] = np.nan
        out["cmd_mass_best_residual"] = np.nan
        return out

    masses_by_age: dict[float, list[float]] = {age: [] for age in ages}
    residuals_by_age: dict[float, list[float]] = {age: [] for age in ages}
    model_colors_by_age: dict[float, list[float]] = {age: [] for age in ages}
    model_mags_by_age: dict[float, list[float]] = {age: [] for age in ages}

    best_mass: list[float] = []
    best_age: list[float] = []
    best_residual: list[float] = []
    source: list[str] = []

    for _, row in out.iterrows():
        color = _finite_float(row.get(color_col))
        mag = _finite_float(row.get(mag_col))
        color_err = _finite_float(row.get(color_err_col))
        mag_err = _finite_float(row.get(mag_err_col))
        row_best: tuple[float, float, float] | None = None

        for age in ages:
            iso = grid[np.isclose(grid["mist_age_myr"], age, rtol=5e-3, atol=1e-6)]
            point = None
            if color is not None and mag is not None:
                point = _nearest_isochrone_point(
                    iso,
                    color=color,
                    mag=mag,
                    color_err=color_err,
                    mag_err=mag_err,
                )
            if point is None:
                mass = residual = model_color = model_mag = np.nan
            else:
                mass = float(point["mist_star_mass"])
                residual = float(point["cmd_model_residual"])
                model_color = float(point["mist_gaia_bp_rp"])
                model_mag = float(point["mist_gaia_g"])
                if row_best is None or residual < row_best[2]:
                    row_best = (mass, age, residual)

            masses_by_age[age].append(mass)
            residuals_by_age[age].append(residual)
            model_colors_by_age[age].append(model_color)
            model_mags_by_age[age].append(model_mag)

        if row_best is None:
            best_mass.append(np.nan)
            best_age.append(np.nan)
            best_residual.append(np.nan)
            source.append("missing_cmd")
        else:
            best_mass.append(row_best[0])
            best_age.append(row_best[1])
            best_residual.append(row_best[2])
            source.append("mist_nearest_isochrone")

    for age in ages:
        key = _age_column_key(age)
        out[f"cmd_mass_{key}"] = masses_by_age[age]
        out[f"cmd_mass_residual_{key}"] = residuals_by_age[age]
        out[f"cmd_model_color_{key}"] = model_colors_by_age[age]
        out[f"cmd_model_mag_{key}"] = model_mags_by_age[age]

    mass_cols = [f"cmd_mass_{_age_column_key(age)}" for age in ages]
    out["cmd_mass_best"] = best_mass
    out["cmd_mass_best_age_myr"] = best_age
    out["cmd_mass_best_residual"] = best_residual
    out["cmd_mass_min_age_grid"] = out[mass_cols].min(axis=1, skipna=True)
    out["cmd_mass_max_age_grid"] = out[mass_cols].max(axis=1, skipna=True)
    out["cmd_mass_age_grid_span"] = out["cmd_mass_max_age_grid"] - out["cmd_mass_min_age_grid"]
    out["cmd_mass_source"] = source
    return out


def mist_mass_tracks(
    mist_grid: pd.DataFrame,
    masses: Sequence[float],
    *,
    ages_myr: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Interpolate fixed-mass tracks through the available MIST isochrones."""
    grid = normalize_mist_cmd_grid(mist_grid)
    ages = _resolve_grid_ages(grid, ages_myr)
    rows: list[dict[str, float]] = []
    for mass in masses:
        target_mass = float(mass)
        for age in ages:
            iso = grid[np.isclose(grid["mist_age_myr"], age, rtol=5e-3, atol=1e-6)].copy()
            iso = iso.sort_values("mist_initial_mass")
            mass_grid = pd.to_numeric(iso["mist_initial_mass"], errors="coerce").to_numpy(dtype=float)
            color_grid = pd.to_numeric(iso["mist_gaia_bp_rp"], errors="coerce").to_numpy(dtype=float)
            mag_grid = pd.to_numeric(iso["mist_gaia_g"], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(mass_grid) & np.isfinite(color_grid) & np.isfinite(mag_grid)
            if ok.sum() < 2:
                continue
            mass_grid = mass_grid[ok]
            color_grid = color_grid[ok]
            mag_grid = mag_grid[ok]
            if target_mass < np.nanmin(mass_grid) or target_mass > np.nanmax(mass_grid):
                continue
            rows.append(
                {
                    "mass": target_mass,
                    "age_myr": float(age),
                    "gaia_bp_rp": float(np.interp(target_mass, mass_grid, color_grid)),
                    "gaia_g": float(np.interp(target_mass, mass_grid, mag_grid)),
                }
            )
    return pd.DataFrame(rows)


def cmd_uncertainty_from_fields(
    *,
    g_mag_err: object = None,
    bp_mag_err: object = None,
    rp_mag_err: object = None,
    dist_pc: object = None,
    dist_pc_err: object = None,
    parallax_mas: object = None,
    parallax_err_mas: object = None,
    a_v_3d: object = None,
    a_v_3d_err: object = None,
    a_g_per_av: float = CMD_A_G_PER_AV,
    e_bp_rp_per_av: float = CMD_E_BP_RP_PER_AV,
    color_err_floor: float = DEFAULT_CMD_COLOR_ERR_FLOOR,
    mag_err_floor: float = DEFAULT_CMD_MAG_ERR_FLOOR,
) -> dict[str, object]:
    """Propagate simple Gaia CMD uncertainties for observed and dereddened axes."""
    g_err = _finite_float(g_mag_err)
    bp_err = _finite_float(bp_mag_err)
    rp_err = _finite_float(rp_mag_err)

    phot_sources: list[str] = []
    if g_err is None or g_err <= 0:
        g_err = mag_err_floor
        phot_sources.append("g_floor")
    else:
        phot_sources.append("g_error")

    color_terms = []
    if bp_err is not None and bp_err > 0:
        color_terms.append(bp_err)
        phot_sources.append("bp_error")
    else:
        color_terms.append(color_err_floor / np.sqrt(2.0))
        phot_sources.append("bp_floor")
    if rp_err is not None and rp_err > 0:
        color_terms.append(rp_err)
        phot_sources.append("rp_error")
    else:
        color_terms.append(color_err_floor / np.sqrt(2.0))
        phot_sources.append("rp_floor")
    bp_rp_err = float(np.sqrt(np.sum(np.square(color_terms))))

    dist_mod_err = np.nan
    dist_source = "none"
    dist_f = _finite_float(dist_pc)
    dist_err_f = _finite_float(dist_pc_err)
    if dist_f is not None and dist_f > 0 and dist_err_f is not None and dist_err_f > 0:
        dist_mod_err = float(5.0 / np.log(10.0) * dist_err_f / dist_f)
        dist_source = "distance_error"
    else:
        plx_f = _finite_float(parallax_mas)
        plx_err_f = _finite_float(parallax_err_mas)
        if plx_f is not None and plx_f > 0 and plx_err_f is not None and plx_err_f > 0:
            dist_mod_err = float(5.0 / np.log(10.0) * plx_err_f / plx_f)
            dist_source = "parallax_error"

    mg_terms = [g_err]
    if np.isfinite(dist_mod_err):
        mg_terms.append(dist_mod_err)
    mg_err = float(np.sqrt(np.sum(np.square(mg_terms))))

    av_f = _finite_float(a_v_3d)
    av_err_f = _finite_float(a_v_3d_err)
    if av_f is not None and av_f >= 0 and av_err_f is not None and av_err_f > 0:
        color_dered_err = float(np.sqrt(bp_rp_err**2 + (e_bp_rp_per_av * av_err_f) ** 2))
        mag_dered_err = float(np.sqrt(mg_err**2 + (a_g_per_av * av_err_f) ** 2))
        extinction_source = "av_error"
    else:
        color_dered_err = bp_rp_err
        mag_dered_err = mg_err
        extinction_source = "no_av_error"

    return {
        "bp_rp_err": bp_rp_err,
        "mg_err": mg_err,
        "cmd_color_err": color_dered_err,
        "cmd_mag_err": mag_dered_err,
        "cmd_phot_error_source": "+".join(phot_sources),
        "cmd_distance_error_source": dist_source,
        "cmd_extinction_error_source": extinction_source,
    }


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

    Queries ``external.gaiaedr3_distance`` (available on the Gaia archive TAP)
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

    src_col = source_id_col or _first_existing_column(df, ["source_id", "gaia_id"])
    if src_col is None:
        if verbose:
            print("Warning: no source_id column found; skipping Bailer-Jones query")
        return df

    valid_mask = df[src_col].notna()
    source_ids = df.loc[valid_mask, src_col].astype(np.int64)
    if len(source_ids) == 0:
        return df

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

    def _query_chunk(tap_url: str, chunk: pd.Series) -> pd.DataFrame:
        tap = pyvo.dal.TAPService(tap_url)
        upload = Table({"source_id": chunk.values.astype(np.int64)})
        job = tap.run_sync(adql, uploads={"t": upload})
        return job.to_table().to_pandas()

    results = []
    tap_urls = [GAIA_ESA_TAP_URL, GAIA_AIP_TAP_URL]
    for tap_url in tap_urls:
        endpoint_results = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_query_chunk, tap_url, ch): ch for ch in chunks}
            it = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"BJ distances ({tap_url})",
                disable=not verbose,
            )
            for fut in it:
                try:
                    endpoint_results.append(fut.result())
                except Exception as exc:
                    if verbose:
                        tqdm.write(f"[fetch_bailer_jones_distances] {tap_url} chunk failed: {exc}")
        endpoint_results = [res for res in endpoint_results if res is not None and not res.empty]
        if endpoint_results:
            results = endpoint_results
            break
        if verbose and tap_url != tap_urls[-1]:
            print(f"[fetch_bailer_jones_distances] no usable results from {tap_url}; trying fallback TAP")

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


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def dustmaps_cmd_from_fields(
    *,
    g_mag: object = None,
    bp_rp: object = None,
    dist_pc: object = None,
    a_v_3d: object = None,
    bp_mag: object = None,
    rp_mag: object = None,
    parallax_mas: object = None,
    a_g_per_av: float = CMD_A_G_PER_AV,
    e_bp_rp_per_av: float = CMD_E_BP_RP_PER_AV,
) -> dict[str, object]:
    """Return CMD coordinates using dustmaps3d extinction when available.

    Ignores any pre-stored StarHorse ``mg0``/``bprp0``; callers pass only
    observed Gaia photometry, distance, and ``A_v_3d``.
    """
    g_f = _finite_float(g_mag)
    if g_f is None:
        return {
            "cmd_color": np.nan,
            "cmd_mag": np.nan,
            "cmd_coordinate_source": "missing",
            "bp_rp": np.nan,
            "mg": np.nan,
            "bprp0": np.nan,
            "mg0": np.nan,
        }

    bprp_f = _finite_float(bp_rp)
    if bprp_f is None:
        bp_f = _finite_float(bp_mag)
        rp_f = _finite_float(rp_mag)
        if bp_f is not None and rp_f is not None:
            bprp_f = bp_f - rp_f

    if bprp_f is None:
        return {
            "cmd_color": np.nan,
            "cmd_mag": np.nan,
            "cmd_coordinate_source": "missing",
            "bp_rp": np.nan,
            "mg": np.nan,
            "bprp0": np.nan,
            "mg0": np.nan,
        }

    dist_f = _finite_float(dist_pc)
    if dist_f is None or dist_f <= 0:
        plx_f = _finite_float(parallax_mas)
        if plx_f is not None and plx_f > 0:
            dist_f = 1000.0 / plx_f

    if dist_f is None or dist_f <= 0:
        return {
            "cmd_color": np.nan,
            "cmd_mag": np.nan,
            "cmd_coordinate_source": "missing",
            "bp_rp": bprp_f,
            "mg": np.nan,
            "bprp0": np.nan,
            "mg0": np.nan,
        }

    mg_f = g_f - 5.0 * np.log10(dist_f) + 5.0
    av_f = _finite_float(a_v_3d)
    if av_f is not None and av_f >= 0:
        bprp0_f = bprp_f - e_bp_rp_per_av * av_f
        mg0_f = mg_f - a_g_per_av * av_f
        source = "dustmaps3d" if av_f > 0 else "observed_no_extinction"
        return {
            "cmd_color": bprp0_f,
            "cmd_mag": mg0_f,
            "cmd_coordinate_source": source,
            "bp_rp": bprp_f,
            "mg": mg_f,
            "bprp0": bprp0_f,
            "mg0": mg0_f,
        }

    return {
        "cmd_color": bprp_f,
        "cmd_mag": mg_f,
        "cmd_coordinate_source": "observed_fallback",
        "bp_rp": bprp_f,
        "mg": mg_f,
        "bprp0": np.nan,
        "mg0": np.nan,
    }


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
    Compute canonical CMD quantities with optional extinction correction.

    Uses A_v_3d if present to compute:
      A_G = a_g_per_av * A_V
      E(BP-RP) = e_bp_rp_per_av * A_V

    Adds columns:
      - bp_rp, bprp0
      - mg, mg0
      - distance_pc (if derived from parallax)
    """
    if df.empty:
        return df

    df = df.copy()

    g_col = g_col or _first_existing_column(df, ["phot_g_mean_mag", "G", "g_mag"])
    bp_col = bp_col or _first_existing_column(df, ["phot_bp_mean_mag", "BP", "bp_mag"])
    rp_col = rp_col or _first_existing_column(df, ["phot_rp_mean_mag", "RP", "rp_mag"])
    bp_rp_col = _first_existing_column(df, ["bp_rp", "derived_bp_rp", "BP_RP", "gaia_bp_rp"])

    if g_col is None:
        return append_derived_features(df)
    if bp_col is None or rp_col is None:
        if bp_rp_col is None:
            return append_derived_features(df)

    # Distance in parsec — prefer Bailer-Jones photogeometric, then geometric,
    # then Gaia GSP-Phot, then a pre-existing distance_pc column.
    dist_col = distance_pc_col or _first_existing_column(
        df,
        [
            "bj_r_med_photogeo",
            "bj_r_med_geo",
            "distance_gspphot",
            "distance_pc",
        ],
    )
    parallax_col = parallax_mas_col or _first_existing_column(df, ["parallax"])

    if dist_col is not None:
        dist_pc = df[dist_col].astype(float)
        dist_pc = dist_pc.where(dist_pc > 0, np.nan)
    elif parallax_col is not None:
        plx = df[parallax_col].astype(float)
        dist_pc = pd.Series(np.where(plx > 0, 1000.0 / plx, np.nan), index=df.index)
        df["distance_pc"] = dist_pc
    else:
        return append_derived_features(df)

    g = df[g_col].astype(float)
    if bp_col is not None and rp_col is not None:
        bp = df[bp_col].astype(float)
        rp = df[rp_col].astype(float)
        df["bp_rp"] = bp - rp
    else:
        df["bp_rp"] = pd.to_numeric(df[bp_rp_col], errors="coerce")

    df["mg"] = g - 5.0 * np.log10(dist_pc) + 5.0

    # Extinction correction from dustmaps3d when available (including A_V = 0).
    if av_col in df.columns:
        av = df[av_col].astype(float).fillna(0.0)
        a_g = a_g_per_av * av
        e_bp_rp = e_bp_rp_per_av * av

        df["bprp0"] = df["bp_rp"] - e_bp_rp
        df["mg0"] = df["mg"] - a_g
        df["A_G"] = a_g
        df["E_bp_rp"] = e_bp_rp
        df["R_V"] = r_v

    return append_derived_features(df)


def assign_cmd_groups(
    df: pd.DataFrame,
    *,
    boundaries: dict | None = None,
    cmd_color_col: str = "bprp0",
    cmd_mag_col: str = "mg0",
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
